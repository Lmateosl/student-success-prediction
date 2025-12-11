import os
import json
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------
# Paths and constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
NUMERIC_MEDIANS_PATH = os.path.join(MODELS_DIR, "numeric_medians.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "best_grid_model.pt")  # model recommended for the API

# -----------------------------
# Load artifacts at startup
# -----------------------------
if not os.path.exists(MODELS_DIR):
    raise RuntimeError(f"Models directory not found: {MODELS_DIR}")

if not os.path.exists(FEATURE_COLUMNS_PATH):
    raise RuntimeError("feature_columns.json not found in models directory.")

if not os.path.exists(NUMERIC_MEDIANS_PATH):
    raise RuntimeError("numeric_medians.joblib not found in models directory.")

if not os.path.exists(SCALER_PATH):
    raise RuntimeError("scaler.joblib not found in models directory.")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("best_grid_model.pt not found in models directory.")

# Load feature columns (order matters)
with open(FEATURE_COLUMNS_PATH, "r") as f:
    FEATURE_COLUMNS: List[str] = json.load(f)

numeric_medians: Dict[str, float] = joblib.load(NUMERIC_MEDIANS_PATH)

# Load scaler (fitted on a specific subset of numeric columns during training)
scaler = joblib.load(SCALER_PATH)

# Numeric columns actually used by the scaler.
# Prefer the names stored inside the scaler (feature_names_in_),
# and fall back to numeric_medians filtered by FEATURE_COLUMNS.
if hasattr(scaler, "feature_names_in_"):
    NUMERIC_COLS = list(scaler.feature_names_in_)
else:
    NUMERIC_COLS = [col for col in numeric_medians.keys() if col in FEATURE_COLUMNS]


# -----------------------------
# Categorical encoding helpers
# -----------------------------
# We allow clients to send *raw* codes for categorical variables
# (e.g. "English Grade": 3, "Funding": 9) and convert them
# internally into the one-hot encoded feature columns that the
# trained model expects.

# Mapping from human-friendly raw field name (as in the dataset
# description) to the prefix used in FEATURE_COLUMNS for the
# corresponding one-hot encoded columns.
RAW_CATEGORICAL_PREFIXES: Dict[str, str] = {
    # From Description of Variables.docx
    "First Language": "First Language' numeric_",  # values 1,2,3
    "Funding": "Funding numeric_",                 # values 1..9 (only some may appear)
    "School": "School numeric_",                  # values 1..7 (only some may appear)
    "Fast Track": "FastTrack numeric_",           # 1 = Y, 2 = N
    "Coop": "Coop numeric_",                      # 1 = Y, 2 = N
    "Residency": "Residency numeric_",            # 1 = Domestic, 2 = International
    "Gender": "Gender numeric_",                  # 1 = Female, 2 = Male, 3 = Neutral
    "Prev Education": "Previous Education' numeric_",  # 1 = HighSchool, 2 = PostSecondary (plus 0 if present)
    "Age Group": "Age Group' numeric_",           # 1..10 (only some may appear)
    "English Grade": "English Grade' numeric_",   # 1..11 (only some may appear)
}

# For each raw categorical field, build a mapping from the
# integer code (e.g. 1,2,3) to the full one-hot column name
# present in FEATURE_COLUMNS.
RAW_CAT_VALUE_TO_COLUMN: Dict[str, Dict[int, str]] = {}

for raw_name, prefix in RAW_CATEGORICAL_PREFIXES.items():
    value_to_col: Dict[int, str] = {}
    for col in FEATURE_COLUMNS:
        if col.startswith(prefix):
            # Suffix after the prefix encodes the category, e.g. "1.0" or "1"
            suffix = col[len(prefix):]
            try:
                code = int(float(suffix))
            except ValueError:
                # If parsing fails, skip this column
                continue
            value_to_col[code] = col
    if value_to_col:
        RAW_CAT_VALUE_TO_COLUMN[raw_name] = value_to_col


def expand_raw_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Expand raw categorical codes into one-hot columns.

    If the input DataFrame has columns like "English Grade", "Funding",
    etc. (with integer codes as defined in the project documentation),
    this function creates/updates the corresponding one-hot columns in
    the FEATURE_COLUMNS space and drops the raw columns.

    If a particular raw field is not present in the DataFrame, it is
    simply ignored.
    """
    df = df.copy()

    for raw_name, value_to_col in RAW_CAT_VALUE_TO_COLUMN.items():
        if raw_name not in df.columns:
            continue

        # Ensure all one-hot columns for this categorical are present
        # and initialized to 0 (unless already provided explicitly).
        for col in value_to_col.values():
            if col not in df.columns:
                df[col] = 0

        # Convert raw codes to integers (invalid / missing -> NaN)
        codes = pd.to_numeric(df[raw_name], errors="coerce").astype("Int64")

        # For each possible code, set the corresponding one-hot column to 1
        for code, col in value_to_col.items():
            mask = codes == code
            if mask.any():
                df.loc[mask, col] = 1

        # Drop the raw categorical column, since the model does not use it directly
        df = df.drop(columns=[raw_name])

    return df


# -----------------------------
# Model definition (must match training)
# -----------------------------
class StudentMLP(nn.Module):
    """Simple MLP with two hidden layers.

    This architecture must match the one used when training
    the model whose weights are stored in best_grid_model.pt.
    Here we assume hidden sizes (16, 8) with dropout 0.4,
    as in the grid-search best configuration.

    Note: Layer names (fc1, fc2, out) are chosen to match the
    keys in the saved state_dict (fc1.weight, fc2.weight, out.weight, ...).

    Output is a vector of two logits per instance (before softmax), one per class.
    """

    def __init__(self, input_dim: int, hidden_dims=(16, 8), dropout: float = 0.4):
        super().__init__()
        h1, h2 = hidden_dims

        # Linear layers
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, 2)

        # Non-linearities and dropout (no trainable params, so not in state_dict)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Output is a vector of two logits per instance (before softmax).
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.out(x)
        return x


# Instantiate model and load weights
INPUT_DIM = len(FEATURE_COLUMNS)
model = StudentMLP(input_dim=INPUT_DIM, hidden_dims=(16, 8), dropout=0.4)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Student Risk Prediction API",
    description=(
        "API to predict student risk / dropout using a trained neural network. "
        "It expects features already aligned with the training feature space."
    ),
    version="1.0.0",
)


# -----------------------------
# Pydantic models
# -----------------------------
class StudentFeatures(BaseModel):
    """
    Features for a single student in a human-friendly format.

    Numeric fields use the original dataset names (with quotes) as JSON keys via aliases.
    Categorical fields are provided as integer codes as described in the project
    documentation (e.g. "English Grade": 1..11) and will be expanded internally to
    one-hot encoded columns.
    """

    # Core numeric features (use original column names as JSON keys)
    first_term_gpa: float = Field(..., alias="First Term Gpa' numeric")
    second_term_gpa: float = Field(..., alias="Second Term Gpa' numeric")
    high_school_average_mark: float = Field(..., alias="High School Average Mark' numeric")
    math_score: float = Field(..., alias="Math Score' numeric")

    # Raw categorical codes (optional – if missing, they will be treated as unknown / 0)
    first_language: int | None = Field(None, alias="First Language")
    funding: int | None = Field(None, alias="Funding")
    school: int | None = Field(None, alias="School")
    fast_track: int | None = Field(None, alias="Fast Track")
    coop: int | None = Field(None, alias="Coop")
    residency: int | None = Field(None, alias="Residency")
    gender: int | None = Field(None, alias="Gender")
    prev_education: int | None = Field(None, alias="Prev Education")
    age_group: int | None = Field(None, alias="Age Group")
    english_grade: int | None = Field(None, alias="English Grade")


class PredictRequest(BaseModel):
    """Request body for predictions.

    Each instance is a StudentFeatures object, which is then converted to
    the internal FEATURE_COLUMNS representation before being passed to the model.
    """

    instances: List[StudentFeatures]


class PredictResponse(BaseModel):
    """Response with probabilities and class predictions."""

    probabilities: List[float]
    predictions: List[int]


class CategoricalFieldInfo(BaseModel):
    """Description of a raw categorical field and its allowed integer codes."""

    raw_name: str
    valid_codes: Dict[int, str]


class FeatureColumnsResponse(BaseModel):
    """Metadata about the model's feature space."""

    # Internal columns actually used by the model (one-hot + numeric)
    feature_columns: List[str]
    # Numeric columns that are scaled
    numeric_columns: List[str]
    # Raw categorical fields that can be sent by clients and their valid codes
    raw_categorical_fields: List[CategoricalFieldInfo]


# -----------------------------
# Helper: prepare features
# -----------------------------
def prepare_features(instances: List[Dict[str, float]]) -> torch.Tensor:
    """Convert raw instance dicts into a Torch tensor ready for the model.

    Steps:
    1. Build a DataFrame from the list of dicts.
    2. For numeric columns, fill missing values with training medians.
    3. Ensure all expected FEATURE_COLUMNS exist (missing ones are set to 0).
    4. Order columns according to FEATURE_COLUMNS.
    5. Scale only the numeric columns using the stored scaler.
    6. Convert to float32 tensor.
    """

    df = pd.DataFrame(instances)

    # Expand raw categorical codes (e.g. "English Grade": 3) into
    # one-hot encoded feature columns expected by the model.
    df = expand_raw_categoricals(df)

    # Impute numeric columns with training medians
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(numeric_medians[col])
        else:
            # If the numeric column is missing, fill with its median
            df[col] = numeric_medians[col]

    # Ensure all expected feature columns exist; if not, create with zeros
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    # Keep only columns in the correct order
    df = df[FEATURE_COLUMNS]

    # Scale only numeric columns (scaler was trained on NUMERIC_COLS)
    # Use a DataFrame so feature names match those seen by the scaler
    num_df = df[NUMERIC_COLS].astype(float)
    num_scaled = scaler.transform(num_df)

    # Put scaled numeric features back into a copy of df
    df_scaled = df.copy()
    df_scaled[NUMERIC_COLS] = num_scaled

    # Convert to torch tensor in the correct column order
    X_tensor = torch.tensor(df_scaled[FEATURE_COLUMNS].values, dtype=torch.float32)
    return X_tensor


# -----------------------------
# Routes
# -----------------------------
@app.get("/feature_columns", response_model=FeatureColumnsResponse)
def get_feature_columns():
    """Return metadata about the feature space.

    Includes:
    - internal feature_columns actually used by the model,
    - numeric_columns that are scaled by the StandardScaler,
    - raw_categorical_fields with their allowed integer codes and
      the corresponding internal one-hot column names.
    """

    raw_cat_fields: List[CategoricalFieldInfo] = []
    for raw_name, value_to_col in RAW_CAT_VALUE_TO_COLUMN.items():
        raw_cat_fields.append(
            CategoricalFieldInfo(
                raw_name=raw_name,
                valid_codes=value_to_col,
            )
        )

    return FeatureColumnsResponse(
        feature_columns=FEATURE_COLUMNS,
        numeric_columns=NUMERIC_COLS,
        raw_categorical_fields=raw_cat_fields,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict for one or multiple instances.

    The same route works for a single instance (length 1 list)
    or a batch of instances. Each instance is a StudentFeatures object
    with human-friendly fields (including raw categorical codes).
    You do not need to send all features; missing numeric features are
    imputed with their training medians, and other missing features are
    set to 0 / treated as unknown.
    """

    # Convert Pydantic models to plain dicts using JSON aliases as keys
    instance_dicts: List[Dict[str, float]] = [
        inst.model_dump(by_alias=True, exclude_unset=True) for inst in request.instances
    ]

    # Prepare features
    X_tensor = prepare_features(instance_dicts)

    # Run model in evaluation mode (no gradients)
    with torch.no_grad():
        logits = model(X_tensor)  # shape: (batch_size, 2)
        # Apply softmax along the class dimension and take probability of class 1
        probs_tensor = torch.softmax(logits, dim=1)[:, 1]
        probs = probs_tensor.cpu().numpy().tolist()

    # Default threshold = 0.5 on probability of class 1
    preds = [1 if p >= 0.5 else 0 for p in probs]

    return PredictResponse(probabilities=probs, predictions=preds)


@app.get("/")
def root():
    """Simple health/metadata endpoint."""

    return {
        "message": "Student Risk Prediction API is running",
        "n_features": len(FEATURE_COLUMNS),
        "model_path": MODEL_PATH,
    }
