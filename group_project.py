import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score
import joblib
import json
import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv("Student data.csv", na_values=["?"])

# Print first 5 rows
print("First 5 rows:")
print(df.head())

# Check balance of target column (First Year Persistence Count)
if "target" in df.columns:
    print("\nTarget balance:")
    print(df["target"].value_counts(dropna=False))
else:
    print("\nThe column 'First Year Persistence Count' does not exist in the dataset.")

print("\nMissing values per column:")
print(df.isna().sum())

missing_before = df.isna().sum()  # Save missing values before imputation

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
plt.bar(missing_before.index, missing_before.values)
plt.xticks(rotation=90)
plt.title("Missing values before imputation")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "missing_values_before_imputation.png"))
plt.close()

# ============================
# Separate numeric and categorical features
# ============================

# Based on the variable description document fileciteturn1file0

numeric_features = [
    "First Term Gpa' numeric",
    "Second Term Gpa' numeric",
    "High School Average Mark' numeric",
    "Math Score' numeric"
]

categorical_features = [
    "First Language' numeric",
    "Funding numeric",
    "School numeric",
    "FastTrack numeric",
    "Coop numeric",
    "Residency numeric",
    "Gender numeric",
    "Previous Education' numeric",
    "Age Group' numeric",
    "English Grade' numeric"
]

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# ============================
# Additional Exploratory Data Analysis (EDA)
# ============================

# Correlation heatmap between numeric variables and target
if "target" in df.columns:
    try:
        plt.figure(figsize=(8, 6))
        corr_cols = numeric_features + ["target"]
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation heatmap: numeric features vs target")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "correlation_heatmap_numeric_vs_target.png"))
        plt.close()
    except Exception as e:
        print("\n[EDA] Could not compute correlation heatmap:", e)

    # Scatter plots: all numeric variables vs target in a single figure (with small jitter)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for idx, col in enumerate(numeric_features):
            ax = axes[idx]
            # Small jitter around 0 and 1 to avoid overlapping points
            jitter = (np.random.rand(len(df)) - 0.5) * 0.1
            sns.scatterplot(
                x=df["target"] + jitter,
                y=df[col],
                alpha=0.4,
                ax=ax
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["0 (at risk)", "1 (success)"])
            ax.set_xlabel("target")
            ax.set_ylabel(col)
            ax.set_title(f"{col} vs target")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "scatter_numeric_features_vs_target.png"))
        plt.close()
    except Exception as e:
        print("\n[EDA] Could not create combined scatter plot:", e)
else:
    print("\n[EDA] 'target' column not found. Skipping correlation heatmap and scatter plots.")

# ============================================
# One-hot encoding for categorical variables
# ============================================

print("\nApplying One-Hot Encoding to categorical variables...")

# Create dummy variables
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)

print("\nColumns after One-Hot Encoding:")
print(df_encoded.columns)

# Replace df with the encoded version
df = df_encoded

# Impute missing values (median for numeric columns)
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print("\nMissing values after imputation:")
print(df.isna().sum())

missing_after = df.isna().sum()  # Save missing values after imputation

plt.figure(figsize=(10,5))
plt.bar(missing_after.index, missing_after.values)
plt.xticks(rotation=90)
plt.title("Missing values after imputation")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "missing_values_after_imputation.png"))
plt.close()

print("\nNew columns:")
print(df.head())

# ============================
# Scaling numeric features
# ============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[numeric_features] = scaler.fit_transform(df[numeric_features])

print("\nNumeric features after standardization:")
print(df[numeric_features].head())

# ============================
# Train / Validation / Test split
# ============================
from sklearn.model_selection import train_test_split, StratifiedKFold

# Separar features y target
X = df.drop("target", axis=1)
y = df["target"]

print("\nTotal shape of X and y:")
print("X:", X.shape)
print("y:", y.shape)

# Primero: Train+Val vs Test (por ejemplo 80% / 20%), estratificado por target
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Segundo: Train vs Val (por ejemplo 60% / 20% del total),
# es decir, 75% train / 25% val dentro del 80% inicial.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,  # 0.25 de 0.8 = 0.2 del total
    stratify=y_train_val,
    random_state=42
)

print("\nFinal shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

print("\nTarget distribution in each split:")
print("Train:\n", y_train.value_counts(normalize=True))
print("\nVal:\n", y_val.value_counts(normalize=True))
print("\nTest:\n", y_test.value_counts(normalize=True))

# ============================
# Save preprocessing artifacts for the API
# ============================
# Save the order of feature columns (after one-hot encoding)
feature_columns = X.columns.tolist()
with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_columns, f)

# Save numeric medians used for imputation
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
numeric_medians = df[numeric_cols].median()
joblib.dump(numeric_medians, os.path.join(MODELS_DIR, "numeric_medians.joblib"))

# Save trained scaler for inference
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

print("\nPreprocessing artifacts saved in 'models' folder: feature_columns.json, numeric_medians.joblib, scaler.joblib")

# ============================
# Computation of class weights for imbalance
# ============================

# We compute weights inverse to the frequency of each class in the training set
class_counts = y_train.value_counts().sort_index()
num_classes = len(class_counts)
total_samples = class_counts.sum()

# Typical formula: weight_c = total_samples / (num_classes * count_c)
class_weights = total_samples / (num_classes * class_counts)

print("\nClass counts in TRAIN:")
print(class_counts.to_dict())

print("\nSuggested class weights (for use in PyTorch):")
print(class_weights.to_dict())

# If you later use PyTorch, you can do something like:
# import torch
# weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)
# criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

# ============================
# Prepare data for PyTorch
# ============================

# Set seeds for reproducibility
import random

SEED = 26
print(f"\nUsing global SEED = {SEED}")

random.seed(SEED)
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Determinism options (if available)
if hasattr(torch, "use_deterministic_algorithms"):
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

# Note: sklearn already uses a fixed random_state in train_test_split, so the splits are also stable.

from torch.utils.data import TensorDataset, DataLoader

# Convert DataFrames to tensors
X_train_tensor = torch.tensor(X_train.to_numpy(dtype="float32"))
X_val_tensor   = torch.tensor(X_val.to_numpy(dtype="float32"))
X_test_tensor  = torch.tensor(X_test.to_numpy(dtype="float32"))

# Convert labels (targets) to long tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_val_tensor   = torch.tensor(y_val.values, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.long)

print("\nTensor shapes:")
print("X_train_tensor:", X_train_tensor.shape)
print("y_train_tensor:", y_train_tensor.shape)
print("X_val_tensor:", X_val_tensor.shape)
print("y_val_tensor:", y_val_tensor.shape)
print("X_test_tensor:", X_test_tensor.shape)
print("y_test_tensor:", y_test_tensor.shape)

# Create TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\nDataLoaders correctly created with batch_size =", batch_size)

# ============================
# Define neural network model in PyTorch
# ============================

import torch.nn as nn
import torch.nn.functional as F

# Dimensión de entrada = número de columnas de X
input_dim = X_train_tensor.shape[1]

class StudentPersistenceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=16, dropout_p=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(hidden_dim2, 2)  # 2 classes: 0 (does not persist), 1 (persists)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)  # logits para CrossEntropyLoss
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

print("\n============================")
print("BASELINE training before Grid Search")
print("============================")

# Baseline hyperparameters (the ones we were using before)
baseline_hidden1 = 32
baseline_hidden2 = 16
baseline_dropout = 0.4
baseline_lr = 1e-3

baseline_model = StudentPersistenceNet(
    input_dim,
    hidden_dim1=baseline_hidden1,
    hidden_dim2=baseline_hidden2,
    dropout_p=baseline_dropout
).to(device)

# Criterion and optimizer baseline
weights_tensor_baseline = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
criterion_baseline = nn.CrossEntropyLoss(weight=weights_tensor_baseline)
optimizer_baseline = torch.optim.Adam(baseline_model.parameters(), lr=baseline_lr, weight_decay=1e-4)

num_epochs_baseline = 30

for epoch in range(num_epochs_baseline):
    # --- Training phase (BASELINE) ---
    baseline_model.train()
    train_loss_b = 0.0
    correct_train_b = 0
    total_train_b = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_baseline.zero_grad()
        outputs = baseline_model(X_batch)
        loss = criterion_baseline(outputs, y_batch)
        loss.backward()
        optimizer_baseline.step()

        train_loss_b += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct_train_b += (preds == y_batch).sum().item()
        total_train_b += y_batch.size(0)

    avg_train_loss_b = train_loss_b / total_train_b
    train_acc_b = correct_train_b / total_train_b

    # --- Validation phase (BASELINE) ---
    baseline_model.eval()
    val_loss_b = 0.0
    correct_val_b = 0
    total_val_b = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = baseline_model(X_batch)
            loss = criterion_baseline(outputs, y_batch)

            val_loss_b += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct_val_b += (preds == y_batch).sum().item()
            total_val_b += y_batch.size(0)

    avg_val_loss_b = val_loss_b / total_val_b
    val_acc_b = correct_val_b / total_val_b

    print(
        f"[BASELINE] Epoch {epoch+1}/{num_epochs_baseline} - "
        f"Train Loss: {avg_train_loss_b:.4f} | Train Acc: {train_acc_b:.4f} - "
        f"Val Loss: {avg_val_loss_b:.4f} | Val Acc: {val_acc_b:.4f}"
    )

# Final BASELINE metrics on Train, Val and Test
print("\n===== BASELINE metrics (before Grid Search) =====")
def evaluate_split(model_to_eval, loader, name):
    model_to_eval.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model_to_eval(X_batch)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    recall_0 = recall_score(all_labels, all_preds, pos_label=0)
    recall_1 = recall_score(all_labels, all_preds, pos_label=1)

    f1_0 = f1_score(all_labels, all_preds, pos_label=0)
    f1_1 = f1_score(all_labels, all_preds, pos_label=1)

    acc = accuracy_score(all_labels, all_preds)

    print(f"\n--- Metrics for {name} ---")
    print(f"Recall (Clase 0): {recall_0:.4f}")
    print(f"Recall (Clase 1): {recall_1:.4f}")
    print(f"F1-score (Clase 0): {f1_0:.4f}")
    print(f"F1-score (Clase 1): {f1_1:.4f}")
    print(f"Accuracy: {acc:.4f}")

evaluate_split(baseline_model, train_loader, "BASELINE TRAIN")
evaluate_split(baseline_model, val_loader, "BASELINE VALIDATION")
evaluate_split(baseline_model, test_loader, "BASELINE TEST")

# ============================
# Random Search of hyperparameters with K-Fold (using TRAIN with CV to choose best model)
# ============================

print("\n============================")
print("Starting Random Search of hyperparameters with K-Fold (using TRAIN with CV to choose best model)")
print("============================")

# Search spaces
hidden_configs = [
    (16, 8),
    (32, 16),
    (64, 32),
]

dropout_values = [0.3, 0.4, 0.5]
learning_rates = [1e-2, 1e-3, 3e-4]
epoch_options = [8, 12, 20]
weight_decay_values = [0.0, 1e-4, 1e-3]

# Number of random configurations to evaluate
n_random_configs = 18

# Define stratified K-Fold on TRAIN
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

grid_results = []

for i in range(n_random_configs):
    # Randomly choose a combination of hyperparameters
    h1, h2 = random.choice(hidden_configs)
    drop_p = random.choice(dropout_values)
    lr = random.choice(learning_rates)
    num_epochs_gs = random.choice(epoch_options)
    wd = random.choice(weight_decay_values)

    print(
        f"\n>>> Random config {i+1}/{n_random_configs}: "
        f"hidden=({h1},{h2}), dropout={drop_p}, lr={lr}, epochs={num_epochs_gs}, weight_decay={wd}"
    )

    fold_recalls = []
    fold_f1s = []
    fold_accs = []

    # K-Fold on TRAIN
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # Extract data for this fold
        X_tr_fold = X_train.iloc[train_idx].to_numpy(dtype="float32")
        y_tr_fold = y_train.iloc[train_idx].values
        X_val_fold = X_train.iloc[val_idx].to_numpy(dtype="float32")
        y_val_fold = y_train.iloc[val_idx].values

        # Create tensors
        X_tr_tensor = torch.tensor(X_tr_fold)
        y_tr_tensor = torch.tensor(y_tr_fold, dtype=torch.long)
        X_val_tensor_k = torch.tensor(X_val_fold)
        y_val_tensor_k = torch.tensor(y_val_fold, dtype=torch.long)

        # DataLoaders for this fold
        train_dataset_k = TensorDataset(X_tr_tensor, y_tr_tensor)
        val_dataset_k = TensorDataset(X_val_tensor_k, y_val_tensor_k)

        train_loader_k = DataLoader(train_dataset_k, batch_size=batch_size, shuffle=True)
        val_loader_k = DataLoader(val_dataset_k, batch_size=batch_size, shuffle=False)

        # Create a new model for this config and fold
        model_gs = StudentPersistenceNet(
            input_dim,
            hidden_dim1=h1,
            hidden_dim2=h2,
            dropout_p=drop_p
        ).to(device)

        # We use the same class weights as before (computed on global TRAIN)
        weights_tensor_gs = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
        criterion_gs = nn.CrossEntropyLoss(weight=weights_tensor_gs)
        optimizer_gs = torch.optim.Adam(model_gs.parameters(), lr=lr, weight_decay=wd)

        # Quick training on TRAIN for this fold
        for epoch in range(num_epochs_gs):
            model_gs.train()
            for X_batch, y_batch in train_loader_k:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer_gs.zero_grad()
                outputs = model_gs(X_batch)
                loss = criterion_gs(outputs, y_batch)
                loss.backward()
                optimizer_gs.step()

        # Evaluation on VALIDATION (of the fold) for this configuration
        model_gs.eval()
        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader_k:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model_gs(X_batch)
                _, preds = torch.max(outputs, dim=1)

                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(y_batch.cpu().numpy())

        recall_0_val = recall_score(all_labels_val, all_preds_val, pos_label=0)
        f1_0_val = f1_score(all_labels_val, all_preds_val, pos_label=0)
        acc_val = accuracy_score(all_labels_val, all_preds_val)

        fold_recalls.append(recall_0_val)
        fold_f1s.append(f1_0_val)
        fold_accs.append(acc_val)

    # Average metrics over the K folds
    mean_recall_0 = float(np.mean(fold_recalls))
    mean_f1_0 = float(np.mean(fold_f1s))
    mean_acc = float(np.mean(fold_accs))

    print(
        f"Resultado K-Fold VALIDATION (promedio {k_folds} folds) -> "
        f"Recall_0: {mean_recall_0:.4f}, F1_0: {mean_f1_0:.4f}, Acc: {mean_acc:.4f}"
    )

    grid_results.append({
        "hidden1": h1,
        "hidden2": h2,
        "dropout": drop_p,
        "lr": lr,
        "epochs": num_epochs_gs,
        "weight_decay": wd,
        "val_recall_0": mean_recall_0,
        "val_f1_0": mean_f1_0,
        "val_acc": mean_acc,
    })

# Order results by Recall of class 0 (and then F1) on K-Fold validation
grid_results_sorted = sorted(
    grid_results,
    key=lambda r: (r["val_recall_0"], r["val_f1_0"]),
    reverse=True,
)

print("\n===== Top 5 configurations according to VALIDATION K-Fold (sorted by Recall_0 and F1_0) =====")
for i, res in enumerate(grid_results_sorted[:5], start=1):
    print(
        f"Top {i}: hidden=({res['hidden1']},{res['hidden2']}), dropout={res['dropout']}, "
        f"lr={res['lr']}, epochs={res['epochs']}, weight_decay={res['weight_decay']}, "
        f"Recall_0={res['val_recall_0']:.4f}, F1_0={res['val_f1_0']:.4f}, Acc={res['val_acc']:.4f}"
    )

# Choose the best configuration for final training
best_config = grid_results_sorted[0]
best_hidden1 = best_config["hidden1"]
best_hidden2 = best_config["hidden2"]
best_dropout = best_config["dropout"]
best_lr = best_config["lr"]
best_epochs = best_config["epochs"]
best_weight_decay = best_config["weight_decay"]

print(
    f"\nBest configuration selected for final training (according to K-Fold on TRAIN): "
    f"hidden=({best_hidden1},{best_hidden2}), dropout={best_dropout}, "
    f"lr={best_lr}, epochs={best_epochs}, weight_decay={best_weight_decay}"
)

# Instantiate the final model with the best configuration found by Random Search
model = StudentPersistenceNet(
    input_dim,
    hidden_dim1=best_hidden1,
    hidden_dim2=best_hidden2,
    dropout_p=best_dropout
).to(device)

# Criterion using class weights
weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# Use Adam with L2 (weight decay) and lr chosen by Random Search
optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)

print("\nOptimizer configured with weight_decay (L2) =", best_weight_decay, "and lr =", best_lr)

# ============================
# Training loop with validation
# ============================

num_epochs = best_epochs
print(f"\nTraining final model with num_epochs = {num_epochs}")

for epoch in range(num_epochs):
    # --- Training phase ---
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)

    avg_train_loss = train_loss / total_train
    train_acc = correct_train / total_train

    # --- Validation phase ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct_val += (preds == y_batch).sum().item()
            total_val += y_batch.size(0)

    avg_val_loss = val_loss / total_val
    val_acc = correct_val / total_val

    print(
        f"Epoch {epoch+1}/{num_epochs} - "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} - "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

# ============================
# Final evaluation on the test set
# ============================

model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        test_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct_test += (preds == y_batch).sum().item()
        total_test += y_batch.size(0)

avg_test_loss = test_loss / total_test
test_acc = correct_test / total_test

print("\nResults on the TEST set:")
print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f}")

# ============================
# Confusion matrix and Classification Report
# ============================

from sklearn.metrics import confusion_matrix, classification_report

 # Get predictions on test
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)  # Compute confusion matrix
print("\nConfusion Matrix (Test):")
print(cm)

print("\nClassification Report (Test):")
print(classification_report(all_labels, all_preds, digits=4))  # Classification report with F1, Recall, Precision

# ============================
# Metrics (Recall, F1 and Accuracy) for Train, Val and Test
# ============================

from sklearn.metrics import recall_score, f1_score, accuracy_score

# Evaluate on Train, Val and Test using the final model
evaluate_split(model, train_loader, "TRAIN")
evaluate_split(model, val_loader, "VALIDATION")
evaluate_split(model, test_loader, "TEST")

# Save the final Grid Search model (to serve in the API)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_grid_model.pt"))
print("\nGrid Search model saved to 'models/best_grid_model.pt' (this is the recommended model for the API).")

# ==========================================================
# Training and evaluation of the "Balanced Model" (Top 2 K-Fold config)
# ==========================================================
print("\n============================")
print("BALANCED MODEL training (centered config)")
print("============================")

# Hyperparameters of the balanced model (chosen from Top 2 of K-Fold)
balanced_hidden1, balanced_hidden2 = 16, 8
balanced_dropout = 0.5
balanced_lr = 0.001
balanced_epochs = 8
balanced_weight_decay = 0.001

balanced_model = StudentPersistenceNet(
    input_dim,
    hidden_dim1=balanced_hidden1,
    hidden_dim2=balanced_hidden2,
    dropout_p=balanced_dropout
).to(device)

# Criterion and optimizer for the balanced model
weights_tensor_bal = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
criterion_bal = nn.CrossEntropyLoss(weight=weights_tensor_bal)
optimizer_bal = torch.optim.Adam(
    balanced_model.parameters(),
    lr=balanced_lr,
    weight_decay=balanced_weight_decay
)

print(
    f"\nBalanced config -> hidden=({balanced_hidden1},{balanced_hidden2}), "
    f"dropout={balanced_dropout}, lr={balanced_lr}, "
    f"epochs={balanced_epochs}, weight_decay={balanced_weight_decay}"
)

# Training loop for the balanced model
for epoch in range(balanced_epochs):
    balanced_model.train()
    train_loss_bal = 0.0
    correct_train_bal = 0
    total_train_bal = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_bal.zero_grad()
        outputs = balanced_model(X_batch)
        loss = criterion_bal(outputs, y_batch)
        loss.backward()
        optimizer_bal.step()

        train_loss_bal += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct_train_bal += (preds == y_batch).sum().item()
        total_train_bal += y_batch.size(0)

    avg_train_loss_bal = train_loss_bal / total_train_bal
    train_acc_bal = correct_train_bal / total_train_bal

    # Validation with the balanced model
    balanced_model.eval()
    val_loss_bal = 0.0
    correct_val_bal = 0
    total_val_bal = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = balanced_model(X_batch)
            loss = criterion_bal(outputs, y_batch)

            val_loss_bal += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct_val_bal += (preds == y_batch).sum().item()
            total_val_bal += y_batch.size(0)

    avg_val_loss_bal = val_loss_bal / total_val_bal
    val_acc_bal = correct_val_bal / total_val_bal

    print(
        f"[BALANCED] Epoch {epoch+1}/{balanced_epochs} - "
        f"Train Loss: {avg_train_loss_bal:.4f} | Train Acc: {train_acc_bal:.4f} - "
        f"Val Loss: {avg_val_loss_bal:.4f} | Val Acc: {val_acc_bal:.4f}"
    )

print("\n===== BALANCED MODEL metrics =====")
evaluate_split(balanced_model, train_loader, "BALANCED TRAIN")
evaluate_split(balanced_model, val_loader, "BALANCED VALIDATION")
evaluate_split(balanced_model, test_loader, "BALANCED TEST")
