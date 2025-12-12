# 🎓 Student Success Prediction — Neural Network (PyTorch + FastAPI)

This project builds a **machine learning system** for predicting whether a student is **at risk of academic failure (target = 0)** or **likely to succeed (target = 1)**.  
The work includes:

- Complete **data preprocessing pipeline**
- **Exploratory Data Analysis (EDA)** with heatmaps and scatterplots
- **Neural Network (MLP)** model in PyTorch
- **Random Search + K-Fold Cross Validation** for hyperparameter optimization
- Two final models:
  - **High-Recall model for at-risk students**  
  - **Balanced model** with similar recall for both classes
- A production-ready **FastAPI inference service**

This document summarizes the workflow, the rationale behind design decisions, and the results.

---

## 📂 Project Structure

```
group/
│── group_project.py         # Main ML pipeline
│── api.py                   # FastAPI inference service
│── models/                  # Saved model + preprocessing artifacts
│   ├── best_grid_model.pt
│   ├── feature_columns.json
│   ├── numeric_medians.joblib
│   ├── scaler.joblib
│── outputs/                 # Auto-generated plots/EDA
│── README.md                # Documentation
```

---

# 🔍 1. Data Overview & Preprocessing

### Dataset Summary
The dataset contains **1,437 students**, with the following class distribution:

| Target | Meaning | Count |
|--------|---------|-------|
| **1** | Student likely to succeed | 1138 |
| **0** | Student at risk | 299 |

The task is therefore **highly imbalanced**, making prediction of class 0 more difficult.

---

## Missing Values
Several numeric features contained substantial missing values:

| Feature | Missing count |
|---------|----------------|
| High School Average Mark | 743 |
| Math Score | 462 |
| First/Second GPA | 17–160 |

**Solution:**  
Numeric values were imputed using **median imputation** based on the training set statistics.

---

## Categorical Encoding
All categorical variables (Funding, Gender, First Language, Residency, etc.) were **one-hot encoded**, leading to:

- **42 total model features**

One-hot encoding is also implemented in the FastAPI system using a **human-friendly schema**, allowing users to send raw categorical values.

---

## Standardization
All numeric variables (`First Term GPA`, `Second Term GPA`, `High School Average`, `Math Score`) were standardized:

```
z = (x – mean) / std
```

This improves neural network training stability.

---

# 📊 2. Exploratory Data Analysis (EDA)

All plots are exported to the `outputs/` directory.

### EDA Components
- **Correlation Heatmap** showing relationships between numerical features and target.
- **Combined scatterplots (2×2 grid)** illustrating the relationship between:
  - First Term GPA vs Target
  - Second Term GPA vs Target
  - High School Average vs Target
  - Math Score vs Target

These plots help visualize how strongly each numerical variable separates successful vs. at-risk students.

---

# 🎯 3. Why Recall(0) Was the Primary Optimization Goal

The objective of this project is **early detection of at-risk students**.  
Thus, **false negatives** (missing an at-risk student) are significantly more harmful than false positives.

This leads to prioritizing:

### **Recall for Class 0 (students at risk)**

Meaning:
> Out of all the students who *are actually at risk*, how many does the model correctly identify?

### Real-world justification  
Research in educational analytics shows that human prediction of student struggle has **20–35% error rates**, particularly early in the semester.

Achieving **0.70+ recall for class 0** therefore aligns with **human-level performance** and provides actionable value for academic intervention programs.

---

# 🧠 4. Model Development Pipeline

The project includes **three models**:

---

## A. Baseline Model
Architecture:
- Hidden layers: (32, 16)
- Dropout: 0.3
- LR: 0.001
- Epochs: 30

### Baseline Test Results

| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Recall | **0.616** | 0.798 |
| F1     | 0.517 | 0.840 |

Accuracy: **0.760**

This served as the benchmark for later improvements.

---

# 🔎 5. Random Search + K-Fold Cross Validation

**18 randomized configurations** were tested, each evaluated with **5-fold CV**, using three metrics:

- Recall for Class 0  
- F1 for Class 0  
- Accuracy  

This avoids overfitting to a single validation split.

### 🏆 Best performing configuration:
```
hidden = (16, 8)
dropout = 0.4
learning_rate = 0.0003
epochs = 12
weight_decay = 0.001
```

This had the **highest recall(0)** among all tested configurations.

---

# 🚀 6. Final Selected Model — High Recall (At-Risk Students)

### Test Set Results

| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Recall | **0.750** | 0.653 |
| F1     | 0.489 | 0.760 |

Accuracy: **0.6736**

Confusion Matrix:
```
[[ 45  15 ]
 [ 79 149 ]]
```

This model is intentionally tuned to **maximize detection of students at risk**, even at the cost of increased false positives.

---

# ⚖️ 7. Balanced Model (Equalized Recall)

A second model was trained with a configuration that historically scored evenly between classes.

### Balanced Model Test Results

| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Recall | 0.6667 | 0.7237 |
| F1     | 0.4908 | 0.7990 |

Accuracy: **0.7118**

This model is useful when **fairness between classes** is more important than maximizing recall for class 0.

---

# 🗂 8. Model Export & API Integration (FastAPI)

After training, the pipeline exports:

- `best_grid_model.pt`  
- `feature_columns.json`  
- `numeric_medians.joblib`  
- `scaler.joblib`  

These are loaded by `api.py`, which exposes:

### **GET /feature_columns**
Returns the model-required internal features.

### **POST /predict**
Accepts **human-friendly input**:

```
{
  "instances": [
    {
      "First Term GPA": 3.2,
      "Second Term GPA": 3.1,
      "High School Average": 70,
      "Math Score": 63,
      "Gender": 1,
      "Residency": 2,
      "Funding": 9,
      ...
    }
  ]
}
```

The backend performs:

- Missing value imputation  
- One-hot encoding  
- Standardization  
- Neural network inference  

---

# 🏁 9. Conclusion

This project demonstrates an end-to-end, production-ready ML system that:

- Handles real-world messy educational data  
- Trains robust neural models using best practices  
- Prioritizes recall for at-risk students (the most important stakeholder group)  
- Provides a deployable backend for live predictions  

Two models are available depending on institutional needs:

1. **High-Recall Model** → maximum detection of struggling students  
2. **Balanced Model** → fairness between predicted classes  

This project can be used for academic analysis, student intervention systems, and as a strong entry in a professional machine learning portfolio.

---

## 🚀 10. How to Run the Complete Application

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

---

### Step 1: Train the Model (if not already done)

```bash
python group_project.py
```

This will:
- Load and preprocess the student data
- Train the neural network with hyperparameter optimization
- Save the model and preprocessing artifacts to the `models/` folder

**Expected outputs in `models/` folder:**
- `best_grid_model.pt`
- `feature_columns.json`
- `numeric_medians.joblib`
- `scaler.joblib`

---

### Step 2: Start the Backend API

#### 1. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

#### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install fastapi uvicorn torch pandas scikit-learn joblib numpy pydantic
```

#### 3. Run the FastAPI server
```bash
uvicorn api:app --reload
```

The API will start at `http://localhost:8000`

#### 4. Verify the API is running
Open your browser and navigate to:
```
http://localhost:8000/docs
```

You should see the interactive API documentation (Swagger UI).

**Available Endpoints:**
- `GET /` - Health check
- `GET /feature_columns` - Get metadata about input features
- `POST /predict` - Make predictions for students
- `GET /predictions` - View recent predictions (admin panel)

---

### Step 3: Start the Frontend

#### 1. Navigate to the frontend directory
```bash
cd frontend
```

#### 2. Install dependencies (first time only)
```bash
npm install
```

#### 3. Run the development server
```bash
npm run dev
```

The frontend will start at `http://localhost:5173`

#### 4. Open the application
Open your browser and navigate to:
```
http://localhost:5173
```

---

### Step 4: Use the Application

#### User Interface Features:

1. **Student Prediction Form** (Left Panel)
   - Enter academic data (GPA, high school average, math score)
   - Select demographic information (residency, gender, funding, etc.)
   - Click "Predict Success" to get results

2. **Prediction Result** (Right Panel)
   - Shows prediction: "Likely to persist" or "At risk"
   - Displays confidence probabilities with visual bars
   - Includes important disclaimer about statistical predictions

3. **Admin Panel** (Bottom Section)
   - Click "Expand" to view recent predictions
   - Shows table with all input features and results
   - Click "Refresh" to update the list
   - Demonstrates the structured data access API

---

### API Usage Examples

#### Example 1: Make a prediction via API

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "First Term GPA": 3.2,
        "Second Term GPA": 3.1,
        "High School Average": 70,
        "Math Score": 63,
        "Residency": 1,
        "Funding": 9,
        "Age Group": 3,
        "English Grade": 7,
        "Gender": 1,
        "School": 2,
        "Fast Track": 2,
        "Coop": 1,
        "First Language": 1,
        "Prev Education": 1
      }
    ]
  }'
```

**Response:**
```json
{
  "probabilities": [0.82],
  "predictions": [1]
}
```

#### Example 2: Get recent predictions

```bash
curl "http://localhost:8000/predictions?limit=10"
```

---

### Production Build (Optional)

To create a production build of the frontend:

```bash
cd frontend
npm run build
```

The optimized build will be in the `frontend/dist` folder.

To serve it with FastAPI, you can add static file serving to `api.py`:

```python
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
```

---

### Troubleshooting

#### Backend Issues

**Error: "Models directory not found"**
- Make sure you've run `python group_project.py` first to generate the model files

**Error: "Port 8000 already in use"**
- Change the port: `uvicorn api:app --reload --port 8001`
- Update the frontend API URL accordingly (in `frontend/src/api/client.js`)

**CORS errors when frontend tries to connect**
- The API includes CORS middleware, but if issues persist, check that the backend is running on port 8000

#### Frontend Issues

**Error: "API Connection Failed"**
- Verify the backend is running at `http://localhost:8000`
- Check the console for detailed error messages
- Try accessing `http://localhost:8000/docs` directly

**Dependencies installation fails**
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`, then run `npm install` again

---

### Project Structure Summary

```
student-success-prediction/
├── group_project.py          # ML training pipeline
├── api.py                    # FastAPI backend
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Student data.csv          # Training data
├── models/                   # Trained model artifacts
│   ├── best_grid_model.pt
│   ├── feature_columns.json
│   ├── numeric_medians.joblib
│   └── scaler.joblib
├── data/                     # Predictions log
│   └── predictions_log.csv
├── outputs/                  # EDA plots
└── frontend/                 # React + Vite UI
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── StudentForm.jsx
    │   │   ├── PredictionResult.jsx
    │   │   └── AdminPanel.jsx
    │   └── api/
    │       └── client.js
    ├── package.json
    └── vite.config.js
```

---