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

## 🚀 10. How to Run the FastAPI Backend

### 1. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Ensure the `models/` folder contains:
- `best_grid_model.pt`
- `feature_columns.json`
- `numeric_medians.joblib`
- `scaler.joblib`

(These are created automatically when you run `group_project.py`.)

### 4. Run the FastAPI server
```bash
uvicorn api:app --reload
```

### 5. Open the interactive docs (Swagger UI)
Navigate to:
```
http://127.0.0.1:8000/docs
```

### 6. Test predictions
Use the **POST /predict** endpoint with a JSON body like:

```json
{
  "instances": [
    {
      "First Term GPA": 2.8,
      "Second Term GPA": 2.5,
      "High School Average": 67,
      "Math Score": 55,
      "Gender": 2,
      "Residency": 1,
      "Funding": 9,
      "Age Group": 3,
      "English Grade": 7
    }
  ]
}
```

The system will:
- Validate the input,
- Convert categorical features,
- Apply scaling and preprocessing,
- Run the neural network model,
- Return the predicted `target` and model confidence.

---