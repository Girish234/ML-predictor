# 🔧 Predictive Maintenance System — Machine Failure Classification

This project builds a machine learning model to predict **machine failure** using industrial sensor data.  
It uses data preprocessing, SMOTE balancing, and a Random Forest classifier to generate accurate predictions.

---

## 📌 Project Overview
The goal is to classify whether a machine will fail (`Target = 1`) or not (`Target = 0`) based on input sensor readings.

The workflow includes:
- Cleaning the dataset
- Handling numerical and categorical features
- Balancing the dataset using SMOTE
- Training a Random Forest Classifier
- Evaluating model performance

---

## 📁 Dataset Information
This project uses the dataset:

**`predictive_maintenance MAin1.csv`**

### **Dropped Columns (Data Leakage / Unnecessary):**
- `Failure Type`
- `Product ID`
- `UDI`

### **Feature Columns Used:**
- Air temperature \[K\]  
- Process temperature \[K\]  
- Rotational speed \[rpm\]  
- Torque \[Nm\]  
- Tool wear \[min\]  
- Type (categorical)

### **Target Column:**
- `Target` → 1 = Failure, 0 = No Failure

---

## 🧹 Preprocessing Steps
1. Load dataset using Pandas  
2. Drop leakage columns  
3. Split into **train/test** (80/20) with stratification  
4. Build preprocessing pipeline using:
   - **SimpleImputer** (median for numeric, most frequent for categorical)
   - **StandardScaler** for numeric features
   - **OneHotEncoder** for categorical features
5. Apply **SMOTE** on the preprocessed training data  
6. Preprocess test data (no SMOTE)

---

## 🤖 Model Used — Random Forest Classifier
The model is configured as:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)
