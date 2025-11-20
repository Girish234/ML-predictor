import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
df=pd.read_csv("predictive_maintenance MAin1.csv")
# ----------------------------------------------------
# 1. Drop leakage/unneeded columns
# ----------------------------------------------------
df = df.drop(columns=["Failure Type", "Product ID", "UDI"])

# ----------------------------------------------------
# 2. Feature columns
# ----------------------------------------------------
num_cols = ["Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"]

cat_cols = ["Type"]

target = "Target"

X = df.drop(columns=[target])
y = df[target]

# ----------------------------------------------------
# 3. Train-test split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------------------------------
# 4. Handle imbalance (SMOTE) - apply after preprocessing
# ----------------------------------------------------
# First, we'll preprocess and then apply SMOTE
preprocessor_fit = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ]
)

X_train_processed = preprocessor_fit.fit_transform(X_train)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_processed, y_train)

# Preprocess test data (no SMOTE on test)
X_test_processed = preprocessor_fit.transform(X_test)

# ====================================================
# 🔥 Random Forest Classifier
# ====================================================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    class_weight="balanced",   # important for imbalance
    random_state=42
)

# ----------------------------------------------------
# 6. Train Model
# ----------------------------------------------------
rf_model.fit(X_train_res, y_train_res)

# ----------------------------------------------------
# 7. Predict
# ----------------------------------------------------
y_pred_rf = rf_model.predict(X_test_processed)

# ----------------------------------------------------
# 8. Evaluate
# ----------------------------------------------------
print("Random Forest Accuracy:", rf_model.score(X_test_processed, y_test))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf))
