"""
Direct Python port of main.R's logistic regression pipeline.

R steps this mirrors:
  df$Amount = scale(df$Amount)          -> StandardScaler on Amount only
  data1 = df[,-c(1)]                    -> drop the Time column
  sample.split(Class, SplitRatio=0.80)  -> stratified 80/20 train/test split
  glm(Class ~ ., family="binomial")     -> LogisticRegression

BEFORE RUNNING:
  1. Download creditcard.csv (Kaggle "Credit Card Fraud Detection" dataset)
  2. Place it in this same `model/` folder
  3. Run: pip install pandas scikit-learn joblib
  4. Run: python train_model.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"creditcard.csv not found at {CSV_PATH}.\n"
        "Download it from the Kaggle 'Credit Card Fraud Detection' dataset "
        "and place it in the model/ folder before running this script."
    )

# --- Load & clean (mirrors: read.csv, na.omit) ---
df = pd.read_csv(CSV_PATH)
df = df.dropna()

# --- Drop Time column (mirrors: data1 = df[,-c(1)]) ---
if "Time" in df.columns:
    df = df.drop(columns=["Time"])

# --- Scale Amount only (mirrors: df$Amount = scale(df$Amount)) ---
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

feature_cols = [c for c in df.columns if c != "Class"]
X = df[feature_cols]
y = df["Class"]

# --- 80/20 stratified split (mirrors: sample.split(..., SplitRatio = 0.80)) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=123
)

# --- Logistic regression (mirrors: glm(Class ~ ., family = "binomial")) ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Evaluate (mirrors: accuracy + AUC printed in the R script) ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy =", accuracy_score(y_test, y_pred))
print("AUC =", roc_auc_score(y_test, y_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump({
    "model": model,
    "scaler": scaler,
    "feature_cols": feature_cols,   # preserves exact column order for inference
}, os.path.join(os.path.dirname(__file__), "..", "api", "model.pkl"))

print("Saved model to api/model.pkl")
