import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load dataset (ganti dengan path ke dataset kamu)
df = pd.read_csv("data_dummy.csv")  # Pastikan kolom target bernama 'Churn'

# Pisahkan fitur dan target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Daftar kolom
numerical_cols = [
    'NumberOfAddress', 'SatisfactionScore', 'DaySinceLastOrder', 'HourSpendOnApp',
    'WarehouseToHome', 'CouponUsed', 'OrderCount', 'OrderAmountHikeFromlastYear',
    'NumberOfDeviceRegistered', 'Tenure', 'CashbackAmount'
]
categorical_cols = [
    'Complain', 'PreferredLoginDevice', 'PreferedOrderCat', 'MaritalStatus',
    'CityTier', 'Gender', 'PreferredPaymentMode'
]

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Final pipeline with classifier
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf.fit(X_train, y_train)

# Evaluate
y_pred_proba = clf.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_score:.4f}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model saved to model.pkl")
