# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

INPUT_CSV = "landmarks.csv"
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"{INPUT_CSV} not found. Run collect_landmarks.py first.")

df = pd.read_csv(INPUT_CSV)
X = df.drop(columns=["label"]).values
y = df["label"].values

# encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)
print("RandomForest accuracy:", acc_rf)

# MLP (with scaling)
mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=800, random_state=42))
mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, pred_mlp)
print("MLP accuracy:", acc_mlp)

# Choose best
if acc_mlp >= acc_rf:
    best = mlp
    print("Selected MLP as best model")
else:
    best = rf
    print("Selected RandomForest as best model")

print("\nClassification report for best model:")
print(classification_report(y_test, best.predict(X_test), target_names=le.classes_))

joblib.dump(best, "gesture_model.joblib")
joblib.dump(le, "label_encoder.joblib")
print("Saved gesture_model.joblib and label_encoder.joblib")
