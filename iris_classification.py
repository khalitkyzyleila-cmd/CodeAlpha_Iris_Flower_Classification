"""
CodeAlpha Task 1 — Iris Flower Classification
Run:
    python src/iris_classification.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "iris.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# 1. Load and clean
df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]
df = df.drop(columns=["Id"], errors="ignore").dropna()

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
target = "Species"

X = df[feature_cols]
y = df[target]

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Model: scaled Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)

# 4. Evaluation
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
report = classification_report(y_test, pred, digits=3)
cm = confusion_matrix(y_test, pred)

with open(OUT / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Test accuracy: {accuracy:.3f}\n\n")
    f.write(report)

pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
).to_csv(OUT / "confusion_matrix.csv")

# 5. Confusion matrix plot
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=model.classes_, yticklabels=model.classes_
)
plt.title("Iris Classification — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=180)
plt.close()

# 6. Exploratory visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x="PetalLengthCm", y="PetalWidthCm",
    hue="Species", s=70
)
plt.title("Iris Species by Petal Measurements")
plt.tight_layout()
plt.savefig(OUT / "petal_measurements.png", dpi=180)
plt.close()

print(f"Test accuracy: {accuracy:.3%}")
print(report)
print(f"Outputs saved to: {OUT}")
