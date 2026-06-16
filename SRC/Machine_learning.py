import pandas as pd
import os

os.makedirs(
    "Outputs/Machine_learning",
    exist_ok=True
)

df = pd.read_csv("Outputs/Feature_Engineering/engineered_dataset.csv"
)

print(df.shape)
features = [
    "age",
    "bmi",
    "socioeconomic_index",
    "comorbidity_count",
    "length_of_stay",
    "num_medications",
    "num_procedures",
    "lab_abnormality_score",
    "clinical_risk_score",
    "elderly_patient",
    "long_stay",
    "high_cost_patient"
]

target = "readmitted_flag"

from sklearn.model_selection import train_test_split

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("\nMODEL PERFORMANCE")

print("Accuracy:",
      accuracy_score(y_test, predictions, ))

print("Precision:",
      precision_score(y_test, predictions, zero_division=0))

print("Recall:",
      recall_score(y_test, predictions, zero_division=0))

print("F1 Score:",
      f1_score(y_test, predictions, zero_division=0))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "Outputs/Machine_learning/confusion_matrix.png"
)

plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

probabilities = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities
)

auc_score = roc_auc_score(
    y_test,
    probabilities
)

plt.figure(figsize=(7,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.legend()

plt.title("ROC Curve")

plt.tight_layout()

plt.savefig(
    "Outputs/Machine_learning/roc_curve.png"
)

plt.show()

importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

feature_importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

print(importance)
importance.to_csv(
    "Outputs/Machine_learning/feature_importance.csv", index=False
    )

