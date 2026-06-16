import pandas as pd

df = pd.read_csv("Data/Data/healthcare_readmission_clean_50000.csv")

print(df.head())
df["readmitted_flag"] = df["readmitted_30days"].map({
    "Yes":1,
    "No":0
})

df["clinical_risk_score"] = (
    df["comorbidity_count"] +
    (df["length_of_stay"] / 5) +
    (df["lab_abnormality_score"] / 20)
)

df["elderly_patient"] = (
    df["age"] >= 65
).astype(int)

df["long_stay"] = (
    df["length_of_stay"] >= 7
).astype(int)

cost_threshold = df["total_hospital_cost_aud"].median()

df["high_cost_patient"] = (
    df["total_hospital_cost_aud"] >= cost_threshold
).astype(int)

import os

os.makedirs(
    "Outputs/Feature_Engineering",
    exist_ok=True
)

df.to_csv(
    "Outputs/Feature_Engineering/engineered_dataset.csv",
    index=False
)