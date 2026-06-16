import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("Data/Data/healthcare_readmission_clean_50000.csv")

os.makedirs("Outputs/EDA", exist_ok=True)

# -----------------------------
# 1. Basic overview
# -----------------------------
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nReadmission Count:")
print(df["readmitted_30days"].value_counts())


# -----------------------------
# 2. Readmission distribution
# -----------------------------
readmission_counts = df["readmitted_30days"].value_counts()

plt.figure(figsize=(7, 5))
readmission_counts.plot(kind="bar")
plt.title("Distribution of 30-Day Readmission")
plt.xlabel("Readmitted within 30 Days")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("Outputs/EDA/readmission_distribution.png")
plt.show()


# -----------------------------
# 3. Readmission rate by age group
# -----------------------------
age_readmission = (
    df.groupby("age_group")["readmitted_30days"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\nReadmission Rate by Age Group:")
print(age_readmission)

plt.figure(figsize=(8, 5))
age_readmission.plot(kind="bar")
plt.title("Readmission Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Readmission Rate (%)")
plt.tight_layout()
plt.savefig("Outputs/EDA/readmission_by_age_group.png")
plt.show()


# -----------------------------
# 4. Readmission rate by diagnosis
# -----------------------------
diagnosis_readmission = (
    df.groupby("primary_diagnosis")["readmitted_30days"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\nReadmission Rate by Diagnosis:")
print(diagnosis_readmission)

plt.figure(figsize=(10, 6))
diagnosis_readmission.plot(kind="bar")
plt.title("Readmission Rate by Primary Diagnosis")
plt.xlabel("Primary Diagnosis")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("Outputs/EDA/readmission_by_diagnosis.png")
plt.show()


# -----------------------------
# 5. Follow-up completion impact
# -----------------------------
followup_readmission = (
    df.groupby("followup_completed")["readmitted_30days"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\nReadmission Rate by Follow-up Completion:")
print(followup_readmission)

plt.figure(figsize=(7, 5))
followup_readmission.plot(kind="bar")
plt.title("Readmission Rate by Follow-up Completion")
plt.xlabel("Follow-up Completed")
plt.ylabel("Readmission Rate (%)")
plt.tight_layout()
plt.savefig("Outputs/EDA/followup_vs_readmission.png")
plt.show()


# -----------------------------
# 6. Cost comparison
# -----------------------------
cost_comparison = df.groupby("readmitted_30days")["total_hospital_cost_aud"].mean()

print("\nAverage Cost by Readmission Status:")
print(cost_comparison)

plt.figure(figsize=(7, 5))
cost_comparison.plot(kind="bar")
plt.title("Average Hospital Cost by Readmission Status")
plt.xlabel("Readmitted within 30 Days")
plt.ylabel("Average Cost AUD")
plt.tight_layout()
plt.savefig("Outputs/EDA/cost_by_readmission.png")
plt.show()


# -----------------------------
# 7. Length of stay distribution
# -----------------------------
plt.figure(figsize=(8, 5))
df["length_of_stay"].plot(kind="hist", bins=20)
plt.title("Distribution of Length of Stay")
plt.xlabel("Length of Stay Days")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("Outputs/EDA/length_of_stay_distribution.png")
plt.show()


# -----------------------------
# 8. Readmission rate by comorbidity count
# -----------------------------
comorbidity_readmission = (
    df.groupby("comorbidity_count")["readmitted_30days"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print("\nReadmission Rate by Comorbidity Count:")
print(comorbidity_readmission)

plt.figure(figsize=(8, 5))
comorbidity_readmission.plot(kind="line", marker="o")
plt.title("Readmission Rate by Comorbidity Count")
plt.xlabel("Comorbidity Count")
plt.ylabel("Readmission Rate (%)")
plt.tight_layout()
plt.savefig("Outputs/EDA/readmission_by_comorbidity_count.png")
plt.show()