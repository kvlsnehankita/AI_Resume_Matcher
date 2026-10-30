
import pandas as pd
import re
import os

input_file = "data/raw/Resume.csv"
output_file = "data/processed/cleaned_resumes.csv"

# Load dataset
df = pd.read_csv(input_file)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Keep only the required columns
df = df[["ID", "Resume_str", "Category"]].copy()

# Remove missing values
df = df.dropna(subset=["Resume_str", "Category"])

# Clean resume text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Resume_str"] = df["Resume_str"].apply(clean_text)

# Remove empty resumes and duplicate records
df = df[df["Resume_str"] != ""]
df = df.drop_duplicates(subset=["Resume_str"])

# Create processed data folder
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nDataset preparation completed!")
print("Total resumes:", len(df))
print("Total categories:", df["Category"].nunique())
print("\nCategory distribution:")
print(df["Category"].value_counts())
print("\nSaved to:", output_file)
