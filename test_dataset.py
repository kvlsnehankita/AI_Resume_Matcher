
import pandas as pd

file_path = "data/processed/cleaned_resumes.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of categories:", df["Category"].nunique())

print("\nSample resume:")
print(df["Resume_str"].iloc[0][:500])

print("\nDataset verification successful!")
