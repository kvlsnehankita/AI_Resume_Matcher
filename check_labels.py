
import pandas as pd

file_path = "data/processed/reviewed_pairs.csv"
df = pd.read_csv(file_path)

print("Total pairs:", len(df))
print("Label counts:")
print(df["label"].value_counts(dropna=False))

df["label"] = pd.to_numeric(df["label"], errors="coerce")

labeled = df[df["label"].isin([0, 1])]

print("\nReviewed pairs:", len(labeled))
print("Positive matches:", (labeled["label"] == 1).sum())
print("Negative matches:", (labeled["label"] == 0).sum())

if labeled["label"].nunique() < 2:
    print("\nYou need both 0 and 1 labels before training.")
else:
    print("\nLabels are ready for further validation.")
