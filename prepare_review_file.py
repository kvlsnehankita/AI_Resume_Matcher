
import pandas as pd

source = "data/processed/candidate_pairs_for_review.csv"
output = "data/processed/reviewed_pairs.csv"

df = pd.read_csv(source)

df["label"] = ""

df.to_csv(output, index=False)

print("Review file created:", output)
print("Total pairs:", len(df))
print("Open the CSV and fill the label column:")
print("1 = relevant match")
print("0 = not a relevant match")
