
import pandas as pd

source = "data/processed/candidate_pairs_for_review.csv"
output = "data/processed/review_batch.csv"

df = pd.read_csv(source)

# Select pairs across different similarity levels
df = df.sort_values("similarity").reset_index(drop=True)

positions = [
    int(i * (len(df) - 1) / 99)
    for i in range(100)
]

review = df.iloc[positions].copy()
review["label"] = ""

review.to_csv(output, index=False)

print("Review batch created:", output)
print("Pairs to review:", len(review))
