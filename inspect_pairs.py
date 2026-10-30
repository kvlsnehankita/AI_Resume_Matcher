
import pandas as pd

file_path = "data/processed/candidate_pairs_for_review.csv"

df = pd.read_csv(file_path)

print("Total pairs:", len(df))
print("\nColumns:", df.columns.tolist())

print("\nSample pairs:")
for i, row in df.head(5).iterrows():
    print(f"\nPAIR {i + 1}")
    print("Job title:", row["job_title"])
    print("Similarity:", round(row["similarity"], 4))
    print("Resume:", str(row["resume_text"])[:250])
    print("Job:", str(row["job_description"])[:250])

