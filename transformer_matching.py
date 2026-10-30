
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

RESUME_FILE = "data/processed/cleaned_resumes.csv"
JOB_FILE = "data/processed/cleaned_jobs.csv"

resumes = pd.read_csv(RESUME_FILE)
jobs = pd.read_csv(JOB_FILE)

# Select the first resume for testing
resume_text = str(resumes.iloc[0]["resume_text"])

# Remove missing and duplicate job descriptions
jobs = jobs.dropna(subset=["job_text"]).copy()
jobs["job_text"] = jobs["job_text"].astype(str)

jobs = jobs.drop_duplicates(
    subset=["Job Title", "job_text"]
).reset_index(drop=True)

# Use a manageable sample for the first test
jobs = jobs.head(5000).copy()

print("Unique jobs being tested:", len(jobs))

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

resume_embedding = model.encode(
    [resume_text],
    normalize_embeddings=True
)

job_embeddings = model.encode(
    jobs["job_text"].tolist(),
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

scores = cosine_similarity(
    resume_embedding,
    job_embeddings
)[0]

jobs["semantic_similarity"] = scores

top_jobs = jobs.sort_values(
    "semantic_similarity",
    ascending=False
).head(10)

print("\nTop 10 Unique Transformer Job Matches:\n")

for _, row in top_jobs.iterrows():
    print("Job title:", row["Job Title"])
    print("Similarity:", round(row["semantic_similarity"], 4))
    print("Description:", row["job_description"][:200])
    print("-" * 60)

