
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

RESUME_FILE = "data/processed/cleaned_resumes.csv"
JOB_FILE = "data/processed/cleaned_jobs.csv"
OUTPUT_FILE = "data/processed/automatic_job_matches.csv"

print("Loading datasets...")

resumes = pd.read_csv(RESUME_FILE)
jobs = pd.read_csv(JOB_FILE)

# Remove duplicate resumes and jobs
resumes = resumes.dropna(subset=["resume_text"]).copy()
jobs = jobs.dropna(subset=["job_text", "Job Title"]).copy()

jobs = jobs.drop_duplicates(
    subset=["Job Title", "job_text"]
).reset_index(drop=True)

# Use the first resume for the initial test
resume = resumes.iloc[0]
resume_text = str(resume["resume_text"])

print("Resume selected:", resume.get("category", "Unknown"))
print("Unique jobs:", len(jobs))

# Load pretrained transformer
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Convert resume and jobs into embeddings
resume_embedding = model.encode(
    [resume_text],
    normalize_embeddings=True
)

job_embeddings = model.encode(
    jobs["job_text"].astype(str).tolist(),
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Calculate semantic similarity
scores = cosine_similarity(
    resume_embedding,
    job_embeddings
)[0]

jobs["similarity_score"] = scores

# Display the top 10 jobs
top_jobs = jobs.nlargest(
    10, "similarity_score"
).copy()

top_jobs["similarity_score"] = (
    top_jobs["similarity_score"].round(4)
)

# Save readable results
top_jobs[
    ["Job Title", "job_description", "skills", "similarity_score"]
].to_csv(OUTPUT_FILE, index=False)

print("\nTOP 10 JOB MATCHES")

for rank, (_, row) in enumerate(top_jobs.iterrows(), start=1):
    print(f"\n{rank}. {row['Job Title']}")
    print("Similarity:", row["similarity_score"])
    print("Skills:", str(row["skills"])[:200])
    print("Description:", str(row["job_description"])[:250])

print("\nResults saved to:", OUTPUT_FILE)
