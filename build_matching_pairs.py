
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumes = pd.read_csv(
    "data/processed/cleaned_resumes.csv"
)

jobs = pd.read_csv(
    "data/processed/cleaned_jobs.csv"
)

# Limit the initial experiment to keep it manageable
resumes = resumes.head(500).copy()
jobs = jobs.head(200).copy()

# Compare resume text with job text
all_text = (
    resumes["resume_text"].tolist()
    + jobs["job_text"].tolist()
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

vectors = vectorizer.fit_transform(all_text)

resume_vectors = vectors[:len(resumes)]
job_vectors = vectors[len(resumes):]

similarities = cosine_similarity(
    resume_vectors,
    job_vectors
)

rows = []

for i, resume in resumes.iterrows():
    local_i = resumes.index.get_loc(i)

    # Select the most similar jobs for each resume
    ranked = similarities[local_i].argsort()[::-1][:5]

    for job_idx in ranked:
        rows.append({
            "resume_text": resume["resume_text"],
            "job_title": jobs.iloc[job_idx]["Job Title"],
            "job_description": jobs.iloc[job_idx]["job_text"],
            "similarity": float(
                similarities[local_i][job_idx]
            )
        })

pairs = pd.DataFrame(rows)

pairs.to_csv(
    "data/processed/candidate_pairs_for_review.csv",
    index=False
)

print("Candidate pairs created:", len(pairs))
print("Saved for manual review.")
