
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

RESUME_FILE = "data/processed/cleaned_resumes.csv"
JOB_FILE = "data/processed/cleaned_jobs.csv"

SKILLS = [
    "python", "java", "c++", "c#", "sql",
    "html", "css", "javascript", "react",
    "machine learning", "deep learning", "nlp",
    "natural language processing", "bert",
    "transformers", "pytorch", "tensorflow",
    "pandas", "numpy", "data analysis",
    "power bi", "aws", "docker", "git",
    "mongodb", "mysql", "recruitment",
    "employee relations", "human resources",
    "hr", "marketing", "customer service",
    "project management", "payroll",
    "benefits administration", "training",
    "compliance", "sales", "communication",
    "leadership", "reporting"
]

def extract_skills(text):
    text = str(text).lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))

resumes = pd.read_csv(RESUME_FILE)
jobs = pd.read_csv(JOB_FILE)

resume_text = str(resumes.iloc[0]["resume_text"])

jobs = jobs.dropna(subset=["job_text"]).copy()
jobs["job_text"] = jobs["job_text"].astype(str)

jobs = jobs.drop_duplicates(
    subset=["Job Title", "job_text"]
).reset_index(drop=True)

resume_skills = extract_skills(resume_text)

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

jobs["similarity"] = scores

top_jobs = jobs.sort_values(
    "similarity",
    ascending=False
).head(10)

print("\nResume skills:")
print(", ".join(resume_skills) or "No listed skills detected")

print("\nTOP JOB MATCHES WITH SKILL ANALYSIS\n")

for _, row in top_jobs.iterrows():
    job_skills = extract_skills(row["job_text"])

    matched = sorted(set(resume_skills) & set(job_skills))
    missing = sorted(set(job_skills) - set(resume_skills))

    print("Job:", row["Job Title"])
    print("Similarity:", round(float(row["similarity"]), 4))
    print("Matched skills:", ", ".join(matched) or "None detected")
    print("Potentially missing skills:", ", ".join(missing) or "None detected")
    print("-" * 60)
