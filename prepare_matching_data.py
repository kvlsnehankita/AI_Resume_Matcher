
import pandas as pd
import re
import os

RESUME_FILE = "data/raw/Resume.csv"
JOB_FILE = "data/raw/job_descriptions.csv"
OUTPUT_FILE = "data/processed/matching_pairs.csv"

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Load resume data
resumes = pd.read_csv(RESUME_FILE)
resumes = resumes.dropna(subset=["Resume_str", "Category"])

resumes["resume_text"] = resumes["Resume_str"].apply(clean_text)
resumes["category"] = resumes["Category"].astype(str).str.strip()

# Load job data
jobs = pd.read_csv(JOB_FILE)
jobs = jobs.dropna(subset=["Job Title", "Job Description"])

jobs["job_title"] = jobs["Job Title"].apply(clean_text)
jobs["job_description"] = jobs["Job Description"].apply(clean_text)
jobs["skills"] = jobs["skills"].apply(clean_text)

jobs["job_text"] = (
    jobs["job_title"] + " " +
    jobs["job_description"] + " " +
    jobs["skills"]
).str.strip()

# Remove empty records
resumes = resumes[resumes["resume_text"] != ""]
jobs = jobs[jobs["job_text"] != ""]

# Save clean datasets
os.makedirs("data/processed", exist_ok=True)

resumes[
    ["ID", "resume_text", "category"]
].to_csv(
    "data/processed/cleaned_resumes.csv",
    index=False
)

jobs[
    ["Job Title", "job_description", "skills", "job_text"]
].to_csv(
    "data/processed/cleaned_jobs.csv",
    index=False
)

print("Resume records:", len(resumes))
print("Job records:", len(jobs))
print("Resume categories:", resumes["category"].nunique())
print("Cleaned datasets saved successfully.")

