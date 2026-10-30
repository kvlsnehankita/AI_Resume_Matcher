
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

resume = """
Python developer with experience in machine learning,
NLP, deep learning, pandas and SQL.
"""

job = """
We need a Python engineer with NLP, machine learning
and data processing skills.
"""

resume_embedding = model.encode(
    [resume],
    normalize_embeddings=True
)

job_embedding = model.encode(
    [job],
    normalize_embeddings=True
)

similarity = cosine_similarity(
    resume_embedding,
    job_embedding
)[0][0]

print("Resume embedding shape:", resume_embedding.shape)
print("Job embedding shape:", job_embedding.shape)
print("Semantic similarity:", round(float(similarity), 4))
print("Transformer test completed!")
