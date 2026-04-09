from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = OllamaEmbeddings(model="embeddinggemma")
document = [
    "The capital of Nepal is Kathmandu.",
    "Mount Everest is the highest mountain in the world.",
    "The Amazon rainforest is the largest tropical rainforest in the world.",
    "Paris is the capital of France.",
    "Python is a popular programming language."
]

query = "Where is the Amrica located?"
doc_embedding= embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)
similarity_scores = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)[0]
print(f"Most similar document: '{document[index]}' with similarity score: {score:.4f}")
