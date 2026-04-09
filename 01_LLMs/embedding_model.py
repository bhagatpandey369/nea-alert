from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="embeddinggemma")
document = [
    "The capital of Nepal is Kathmandu.",
    "Mount Everest is the highest mountain in the world.",
    "The Amazon rainforest is the largest tropical rainforest in the world."
]

result = embedding.embed_documents(document)
print([str(r) for r in result])