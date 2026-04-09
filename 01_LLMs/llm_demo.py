from langchain_ollama import ChatOllama


llm = ChatOllama(model="gemma4:e2b-it-q4_K_M", temperature=0.7)
result = llm.invoke("Write a poem about the beauty of nature.")
print(result.content)