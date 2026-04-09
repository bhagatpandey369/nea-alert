from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


llm = ChatOllama(model="gemma3:4b", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant. give me answers in very short way."),
    HumanMessage(content="What is the capital of France?")
]

result = llm.invoke(messages)
messages.append(AIMessage(content=result.content))


print(messages)
