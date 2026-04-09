from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOllama(model="gemma3:4b", temperature=0)

chat_history = [
    SystemMessage(content="You are a helpful assistant. give me answers in very short way.")
]

while True:
    user_input = input('you: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")

print(chat_history)
