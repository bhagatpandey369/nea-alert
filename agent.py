"""
LangGraph Chatbot using Qwen3:0.8b via Ollama
Thinking mode DISABLED — direct answers only
"""

from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── LLM  (think=False disables Qwen3's chain-of-thought) ────────────────────

llm = ChatOllama(
    model="gemma4:e2b-it-q4_K_M",
    temperature=0.7,
    # Qwen3-specific flag that suppresses the <think>…</think> block
    think=False,
)


# ── Graph node ───────────────────────────────────────────────────────────────

def chatbot_node(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ── Build graph ──────────────────────────────────────────────────────────────

builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()


# ── CLI loop ─────────────────────────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  LangGraph × Qwen3:0.8b  (thinking OFF)")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 55)

    history: list = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        result = graph.invoke({"messages": history})

        # Extract assistant reply
        assistant_msg = result["messages"][-1]
        reply = assistant_msg.content if hasattr(assistant_msg, "content") else str(assistant_msg)

        print(f"\nAssistant: {reply}")

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run()