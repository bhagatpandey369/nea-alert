import json
import ollama


DEFAULT_MODEL = "gemma4:e2b-it-q4_K_M"


def get_status(model: str = DEFAULT_MODEL) -> dict:
    return {"status": "running", "model": model}


def chat(prompt: str, model: str = DEFAULT_MODEL) -> dict:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"response": response["message"]["content"]}


def chat_stream(prompt: str, model: str = DEFAULT_MODEL):
    for chunk in ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        yield f"data: {json.dumps({'token': chunk['message']['content']})}\n\n"
    yield "data: [DONE]\n\n"


def list_models() -> dict:
    models = ollama.list()
    return {"models": [m["name"] for m in models["models"]]}