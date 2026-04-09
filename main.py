from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    model: str = "gemma4:e2b-it-q4_K_M"


@app.get("/")
def root():
    return {"status": "running", "model": "gemma4:e2b-it-q4_K_M"}

@app.post("/chat")
def chat(req: ChatRequest):
    response = ollama.chat(
        model=req.model,
        messages=[{"role": "user", "content": req.prompt}]
    )
    return {"response": response["message"]["content"]}

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def generate():
        for chunk in ollama.chat(
            model=req.model,
            messages=[{"role": "user", "content": req.prompt}],
            stream=True
        ):
            yield f"data: {json.dumps({'token': chunk['message']['content']})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/models")
def list_models():
    models = ollama.list()
    return {"models": [m["name"] for m in models["models"]]}

