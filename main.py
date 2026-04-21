from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import model as llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str
    model: str = llm.DEFAULT_MODEL


@app.get("/")
def root():
    return llm.get_status()


@app.post("/chat")
def chat(req: ChatRequest):
    return llm.chat(req.prompt, req.model)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(
        llm.chat_stream(req.prompt, req.model),
        media_type="text/event-stream"
    )


@app.get("/models")
def list_models():
    return llm.list_models()