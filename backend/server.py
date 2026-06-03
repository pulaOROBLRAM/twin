from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime
from pathlib import Path

try:
    from google import genai
except ImportError:
    import google.genai as genai

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

app = FastAPI()

origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,https://master.twin-frontend.pages.dev",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if hasattr(genai, "configure"):
    genai.configure(api_key=api_key)
client = genai.Client(api_key=api_key) if hasattr(genai, "Client") else None

MEMORY_DIR = BASE_DIR.parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

PERSONALITY_FILE = BASE_DIR / "me.txt"

def load_personality():
    with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

PERSONALITY = load_personality()

def load_conversation(session_id: str) -> List[Dict]:
    file_path = MEMORY_DIR / f"{session_id}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return []

def save_conversation(session_id: str, messages: List[Dict]):
    file_path = MEMORY_DIR / f"{session_id}.json"
    with open(file_path, "w") as f:
        json.dump(messages, f, indent=2)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str


def extract_response_text(response) -> str:
    if response is None:
        return ""

    if hasattr(response, "text"):
        return response.text or ""

    # New google-genai convenience property
    if hasattr(response, "output_text"):
        return response.output_text

    # Handle dict-like responses
    if isinstance(response, dict):
        output = response.get("output") or response.get("choices")
        if output:
            first = output[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, list) and content:
                    text = content[0].get("text")
                    if text:
                        return text
                message = first.get("message")
                if isinstance(message, dict):
                    return message.get("content", "")

    # Handle object-like responses
    if hasattr(response, "output"):
        output = response.output
        if output:
            first = output[0]
            if hasattr(first, "content"):
                content = first.content
                if isinstance(content, list) and content:
                    item = content[0]
                    if hasattr(item, "text"):
                        return item.text
    if hasattr(response, "choices"):
        choices = response.choices
        if choices:
            first = choices[0]
            if hasattr(first, "message"):
                message = first.message
                return getattr(message, "content", "")

    return str(response)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if client is None:
            raise HTTPException(
                status_code=500,
                detail="Gemini client is unavailable. Install google-genai or ensure the package is available.",
            )

        session_id = request.session_id or str(uuid.uuid4())
        conversation = load_conversation(session_id)

        prompt = f"{PERSONALITY}\n\nUser: {request.message}"
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

        if hasattr(client, "chats"):
            # Convert conversation history to genai Content format
            history = []
            for msg in conversation:
                history.append({
                    "role": msg["role"],
                    "parts": [{"text": msg["content"]}]
                })
            
            chat_session = client.chats.create(model=model_name, history=history)
            response = chat_session.send_message(prompt)
        else:
            raise HTTPException(
                status_code=500,
                detail="Gemini client does not support the expected chats API interface.",
            )

        assistant_response = extract_response_text(response)

        conversation.append({"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()})
        conversation.append({"role": "model", "content": assistant_response, "timestamp": datetime.now().isoformat()})
        save_conversation(session_id, conversation)

        return ChatResponse(response=assistant_response, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)