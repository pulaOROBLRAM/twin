from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import json
import uuid
from typing import Optional, List, Dict
from supabase import create_client, Client

# Supabase setup
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # secret key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# Hugging Face setup
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL = "microsoft/DialoGPT-medium"  # good free conversational model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_personality():
    with open("me.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

PERSONALITY = load_personality()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

async def call_huggingface(prompt: str) -> str:
    """Call Hugging Face Inference API (free)"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150, "temperature": 0.7}
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            json=payload,
            headers=headers,
            timeout=30.0
        )
        if resp.status_code != 200:
            raise Exception(f"HF API error: {resp.text}")
        data = resp.json()
        # DialoGPT returns list of generated texts
        return data[0]["generated_text"].split(prompt)[-1].strip()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    # Load conversation from Supabase
    if supabase:
        result = supabase.table("conversations").select("messages").eq("session_id", session_id).execute()
        if result.data:
            messages = result.data[0]["messages"]
        else:
            messages = []
    else:
        # fallback to in‑memory for local dev
        messages = []
    
    # Build prompt with personality + history
    prompt = f"{PERSONALITY}\n\n"
    for msg in messages[-10:]:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"user: {request.message}\nassistant:"
    
    response_text = await call_huggingface(prompt)
    
    # Save new messages
    messages.append({"role": "user", "content": request.message})
    messages.append({"role": "assistant", "content": response_text})
    if supabase:
        supabase.table("conversations").upsert({
            "session_id": session_id,
            "messages": messages,
            "updated_at": "now()"
        }).execute()
    
    return ChatResponse(response=response_text, session_id=session_id)

# Vercel serverless handler (mangum)
from mangum import Mangum
handler = Mangum(app)