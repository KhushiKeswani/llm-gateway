from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import httpx
import json
import os
from google import genai
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
app = FastAPI()


class OllamaStreamError(Exception):
    pass


@app.get("/health")
async def root():
    return {"status": "ok"}


class PromptRequest(BaseModel):
    prompt: str
    provider: Literal["auto", "ollama", "gemini"] = "auto"

def format_event(event_type: str, text: str, provider: str):
    data = {
        "type": event_type,
        "text": text,
        "provider": provider,
    }
    return f"data: {json.dumps(data)}\n\n"


def chat_with_gemini(user_prompt):
    stream_started = False

    try:
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(model="gemini-3.5-flash")
        response = chat.send_message_stream(user_prompt)

        for chunk in response:
            if chunk.text:
                stream_started = True
                yield format_event("token", chunk.text, "gemini")

        yield format_event("done", "", "gemini")

    except Exception as exc:
        print(f"Gemini error: {exc}")
        message = (
            "Gemini stream interrupted"
            if stream_started
            else "Gemini request failed"
        )
        yield format_event("error", message, "gemini")


async def chat_with_ollama(user_prompt):
    stream_started = False

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/chat",
                json={
                    "model": "qwen2.5:1.5b",
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaStreamError(
                            "Invalid JSON received from Ollama"
                        ) from exc

                    content = chunk.get("message", {}).get("content", "")

                    if content:
                        stream_started = True
                        yield format_event("token", content, "ollama")

        yield format_event("done", "", "ollama")

    except Exception as exc:
        print(f"Ollama error: {exc}")

        if stream_started:
            yield format_event(
                "error", "Ollama stream interrupted", "ollama"
            )
        else:
            raise OllamaStreamError(
                "Ollama failed before streaming started"
            ) from exc


async def generate_response(user_prompt: str, provider: str):
    if provider == "gemini":
        for event in chat_with_gemini(user_prompt):
            yield event
        return

    try:
        async for event in chat_with_ollama(user_prompt):
            yield event

    except OllamaStreamError as exc:
        print(exc)

        if provider == "auto":
            for event in chat_with_gemini(user_prompt):
                yield event
        else:
            yield format_event(
                "error", "Ollama request failed", "ollama"
            )


@app.post("/chat")
async def generate_text(request_data: PromptRequest):
    return StreamingResponse(
        generate_response(request_data.prompt, request_data.provider),
        media_type="text/event-stream",
    )
