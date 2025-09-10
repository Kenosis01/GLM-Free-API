from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Union, Literal
import time
import uuid
import json

from providers.deepinfra import DeepInfra


# --- Models ---
class ContentPart(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]  # Accept either string or list


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    stream: Optional[bool] = False


# --- App Init ---
app = FastAPI()


# --- Utility ---
def flatten_content(content: Union[str, List[ContentPart]]) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join(part.text for part in content if part.type == "text")
    return ""


# --- Routes ---
@app.get("/v1/models")
def list_models():
    created_ts = int(time.time())
    data = []
    for m in DeepInfra.AVAILABLE_MODELS:
        data.append({
            "id": m,
            "object": "model",
            "created": created_ts,
            "owned_by": "local",
        })

    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    raw_body = await request.body()
    try:
        body = ChatRequest.model_validate_json(raw_body)
    except ValidationError as e:
        print("Validation error:", e)
        raise HTTPException(status_code=422, detail=e.errors())

    # Extract system prompt (optional) and assemble user prompt
    system_prompt = None
    user_parts: List[str] = []
    for m in body.messages:
        flattened = flatten_content(m.content)
        if m.role == "system":
            system_prompt = flattened
        else:
            user_parts.append(flattened)

    prompt = "\n".join(user_parts).strip()

    try:
        provider = DeepInfra(system_prompt=system_prompt) if system_prompt else DeepInfra()
    except Exception:
        provider = DeepInfra()

    out_model = body.model or provider.model

    if body.stream:
        async def sse_generator():
            created_ts = int(time.time())
            id_str = f"chatcmpl-{uuid.uuid4().hex}"
            first_chunk = True

            try:
                gen = provider.ask(prompt, stream=True, raw=False)
                for piece in gen:
                    text_piece = None
                    if isinstance(piece, dict):
                        text_piece = piece.get("text")
                    elif isinstance(piece, str):
                        text_piece = piece

                    chunk = {
                        "id": id_str,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": out_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": text_piece if text_piece else None,
                                    "role": "assistant" if first_chunk else None,
                                    "function_call": None,
                                    "tool_calls": None,
                                    "refusal": None,
                                },
                                "finish_reason": None,
                                "logprobs": None,
                            }
                        ],
                        "usage": None,
                        "service_tier": None,
                        "system_fingerprint": None,
                    }

                    first_chunk = False
                    yield f"data: {json.dumps(chunk)}\n\n"

                final = {
                    "id": id_str,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": out_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": None},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ],
                    "usage": None,
                    "service_tier": None,
                    "system_fingerprint": None,
                }
                yield f"data: {json.dumps(final)}\n\n"

            except Exception as e:
                err = {"error": str(e)}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # Non-streaming path
    try:
        resp = provider.ask(prompt, stream=False, raw=False)
        if isinstance(resp, dict):
            text = provider.get_message(resp)
        elif isinstance(resp, str):
            text = resp
        else:
            text = ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    created_ts = int(time.time())
    id_str = f"chatcmpl-{uuid.uuid4().hex}"

    prompt_tokens = len(prompt.split())
    completion_tokens = len(text.split())
    total_tokens = prompt_tokens + completion_tokens

    response_payload = {
        "id": id_str,
        "object": "chat.completion",
        "created": created_ts,
        "model": out_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "refusal": None,
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        },
        "service_tier": None,
        "system_fingerprint": None,
    }

    return JSONResponse(content=response_payload)
