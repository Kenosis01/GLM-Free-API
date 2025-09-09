from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Generator, Literal
import json
import time
import uuid
from datetime import datetime


# Core ChatGLM imports
from curl_cffi import CurlError
from curl_cffi.requests import Session
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

# FastAPI app
app = FastAPI(title="ChatGLM OpenAI Compatible API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Compatible Models
class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "chatglm"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]

# OpenAI Compatible Chat Models
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 600
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]

# Core ChatGLM Logic (Extracted and Adapted)
class ChatGLMCore:
    """
    Core ChatGLM logic extracted from the original class
    """
    
    url = "https://chat.z.ai"
    MODEL_MAPPING = {
        "glm-4.5V": "glm-4.5v",
        "glm-4-32B": "main_chat",
        "glm-4.5-Air": "0727-106B-API",
        "glm-4.5": "0727-360B-API",
    }
    GLM_TO_MODEL = {v: k for k, v in MODEL_MAPPING.items()}
    AVAILABLE_MODELS = list(MODEL_MAPPING.keys()) + list(GLM_TO_MODEL.keys()) + ["0727-106B-API", "0727-360B-API", "glm-4.5v", "main_chat"]

    def __init__(self, model: str = "0727-106B-API", max_tokens: int = 600, timeout: int = 30):
        self.session = Session()
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.model = self._resolve_model(model)
        
        self.headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'App-Name': 'chatglm',
            'Content-Type': 'application/json',
            'Origin': self.url,
            'User-Agent': LitAgent().random(),
            'X-App-Platform': 'pc',
            'X-App-Version': '0.0.1',
            'Accept': 'text/event-stream',
        }
        self.api_endpoint = f"{self.url}/api/chat/completions"
        self.session.headers.update(self.headers)
        
        # Initialize API key and cookie
        self.api_key = None
        self.cookie = None

    @classmethod
    def _resolve_model(cls, model: str) -> str:
        """Resolve a model nickname or API name to the API format."""
        if model in cls.GLM_TO_MODEL:
            return model
        if model in cls.MODEL_MAPPING:
            return cls.MODEL_MAPPING[model]
        if model in ["0727-106B-API", "0727-360B-API", "glm-4.5v", "main_chat"]:
            return model
        raise ValueError(f"Invalid model: {model}. Choose from: {cls.AVAILABLE_MODELS}")

    def _get_api_key(self):
        if not self.api_key:
            response = self.session.get(f"{self.url}/api/v1/auths/")
            self.api_key = response.json().get("token")
        return self.api_key

    def _get_cookie(self):
        if not self.cookie:
            response = self.session.get(f"{self.url}/")
            self.cookie = response.headers.get('Set-Cookie', '')
        return self.cookie

    def _format_messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert OpenAI format messages to a single prompt"""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        # Join with newlines and return the last user message as the main prompt
        # For simplicity, we'll use the last user message as the prompt
        user_messages = [msg.content for msg in messages if msg.role == "user"]
        return user_messages[-1] if user_messages else "Hello"

    def chat_completion(self, request: ChatCompletionRequest) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Main chat completion method"""
        prompt = self._format_messages_to_prompt(request.messages)
        
        api_key = self._get_api_key()
        payload = {
            "stream": True,
            "model": self._resolve_model(request.model),
            "messages": [{"role": "user", "content": prompt}],
            "params": {},
            "features": {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "features": [
                    {"type": "mcp", "server": "vibe-coding", "status": "hidden"},
                    {"type": "mcp", "server": "ppt-maker", "status": "hidden"},
                    {"type": "mcp", "server": "image-search", "status": "hidden"}
                ],
                "enable_thinking": True
            },
            "actions": [],
            "tags": [],
            "chat_id": "local",
            "id": str(uuid.uuid4())
        }

        if request.stream:
            return self._stream_response(payload, api_key, request)
        else:
            return self._non_stream_response(payload, api_key, request)

    def _stream_response(self, payload: dict, api_key: str, request: ChatCompletionRequest) -> Generator[str, None, None]:
        """Handle streaming response"""
        try:
            cookie = self._get_cookie()
            response = self.session.post(
                self.api_endpoint,
                json=payload,
                stream=True,
                timeout=self.timeout,
                impersonate="chrome120",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "x-fe-version": "prod-fe-1.0.70",
                }
            )
            response.raise_for_status()

            def glm_content_extractor(chunk):
                if not isinstance(chunk, dict) or chunk.get("type") != "chat:completion":
                    return None
                data = chunk.get("data", {})
                phase = data.get("phase")
                usage = data.get("usage")
                if usage:
                    return None
                delta_content = data.get("delta_content")
                if delta_content:
                    if phase == "thinking":
                        split_text = delta_content.split("</summary>\n>")[-1]
                        return {"reasoning_content": split_text}
                    elif phase == "answer":
                        return {"content": delta_content}
                    else:
                        return {"content": delta_content}
                return None

            processed_stream = sanitize_stream(
                data=response.iter_content(chunk_size=None),
                intro_value="data:",
                to_json=True,
                content_extractor=glm_content_extractor,
                yield_raw_on_error=False
            )

            chat_id = str(uuid.uuid4())
            last_content = ""
            last_reasoning = ""
            in_think = False
            
            for chunk in processed_stream:
                if not chunk:
                    continue
                    
                content = chunk.get('content') if isinstance(chunk, dict) else None
                reasoning = chunk.get('reasoning_content') if isinstance(chunk, dict) else None
                
                # Handle reasoning content
                if reasoning and reasoning != last_reasoning:
                    if not in_think:
                        stream_chunk = ChatCompletionStreamResponse(
                            id=chat_id,
                            model=request.model,
                            choices=[ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": "<think>\n\n"}
                            )]
                        )
                        yield f"data: {stream_chunk.model_dump_json()}\n\n"
                        in_think = True
                    
                    stream_chunk = ChatCompletionStreamResponse(
                        id=chat_id,
                        model=request.model,
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": reasoning}
                        )]
                    )
                    yield f"data: {stream_chunk.model_dump_json()}\n\n"
                    last_reasoning = reasoning
                
                # Close think tag if needed
                if in_think and content and content != last_content:
                    stream_chunk = ChatCompletionStreamResponse(
                        id=chat_id,
                        model=request.model,
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": "\n</think>\n\n"}
                        )]
                    )
                    yield f"data: {stream_chunk.model_dump_json()}\n\n"
                    in_think = False
                
                # Handle normal content
                if content and content != last_content:
                    stream_chunk = ChatCompletionStreamResponse(
                        id=chat_id,
                        model=request.model,
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": content}
                        )]
                    )
                    yield f"data: {stream_chunk.model_dump_json()}\n\n"
                    last_content = content

            # End stream
            final_chunk = ChatCompletionStreamResponse(
                id=chat_id,
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )]
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_chunk = ChatCompletionStreamResponse(
                id=str(uuid.uuid4()),
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": f"Error: {str(e)}"},
                    finish_reason="stop"
                )]
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

    def _non_stream_response(self, payload: dict, api_key: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle non-streaming response"""
        full_text = ""
        
        try:
            # Collect all streaming chunks
            for chunk_data in self._stream_response(payload, api_key, request):
                if chunk_data.startswith("data: ") and not chunk_data.startswith("data: [DONE]"):
                    try:
                        chunk_json = json.loads(chunk_data[6:])  # Remove "data: "
                        if chunk_json.get("choices") and len(chunk_json["choices"]) > 0:
                            delta_content = chunk_json["choices"][0].get("delta", {}).get("content", "")
                            if delta_content:
                                full_text += delta_content
                    except json.JSONDecodeError:
                        continue
            
            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_text),
                    finish_reason="stop"
                )],
                usage=ChatCompletionUsage(
                    prompt_tokens=len(request.messages[0].content.split()) if request.messages else 0,
                    completion_tokens=len(full_text.split()),
                    total_tokens=len(request.messages[0].content.split()) + len(full_text.split()) if request.messages else len(full_text.split())
                )
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

# Global ChatGLM instances cache
chatglm_instances = {}

def get_chatglm_instance(model: str, max_tokens: int = 600) -> ChatGLMCore:
    """Get or create a ChatGLM instance for the specified model"""
    key = f"{model}_{max_tokens}"
    if key not in chatglm_instances:
        chatglm_instances[key] = ChatGLMCore(model=model, max_tokens=max_tokens)
    return chatglm_instances[key]

# API Endpoints
@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    models = []
    for model_id in ChatGLMCore.AVAILABLE_MODELS:
        models.append(Model(id=model_id, owned_by="chatglm"))
    
    return ModelsResponse(data=models)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""
    try:
        # Validate model
        if request.model not in ChatGLMCore.AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found. Available models: {ChatGLMCore.AVAILABLE_MODELS}"
            )
        
        # Get ChatGLM instance
        chatglm = get_chatglm_instance(request.model, request.max_tokens or 600)
        
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                chatglm.chat_completion(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        else:
            # Return non-streaming response
            return chatglm.chat_completion(request)
            
    except ValueError as e:
        raise HTTPException(
            status_code=422, 
            detail=f"Invalid model or parameters: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ChatGLM OpenAI Compatible API Server",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)