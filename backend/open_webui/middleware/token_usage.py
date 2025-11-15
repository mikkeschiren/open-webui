from typing import Optional, MutableMapping, cast
import json
import time
import uuid
from asgiref.typing import (
    ASGI3Application,
    ASGIReceiveCallable,
    ASGIReceiveEvent,
    ASGISendCallable,
    ASGISendEvent,
    Scope as ASGIScope,
)
from starlette.requests import Request

from open_webui.env import (
    TOKEN_USAGE_LOG_ENABLED,
    log,
)

from open_webui.utils.auth import get_current_user, get_http_authorization_cred
from open_webui.utils.token_logger import token_usage_logger


class TokenUsageMiddleware:
    """
    Middleware specifically for tracking token usage from AI model interactions.
    Separate from audit logging for cleaner separation of concerns.
    """

    # Endpoints to monitor for token usage
    MONITORED_ENDPOINTS = [
        "/api/chat/completions",
        "/api/v1/chat/completions",
        "/ollama/api/chat",
        "/ollama/api/generate",
        "/openai/chat/completions",
        "/api/completions",
    ]

    def __init__(self, app: ASGI3Application) -> None:
        self.app = app
        self.enabled = TOKEN_USAGE_LOG_ENABLED
        if self.enabled:
            self.token_logger = token_usage_logger
            log.info(f"TokenUsageMiddleware initialized - enabled: {self.enabled}")
        else:
            log.info("TokenUsageMiddleware initialized - disabled")

    async def __call__(
        self,
        scope: ASGIScope,
        receive: ASGIReceiveCallable,
        send: ASGISendCallable,
    ) -> None:
        if not self.enabled or scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope=cast(MutableMapping, scope))
        log.debug(f"TokenUsageMiddleware processing: {request.method} {request.url.path}")

        # Check if this is an endpoint we want to monitor
        if not self._should_monitor(request):
            return await self.app(scope, receive, send)

        log.info(f"TokenUsageMiddleware monitoring: {request.url.path}")

        # Create context to capture request and response
        context = TokenUsageContext()

        async def receive_wrapper() -> ASGIReceiveEvent:
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                context.add_request_chunk(body)
            return message

        async def send_wrapper(message: ASGISendEvent) -> None:
            if message["type"] == "http.response.start":
                context.status_code = message.get("status", 200)
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                context.add_response_chunk(body)
                # If this is the last chunk, process token usage
                if not message.get("more_body", False):
                    await self._process_token_usage(request, context)
            await send(message)

        await self.app(scope, receive_wrapper, send_wrapper)

    def _should_monitor(self, request: Request) -> bool:
        """Check if this request should be monitored for token usage"""
        if request.method != "POST":
            return False
        path = request.url.path.lower()
        return any(path.startswith(endpoint.lower()) for endpoint in self.MONITORED_ENDPOINTS)

    async def _get_user_info(self, request: Request) -> dict:
        """Get user information from request"""
        try:
            # Try to get from request state first
            if hasattr(request.state, "user"):
                user = request.state.user
                return {"id": user.id, "email": user.email} if user else {}

            # Try to get from authorization header
            auth_header = request.headers.get("Authorization")
            if auth_header:
                user = get_current_user(
                    request, None, None, get_http_authorization_cred(auth_header)
                )
                if user:
                    return {"id": user.id, "email": user.email}
        except Exception as e:
            log.debug(f"Failed to get user info: {e}")
        return {}

    async def _process_token_usage(self, request: Request, context: 'TokenUsageContext'):
        """Extract and log token usage from the completed request"""
        if context.status_code != 200:
            return  # Only log successful requests

        try:
            # Get request data
            request_body = context.request_body.decode("utf-8", errors="replace")
            response_body = context.response_body.decode("utf-8", errors="replace")

            if not request_body or not response_body:
                return

            req_data = json.loads(request_body)
            model = req_data.get("model", "unknown")
            streaming = req_data.get("stream", False)

            # Get user info
            user_info = await self._get_user_info(request)

            # Extract token usage from response
            token_data = self._extract_token_usage(response_body, streaming)

            # If no token data from API, estimate it
            if not token_data:
                token_data = self._estimate_token_usage(req_data, response_body)
                token_source = "estimated"
            else:
                token_source = "api"

            # Log token usage
            self.token_logger.log_token_usage(
                model=model,
                prompt_tokens=token_data["prompt_tokens"],
                completion_tokens=token_data["completion_tokens"],
                user_id=user_info.get("id"),
                user_email=user_info.get("email"),
                endpoint=str(request.url.path),
                request_id=context.request_id,
                streaming=streaming,
                token_source=token_source,
                metadata={
                    "message_count": len(req_data.get("messages", [])),
                    "temperature": req_data.get("temperature"),
                    "max_tokens": req_data.get("max_tokens"),
                    "top_p": req_data.get("top_p"),
                    "presence_penalty": req_data.get("presence_penalty"),
                    "frequency_penalty": req_data.get("frequency_penalty"),
                }
            )
        except Exception as e:
            log.error(f"Failed to process token usage: {e}")

    def _extract_token_usage(self, response_body: str, streaming: bool) -> Optional[dict]:
        """Extract token usage from API response"""
        try:
            if not streaming:
                # Non-streaming response
                resp_data = json.loads(response_body)
                if "usage" in resp_data:
                    return {
                        "prompt_tokens": resp_data["usage"].get("prompt_tokens", 0),
                        "completion_tokens": resp_data["usage"].get("completion_tokens", 0),
                        "total_tokens": resp_data["usage"].get("total_tokens", 0)
                    }
            else:
                # Streaming response - look for usage in the last chunks
                lines = response_body.strip().split('\n')
                for line in reversed(lines):
                    line = line[6:]
                    if line and line != "[DONE]":
                        try:
                            chunk = json.loads(line)
                            if "usage" in chunk:
                                return {
                                    "prompt_tokens": chunk["usage"].get("prompt_tokens", 0),
                                    "completion_tokens": chunk["usage"].get("completion_tokens", 0),
                                    "total_tokens": chunk["usage"].get("total_tokens", 0)
                                }
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            log.debug(f"Failed to extract token usage from response: {e}")
        return None


    def _estimate_token_usage(self, request_data: dict, response_body: str) -> dict:
        """Estimate token usage for ONLY the current question and response"""
        try:
            messages = request_data.get("messages", [])

            # Get only the LAST user message (the current question)
            current_question = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        current_question = content
                    break

            # Calculate tokens for just the current question
            prompt_tokens = self.token_logger.estimate_tokens(current_question)

            # Get the response (completion)
            completion_text = ""

            # First try to parse as JSON (non-streaming response)
            try:
                resp_data = json.loads(response_body)
                if "choices" in resp_data and len(resp_data["choices"]) > 0:
                    message = resp_data["choices"][0].get("message", {})
                    completion_text = message.get("content", "")
            except json.JSONDecodeError:
                # If JSON parsing fails, try streaming format
                lines = response_body.strip().split('\n')
                content_parts = []

                for line in lines:
                    # Skip empty lines
                    if not line:
                        continue

                    # Remove "data: " prefix if present
                    if line.startswith("data: "):
                        line = line[6:]

                    # Skip [DONE] marker
                    if line == "[DONE]":
                        continue

                    # Try to parse each line as JSON
                    try:
                        chunk = json.loads(line)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                content_parts.append(delta["content"])
                    except json.JSONDecodeError:
                        # Skip lines that aren't valid JSON
                        continue

                completion_text = "".join(content_parts)

            # Calculate completion tokens
            completion_tokens = self.token_logger.estimate_tokens(completion_text)

            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }

        except Exception as e:
            log.debug(f"Failed to estimate token usage: {e}")
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

class TokenUsageContext:
    """Context for capturing request and response data"""

    def __init__(self, max_body_size: int = 10 * 1024 * 1024):  # 10MB default
        self.request_body = bytearray()
        self.response_body = bytearray()
        self.max_body_size = max_body_size
        self.status_code = None
        self.request_id = str(uuid.uuid4())

    def add_request_chunk(self, chunk: bytes):
        if len(self.request_body) < self.max_body_size:
            remaining = self.max_body_size - len(self.request_body)
            self.request_body.extend(chunk[:remaining])

    def add_response_chunk(self, chunk: bytes):
        if len(self.response_body) < self.max_body_size:
            remaining = self.max_body_size - len(self.response_body)
            self.response_body.extend(chunk[:remaining])
