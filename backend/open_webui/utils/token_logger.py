import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import uuid
import threading
from pathlib import Path

from open_webui.env import (
    TOKEN_USAGE_LOG_ENABLED,
    TOKEN_USAGE_LOG_FILE_PATH,
    TOKEN_USAGE_DB_ENABLED,
    log,
)
from open_webui.internal.db import engine
from open_webui.models.token_usage import TokenUsageTable, TokenUsage
from sqlalchemy import inspect
from open_webui.internal.db import SessionLocal

@dataclass
class TokenUsageEntry:
    """Data class for token usage log entries"""
    id: str
    timestamp: int
    user_id: Optional[str]
    user_email: Optional[str]
    model: str
    provider: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: Optional[Dict[str, float]] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    streaming: bool = False
    token_source: str = "api"
    metadata: Optional[Dict[str, Any]] = None

class TokenUsageLogger:
    """Dedicated logger for token usage tracking"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.enabled = TOKEN_USAGE_LOG_ENABLED
            self.db_enabled = TOKEN_USAGE_DB_ENABLED
            self.lock = threading.Lock()
            if self.enabled:
                # Ensure directory exists
                log_path = Path(TOKEN_USAGE_LOG_FILE_PATH)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                # Ensure database table exists
                if self.db_enabled:
                    self._ensure_table_exists()
                log.info(f"Token usage logger initialized: {TOKEN_USAGE_LOG_FILE_PATH}")

                # Test write
                self._write_test_entry()

            self.__class__._initialized = True

    def _ensure_table_exists(self):
        """Ensure the token_usage table exists"""
        try:
            inspector = inspect(engine)
            if 'token_usage' not in inspector.get_table_names():
                TokenUsage.__table__.create(engine)
                log.info("Created token_usage table")
            else:
                log.debug("token_usage table already exists")
        except Exception as e:
            log.error(f"Failed to ensure token_usage table exists: {e}")
            self.db_enabled = False

    def _write_test_entry(self):
        """Write a test entry to verify file writing works"""
        try:
            test_entry = {"test": "Token logger initialized", "timestamp": int(time.time())}
            with self.lock:
                with open(TOKEN_USAGE_LOG_FILE_PATH, 'a') as f:
                    f.write(json.dumps(test_entry) + '\n')
                    f.flush()
            log.info("Test entry written to token usage log")
        except Exception as e:
            log.error(f"Failed to write test entry: {e}", exc_info=True)
            self.enabled = False

    def _identify_provider(self, model: str, endpoint: str = None) -> str:
        """Identify the provider based on model name and endpoint"""
        model_lower = model.lower()
        endpoint_lower = (endpoint or "").lower()

        # Check endpoint first
        if "openai" in endpoint_lower:
            return "openai"
        elif "anthropic" in endpoint_lower:
            return "anthropic"
        elif "ollama" in endpoint_lower:
            return "ollama"
        elif "google" in endpoint_lower or "gemini" in endpoint_lower:
            return "google"

        # Then check model name patterns
        if any(x in model_lower for x in ["gpt-3", "gpt-4", "text-davinci", "text-embedding"]):
            return "openai"
        elif any(x in model_lower for x in ["claude", "anthropic"]):
            return "anthropic"
        elif any(x in model_lower for x in ["llama", "mistral", "mixtral", "qwen", "gemma", "phi"]):
            return "ollama"
        elif any(x in model_lower for x in ["gemini", "palm", "bard"]):
            return "google"
        elif any(x in model_lower for x in ["cohere", "command"]):
            return "cohere"

        return "unknown"

    def _calculate_cost(self, model: str, provider: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
        """Calculate estimated cost based on model and token usage"""
        pricing = {
            "openai": {
                "gpt-4": {"prompt": 0.03, "completion": 0.06},
                "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
                "gpt-4o": {"prompt": 0.005, "completion": 0.015},
                "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            },
            "anthropic": {
                "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
                "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
                "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
            },
        }

        provider_pricing = pricing.get(provider, {})
        model_lower = model.lower()
        model_pricing = None

        for model_key, prices in provider_pricing.items():
            if model_key in model_lower or model_lower.startswith(model_key):
                model_pricing = prices
                break

        if not model_pricing:
            return {}

        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost

        return {
            "prompt_cost": round(prompt_cost, 6),
            "completion_cost": round(completion_cost, 6),
            "total_cost": round(total_cost, 6),
            "currency": "USD",
        }

    def log_token_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        endpoint: Optional[str] = None,
        request_id: Optional[str] = None,
        streaming: bool = False,
        token_source: str = "api",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log token usage to the dedicated file"""
        if not self.enabled:
            return

        try:
            # Identify provider
            provider = self._identify_provider(model, endpoint)

            # Calculate cost
            estimated_cost = self._calculate_cost(
                model, provider, prompt_tokens, completion_tokens
            )

            # Create log entry
            entry = TokenUsageEntry(
                id=str(uuid.uuid4()),
                timestamp=int(time.time()),
                user_id=user_id,
                user_email=user_email,
                model=model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                estimated_cost=estimated_cost if estimated_cost else None,
                request_id=request_id,
                endpoint=endpoint,
                streaming=streaming,
                token_source=token_source,
                metadata=metadata
            )

            entry_dict = asdict(entry)

            self._write_to_file(entry_dict)
            if self.db_enabled:
                self._write_to_database(entry_dict)
            log.debug(f"Token usage logged for {model}: {prompt_tokens}/{completion_tokens} tokens")

        except Exception as e:
            log.error(f"Failed to log token usage: {e}", exc_info=True)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count when not provided by API"""
        return max(1, len(text) // 4)


    def _write_to_file(self, entry_dict: dict):
        """Write entry to file"""
        try:
            entry_json = json.dumps(entry_dict, default=str)

            with self.lock:
                with open(TOKEN_USAGE_LOG_FILE_PATH, 'a') as f:
                    f.write(entry_json + '\n')
                    f.flush()
        except Exception as e:
            log.error(f"Failed to write to file: {e}")

    def _write_to_database(self, entry_dict: dict):
        """Write entry to database"""
        try:
            db = SessionLocal()

            try:
                TokenUsageTable.insert(db, entry_dict)
                log.debug(f"Token usage saved to database for model: {entry_dict.get('model')}")
            finally:
                db.close()

        except Exception as e:
            log.error(f"Failed to write to database: {e}")

token_usage_logger = TokenUsageLogger()
