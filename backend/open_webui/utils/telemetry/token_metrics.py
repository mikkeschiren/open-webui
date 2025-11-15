from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode
from typing import Optional, Dict, Any
import time

from open_webui.env import ENABLE_OTEL, log

# Get meter and tracer
meter = metrics.get_meter("open_webui.token_usage")
tracer = trace.get_tracer("open_webui.token_usage")

# Create metrics instruments
token_counter = meter.create_counter(
    name="token_usage_total",
    description="Total tokens used",
    unit="tokens"
)

token_cost_counter = meter.create_counter(
    name="token_usage_cost",
    description="Total cost of token usage",
    unit="USD"
)

request_duration_histogram = meter.create_histogram(
    name="token_usage_request_duration",
    description="Duration of AI model requests",
    unit="ms"
)

tokens_per_request_histogram = meter.create_histogram(
    name="tokens_per_request",
    description="Number of tokens per request",
    unit="tokens"
)

class TokenUsageTelemetry:
    """Helper class to export token usage metrics to OpenTelemetry"""

    @staticmethod
    def record_token_usage(
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_cost: float = 0.0,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        streaming: bool = False,
        token_source: str = "api",
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record token usage metrics to OpenTelemetry"""
        if not ENABLE_OTEL:
            return

        try:
            # Common attributes for all metrics
            attributes = {
                "model": model,
                "provider": provider,
                "endpoint": endpoint or "unknown",
                "streaming": str(streaming).lower(),
                "token_source": token_source,
            }

            if user_id:
                attributes["user_id"] = user_id

            # Record token counts
            token_counter.add(
                prompt_tokens,
                attributes={**attributes, "token_type": "prompt"}
            )

            token_counter.add(
                completion_tokens,
                attributes={**attributes, "token_type": "completion"}
            )

            # Record cost if available
            if total_cost and total_cost > 0:
                token_cost_counter.add(
                    total_cost,
                    attributes=attributes
                )

            # Record tokens per request distribution
            tokens_per_request_histogram.record(
                prompt_tokens + completion_tokens,
                attributes=attributes
            )

            # Record request duration if available
            if duration_ms:
                request_duration_histogram.record(
                    duration_ms,
                    attributes=attributes
                )

            log.debug(f"Token usage metrics recorded for {model}: {prompt_tokens}+{completion_tokens} tokens")

        except Exception as e:
            log.error(f"Failed to record token usage metrics: {e}")

    @staticmethod
    def create_span_for_token_usage(
        operation_name: str = "token_usage",
        model: str = None,
        provider: str = None
    ):
        """Create a trace span for token usage tracking"""
        if not ENABLE_OTEL:
            return None

        span = tracer.start_span(operation_name)

        if model:
            span.set_attribute("model", model)
        if provider:
            span.set_attribute("provider", provider)

        return span