"""Optional LiteLLM adapter with the same normalized response contract."""

from __future__ import annotations

from typing import Any

from file_analysis_agent.agent.models import CompletionRequest, LLMResponse
from file_analysis_agent.clients.normalization import (
    normalize_chat_response,
    normalize_responses_response,
)
from file_analysis_agent.errors import ClientError, OptionalDependencyError


class LiteLLMClient:
    """Call LiteLLM synchronously and normalize its response."""

    def __init__(
        self, model: str, *, api_mode: str = "chat_completion", module: Any = None
    ) -> None:
        if api_mode not in {"chat_completion", "responses"}:
            raise ClientError(f"unsupported api mode: {api_mode}")
        self.model = model
        self.api_mode = api_mode
        self._module = module if module is not None else self._load_module()

    def complete(self, request: CompletionRequest) -> LLMResponse:
        if request.api_mode != self.api_mode:
            raise ClientError(
                f"client is configured for {self.api_mode}, request selected {request.api_mode}"
            )
        if request.model != self.model:
            raise ClientError(
                f"client model {self.model!r} does not match request model {request.model!r}"
            )
        try:
            if self.api_mode == "chat_completion":
                response = self._module.completion(**self._payload(request))
                return normalize_chat_response(response)
            response_factory = getattr(self._module, "responses", None)
            if response_factory is None or not hasattr(response_factory, "create"):
                raise ClientError("LiteLLM does not expose Responses-format support")
            response = response_factory.create(**self._responses_payload(request))
            return normalize_responses_response(response)
        except ClientError:
            raise
        except Exception as exc:
            raise ClientError(f"LiteLLM request failed: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _load_module() -> Any:
        try:
            import litellm
        except ImportError as exc:
            raise OptionalDependencyError(
                "the litellm package is required for the LiteLLM adapter; "
                "install file-analysis-agent[litellm]"
            ) from exc
        return litellm

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.message_dicts(),
            "tools": [dict(tool) for tool in request.tools],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    def _responses_payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": request.message_dicts(),
            "tools": [dict(tool) for tool in request.tools],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload
