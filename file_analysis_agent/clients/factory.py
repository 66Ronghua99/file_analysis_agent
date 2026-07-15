"""Create a normalized client from validated configuration."""

from file_analysis_agent.clients.fake import StaticClient
from file_analysis_agent.clients.litellm_client import LiteLLMClient
from file_analysis_agent.clients.openai_compatible import OpenAICompatibleClient
from file_analysis_agent.clients.protocol import LLMClient
from file_analysis_agent.config import AgentConfig
from file_analysis_agent.errors import ClientError


def create_client(config: AgentConfig) -> LLMClient:
    if config.provider == "fake":
        return StaticClient()
    if config.provider in {"openrouter", "vllm"}:
        return OpenAICompatibleClient(
            config.model,
            provider=config.provider,
            base_url=config.base_url,
            api_key=config.api_key,
            api_mode=config.api_mode,
        )
    if config.provider == "litellm":
        return LiteLLMClient(config.model, api_mode=config.api_mode)
    raise ClientError(f"unsupported provider: {config.provider}")
