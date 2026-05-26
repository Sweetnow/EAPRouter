"""LLM configuration from environment variables for eaprouter"""

from __future__ import annotations

import os
import re
import uuid
from typing import Literal, Optional
from litellm.router import Router

from eaprouter.logger import get_logger, setup_litellm_logging
from mem0.memory.main import MemoryConfig

__all__ = [
    "Config",
    "get_llm_router",
    "get_llm_router_and_model",
    "get_model_name",
    "extract_json",
]

logger = get_logger()

# Initialize LiteLLM logging once
_litellm_logging_initialized = False


class Config:
    """
    Configuration from environment variables for eaprouter.

    This class manages all configuration settings for the eaprouter framework,
    including LLM API credentials, model selections, and system paths. All settings
    can be configured via environment variables with sensible defaults.

    Environment Variables:
        EAPROUTER_HOME_DIR: Base directory for storing agent data and memories
        EAPROUTER_LLM_API_KEY: API key for the default LLM service
        EAPROUTER_LLM_API_BASE: Base URL for the default LLM API endpoint
        EAPROUTER_LLM_MODEL: Model name for the default LLM
        EAPROUTER_CODER_LLM_API_KEY: API key for code generation LLM (falls back to LLM_API_KEY)
        EAPROUTER_CODER_LLM_API_BASE: Base URL for code generation LLM API
        EAPROUTER_CODER_LLM_MODEL: Model name for code generation tasks
        EAPROUTER_NANO_LLM_API_KEY: API key for high-frequency operations LLM (falls back to LLM_API_KEY)
        EAPROUTER_NANO_LLM_API_BASE: Base URL for nano LLM API
        EAPROUTER_NANO_LLM_MODEL: Model name for high-frequency/low-latency operations
        EAPROUTER_EMBEDDING_API_KEY: API key for embedding model (falls back to LLM_API_KEY)
        EAPROUTER_EMBEDDING_API_BASE: Base URL for embedding API
        EAPROUTER_EMBEDDING_MODEL: Model name for text embeddings
        EAPROUTER_EMBEDDING_DIMS: Dimensionality of embedding vectors
    """

    HOME_DIR: str = os.getenv("EAPROUTER_HOME_DIR", "./eaprouter_data")
    """
    Base directory path for storing agent data, memories, and persistent files.

    Environment variable: EAPROUTER_HOME_DIR
    Default: "./eaprouter_data"

    This directory will contain subdirectories for memories, agent states, and other
    persistent data. The path can be absolute or relative to the current working directory.
    """

    # Default LLM settings
    # These are used for general-purpose language model operations throughout the system.

    LLM_API_KEY: Optional[str] = os.getenv("EAPROUTER_LLM_API_KEY")
    """
    API key for authenticating with the default LLM service.

    Environment variable: EAPROUTER_LLM_API_KEY
    Default: None (must be set for the system to function)

    This is the primary API key used for most LLM operations. If not set, the system
    will raise an error when attempting to create LLM routers. Other specialized LLM
    configurations (coder, nano, embedding) will fall back to this key if their specific
    keys are not provided.
    """

    LLM_API_BASE: str = os.getenv(
        "EAPROUTER_LLM_API_BASE", "https://cloud.infini-ai.com/maas/v1"
    )
    """
    Base URL endpoint for the default LLM API service.

    Environment variable: EAPROUTER_LLM_API_BASE
    Default: "https://cloud.infini-ai.com/maas/v1"

    This should point to the API endpoint that supports OpenAI-compatible API calls.
    The URL should include the protocol (https://) and the base path, but not the
    specific model endpoint (e.g., /chat/completions).
    """

    LLM_MODEL: str = os.getenv("EAPROUTER_LLM_MODEL", "qwen3-next-80b-a3b-instruct")
    """
    Model identifier for the default LLM used in general operations.

    Environment variable: EAPROUTER_LLM_MODEL
    Default: "qwen3-next-80b-a3b-instruct"

    This model is used for most language understanding and generation tasks that don't
    require specialized models. The model name should match what your API provider expects.
    """

    # Coder LLM settings
    # These are specifically optimized for code generation, analysis, and programming tasks.

    CODER_LLM_API_KEY: Optional[str] = (
        os.getenv("EAPROUTER_CODER_LLM_API_KEY") or LLM_API_KEY
    )
    """
    API key for the code generation LLM service.

    Environment variable: EAPROUTER_CODER_LLM_API_KEY
    Default: Falls back to LLM_API_KEY if not set

    This key is used specifically for code-related operations. If not provided, the system
    will use the default LLM_API_KEY. Setting a separate key allows you to use a different
    API provider or account for code generation tasks, which may have different rate limits
    or pricing structures.
    """

    CODER_LLM_API_BASE: str = os.getenv("EAPROUTER_CODER_LLM_API_BASE", LLM_API_BASE)
    """
    Base URL endpoint for the code generation LLM API.

    Environment variable: EAPROUTER_CODER_LLM_API_BASE
    Default: Falls back to LLM_API_BASE if not set

    Allows you to use a different API endpoint specifically for code generation tasks.
    This is useful if you want to route code generation requests to a different service
    or region for better performance or cost optimization.
    """

    CODER_LLM_MODEL: str = os.getenv("EAPROUTER_CODER_LLM_MODEL", "glm-4.7")
    """
    Model identifier for code generation and programming tasks.

    Environment variable: EAPROUTER_CODER_LLM_MODEL
    Default: "glm-4.7"

    This model is specifically used for code generation, code analysis, and other
    programming-related operations. Choose a model that is optimized for code understanding
    and generation, such as models trained on codebases.
    """

    # Nano LLM settings
    # These are optimized for high-frequency, low-latency operations that require fast responses.

    NANO_LLM_API_KEY: Optional[str] = (
        os.getenv("EAPROUTER_NANO_LLM_API_KEY") or LLM_API_KEY
    )
    """
    API key for the nano LLM service used in high-frequency operations.

    Environment variable: EAPROUTER_NANO_LLM_API_KEY
    Default: Falls back to LLM_API_KEY if not set

    The nano LLM is used for operations that require frequent, fast responses such as
    memory operations, quick decision-making, and low-latency tasks. Setting a separate
    key allows you to use a faster or cheaper model for these high-frequency operations.
    """

    NANO_LLM_API_BASE: str = os.getenv("EAPROUTER_NANO_LLM_API_BASE", LLM_API_BASE)
    """
    Base URL endpoint for the nano LLM API.

    Environment variable: EAPROUTER_NANO_LLM_API_BASE
    Default: Falls back to LLM_API_BASE if not set

    Allows routing high-frequency operations to a different endpoint, which may be
    optimized for low latency or located in a different geographic region for better
    response times.
    """

    NANO_LLM_MODEL: str = os.getenv(
        "EAPROUTER_NANO_LLM_MODEL", "qwen3-next-80b-a3b-instruct"
    )
    """
    Model identifier for high-frequency, low-latency operations.

    Environment variable: EAPROUTER_NANO_LLM_MODEL
    Default: "qwen3-next-80b-a3b-instruct"

    This model is used for operations that require fast responses, such as memory
    retrieval, quick reasoning, and other tasks where latency is critical. Typically,
    you might choose a smaller or faster model for these operations to reduce response time.
    """

    # Embedding model settings
    # These are used for converting text into vector embeddings for semantic search and similarity.

    EMBEDDING_API_KEY: Optional[str] = (
        os.getenv("EAPROUTER_EMBEDDING_API_KEY") or LLM_API_KEY
    )
    """
    API key for the embedding model service.

    Environment variable: EAPROUTER_EMBEDDING_API_KEY
    Default: Falls back to LLM_API_KEY if not set

    This key is used specifically for embedding operations, which convert text into
    high-dimensional vectors for semantic search, similarity matching, and memory
    operations. Some providers offer separate embedding services with different pricing.
    """

    EMBEDDING_API_BASE: str = os.getenv("EAPROUTER_EMBEDDING_API_BASE", LLM_API_BASE)
    """
    Base URL endpoint for the embedding API service.

    Environment variable: EAPROUTER_EMBEDDING_API_BASE
    Default: Falls back to LLM_API_BASE if not set

    Allows you to use a different API endpoint specifically for embedding operations.
    Some providers have dedicated embedding endpoints that may offer better performance
    or different pricing models for embedding tasks.
    """

    EMBEDDING_MODEL: str = os.getenv("EAPROUTER_EMBEDDING_MODEL", "bge-m3")
    """
    Model identifier for text embedding generation.

    Environment variable: EAPROUTER_EMBEDDING_MODEL
    Default: "bge-m3"

    This model is used to convert text into dense vector representations (embeddings).
    The embeddings are used for semantic search, similarity matching, and storing
    memories in vector databases. Choose a model that produces high-quality embeddings
    for your use case and language.
    """

    EMBEDDING_DIMS: int = int(os.getenv("EAPROUTER_EMBEDDING_DIMS", "1024"))
    """
    Dimensionality of the embedding vectors produced by the embedding model.

    Environment variable: EAPROUTER_EMBEDDING_DIMS
    Default: 1024

    This specifies the size of the vector space for embeddings. Higher dimensions
    can capture more nuanced semantic information but require more storage and computation.
    The value must match the actual output dimensionality of the selected embedding model.
    Common values are 384, 512, 768, 1024, or 1536 depending on the model.
    """

    # MiroFlow MCP settings

    MIROFLOW_MCP_URL: str = os.getenv("MIROFLOW_MCP_URL", "").strip()
    """
    Base URL for the MiroFlow MCP HTTP endpoint.

    Environment variable: MIROFLOW_MCP_URL
    Example: "http://127.0.0.1:18001/mcp/"
    """

    MIROFLOW_MCP_TOKEN: str = os.getenv("MIROFLOW_MCP_TOKEN", "").strip()
    """
    Authentication token for the MiroFlow MCP server.

    Environment variable: MIROFLOW_MCP_TOKEN
    The token is sent as a Bearer token in the Authorization header.
    """

    MIROFLOW_DEFAULT_LLM: str = os.getenv("MIROFLOW_DEFAULT_LLM", "qwen-3").strip()
    """
    Default LLM model name used by MiroFlow MCP tasks.

    Environment variable: MIROFLOW_DEFAULT_LLM
    Default: "qwen-3"
    """

    MIROFLOW_DEFAULT_AGENT: str = os.getenv(
        "MIROFLOW_DEFAULT_AGENT", "mirothinker_v1.5_keep5_max200"
    ).strip()
    """
    Default agent configuration name used by MiroFlow MCP tasks.

    Environment variable: MIROFLOW_DEFAULT_AGENT
    Default: "mirothinker_v1.5_keep5_max200"
    """

    @classmethod
    def get_router(
        cls, model_type: Literal["default", "coder", "nano"] = "default"
    ) -> Router:
        """
        Get LLM router for specified model type

        Args:
            model_type: One of "default", "coder", "nano"

        Returns:
            Router instance
        """
        global _litellm_logging_initialized

        # Initialize LiteLLM logging on first router creation
        if not _litellm_logging_initialized:
            setup_litellm_logging()
            _litellm_logging_initialized = True

        if model_type == "coder":
            # Coder model with fallback to default, then nano
            coder_api_key = cls.CODER_LLM_API_KEY
            coder_api_base = cls.CODER_LLM_API_BASE
            coder_model = cls.CODER_LLM_MODEL
            
            default_api_key = cls.LLM_API_KEY
            default_api_base = cls.LLM_API_BASE
            default_model = cls.LLM_MODEL
            
            nano_api_key = cls.NANO_LLM_API_KEY
            nano_api_base = cls.NANO_LLM_API_BASE
            nano_model = cls.NANO_LLM_MODEL

            if not coder_api_key:
                raise ValueError(
                    "API key not configured for coder model. "
                    "Set EAPROUTER_CODER_LLM_API_KEY or EAPROUTER_LLM_API_KEY"
                )
            if not default_api_key:
                raise ValueError(
                    "API key not configured for default model (fallback). "
                    "Set EAPROUTER_LLM_API_KEY"
                )
            if not nano_api_key:
                raise ValueError(
                    "API key not configured for nano model (fallback). "
                    "Set EAPROUTER_NANO_LLM_API_KEY or EAPROUTER_LLM_API_KEY"
                )

            # Build model_list with all three models
            model_list = [
                {
                    "model_name": coder_model,
                    "litellm_params": {
                        "model": f"openai/{coder_model}",
                        "api_key": coder_api_key,
                        "api_base": coder_api_base,
                    },
                },
                {
                    "model_name": default_model,
                    "litellm_params": {
                        "model": f"openai/{default_model}",
                        "api_key": default_api_key,
                        "api_base": default_api_base,
                    },
                },
                {
                    "model_name": nano_model,
                    "litellm_params": {
                        "model": f"openai/{nano_model}",
                        "api_key": nano_api_key,
                        "api_base": nano_api_base,
                    },
                },
            ]
            
            # Configure fallback chain: coder -> default -> nano
            # fallbacks should be a list of dicts, where each dict maps primary model to fallback models
            fallbacks = [
                {coder_model: [default_model, nano_model]}
            ]
            
            print(f"Model list for coder (with fallbacks): {model_list}")
            print(f"Fallbacks: {fallbacks}")
            
            return Router(
                model_list=model_list,
                fallbacks=fallbacks,
                cache_responses=True,
                num_retries=5,  # 设置429错误的重试次数为5次
            )
        elif model_type == "default":
            # Default model with fallback to nano
            default_api_key = cls.LLM_API_KEY
            default_api_base = cls.LLM_API_BASE
            default_model = cls.LLM_MODEL
            
            nano_api_key = cls.NANO_LLM_API_KEY
            nano_api_base = cls.NANO_LLM_API_BASE
            nano_model = cls.NANO_LLM_MODEL

            if not default_api_key:
                raise ValueError(
                    "API key not configured for default model. "
                    "Set EAPROUTER_LLM_API_KEY"
                )
            if not nano_api_key:
                raise ValueError(
                    "API key not configured for nano model (fallback). "
                    "Set EAPROUTER_NANO_LLM_API_KEY or EAPROUTER_LLM_API_KEY"
                )

            # Build model_list with default and nano models
            model_list = [
                {
                    "model_name": default_model,
                    "litellm_params": {
                        "model": f"openai/{default_model}",
                        "api_key": default_api_key,
                        "api_base": default_api_base,
                    },
                },
                {
                    "model_name": nano_model,
                    "litellm_params": {
                        "model": f"openai/{nano_model}",
                        "api_key": nano_api_key,
                        "api_base": nano_api_base,
                    },
                },
            ]
            
            # Configure fallback chain: default -> nano
            fallbacks = [
                {default_model: [nano_model]}
            ]
            
            print(f"Model list for default (with fallbacks): {model_list}")
            print(f"Fallbacks: {fallbacks}")
            
            return Router(
                model_list=model_list,
                fallbacks=fallbacks,
                cache_responses=True,
                num_retries=5,  # 设置429错误的重试次数为5次
            )
        else:  # nano
            api_key = cls.NANO_LLM_API_KEY
            api_base = cls.NANO_LLM_API_BASE
            model = cls.NANO_LLM_MODEL

            if not api_key:
                raise ValueError(
                    f"API key not configured for {model_type} model. "
                    f"Set EAPROUTER_{model_type.upper()}_LLM_API_KEY or EAPROUTER_LLM_API_KEY"
                )

            model_list = [
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": f"openai/{model}",
                        "api_key": api_key,
                        "api_base": api_base,
                    },
                },
            ]
            print(model_list)
            return Router(
                model_list=model_list,
                cache_responses=True,
                num_retries=5,  # 设置429错误的重试次数为5次
            )

    @classmethod
    def get_default_router(cls) -> Router:
        """Get default LLM router"""
        return cls.get_router("default")

    @classmethod
    def get_mem0_config(cls, id: str) -> MemoryConfig:
        # Generate a random string to avoid path conflicts
        random_suffix = uuid.uuid4().hex[:8]
        memory_config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": f"memories_{id}_{random_suffix}",
                    "path": os.path.join(cls.HOME_DIR, f"memories_{id}_{random_suffix}"),
                },
            },
            "storage_config": {
                "provider": "sqlite",
                "path": os.path.join(cls.HOME_DIR, f"memories_{id}_{random_suffix}.db"),
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": cls.NANO_LLM_MODEL,
                    "api_key": cls.NANO_LLM_API_KEY,
                    "openai_base_url": cls.NANO_LLM_API_BASE,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": cls.EMBEDDING_MODEL,
                    "api_key": cls.EMBEDDING_API_KEY,
                    "openai_base_url": cls.EMBEDDING_API_BASE,
                    "embedding_dims": cls.EMBEDDING_DIMS,
                },
            },
        }
        return MemoryConfig.model_validate(memory_config)


# Global router instances (lazy initialization)
_default_router: Optional[Router] = None
_coder_router: Optional[Router] = None
_nano_router: Optional[Router] = None


def get_llm_router(model_type: str = "default") -> Router:
    """
    Get LLM router (cached)

    Args:
        model_type: One of "default", "coder", "nano"

    Returns:
        Router instance
    """
    global _default_router, _coder_router, _nano_router

    if model_type == "coder":
        if _coder_router is None:
            _coder_router = Config.get_router("coder")
        return _coder_router
    elif model_type == "nano":
        if _nano_router is None:
            _nano_router = Config.get_router("nano")
        return _nano_router
    else:  # default
        if _default_router is None:
            _default_router = Config.get_router("default")
        return _default_router


def get_model_name(model_type: str = "default") -> str:
    """
    Get model name for specified model type

    Args:
        model_type: One of "default", "coder", "nano"

    Returns:
        Model name string
    """
    if model_type == "coder":
        return Config.CODER_LLM_MODEL
    elif model_type == "nano":
        return Config.NANO_LLM_MODEL
    else:  # default
        return Config.LLM_MODEL


def get_llm_router_and_model(model_type: str = "default") -> tuple[Router, str]:
    """
    Get LLM router and model name (cached)

    Args:
        model_type: One of "default", "coder", "nano"

    Returns:
        Tuple of (Router instance, model_name)
    """
    router = get_llm_router(model_type)
    model_name = get_model_name(model_type)
    return router, model_name


def extract_json(text: str) -> str | None:
    """
    Robustly extract JSON text from content that may contain other text.

    This function finds and extracts JSON objects or arrays from text that may
    have surrounding non-JSON content. It handles nested structures and correctly
    tracks string boundaries to find the matching closing bracket/brace.

    Note: This function only extracts the JSON text content. For repairing
    malformed JSON, use json_repair or similar libraries.

    Args:
        text: The input text that may contain JSON.

    Returns:
        The extracted JSON string if found, None otherwise.

    Examples:
        >>> extract_json("Some text {\"key\": \"value\"} more text")
        '{\"key\": \"value\"}'

        >>> extract_json("```json\n{\"a\": 1}\n```")
        '{\"a\": 1}'

        >>> extract_json("Response: {'key': 'value'}")
        "{'key': 'value'}"
    """
    if not text:
        return None

    # Try to find JSON code blocks first (common in LLM responses)
    json_block_pattern = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"
    json_block_match = re.search(json_block_pattern, text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1)

    # Find the first occurrence of { or [
    start_idx = -1
    start_char = None
    end_char = None

    for i, char in enumerate(text):
        if char == "{":
            start_idx = i
            start_char = "{"
            end_char = "}"
            break
        elif char == "[":
            start_idx = i
            start_char = "["
            end_char = "]"
            break

    if start_idx == -1:
        return None

    # Find matching closing bracket/brace
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                # Found matching closing bracket
                return text[start_idx : i + 1]

    # If we didn't find a closing bracket, return what we have
    # (might be incomplete JSON, but let json_repair handle it)
    return text[start_idx:]
