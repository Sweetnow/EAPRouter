"""
Base class for Router implementations in EAPRouter V2.
This module provides the abstract interface for different routing strategies.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    Tuple,
    Dict,
    Any,
    List,
    Literal,
    overload,
    Optional,
    TYPE_CHECKING,
    Type,
    TypeVar,
)

if TYPE_CHECKING:
    from eaprouter.storage import ReplayWriter

from pydantic import BaseModel, ValidationError
from litellm import AllMessageValues
from litellm.exceptions import RateLimitError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

from eaprouter.env.base import EnvBase
from eaprouter.config import get_llm_router_and_model, extract_json
from eaprouter.logger import get_logger
from eaprouter.env.function_parser import FunctionParser, FunctionParts
from eaprouter.env.pydantic_collector import PydanticModelCollector

import black
import json_repair
import logging

T = TypeVar("T", bound=BaseModel)


class TokenUsageStats(BaseModel):
    """
    Token usage statistics for a model.

    Attributes:
        call_count: Number of API calls made.
        input_tokens: Total number of input tokens consumed.
        output_tokens: Total number of output tokens consumed.
    """

    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class FinalAnswerResponse(BaseModel):
    """
    Pydantic model for LLM final answer response.

    Attributes:
        status: Execution status (success, in_progress, fail, error)
        summary: Summary of what was accomplished or what went wrong
    """

    status: Literal["success", "in_progress", "fail", "error"]
    summary: str


class ToolInfo(BaseModel):
    """
    Pydantic model for tool information.

    Attributes:
        function_parts: Parsed function parts (signature, docstring, etc.)
        name: Tool name
        description: Tool description
        readonly: Whether the tool is readonly
        kind: Tool kind (None, "observe", or "statistics")
    """

    function_parts: FunctionParts
    name: str
    description: str
    readonly: bool
    kind: str | None = None


class ModuleToolsInfo(BaseModel):
    """
    Pydantic model for module tools information.

    Attributes:
        description: Module description
        tools: List of tool information
    """

    description: str
    tools: List[ToolInfo]


# ToolsInfoDict is just a type alias for Dict[str, ModuleToolsInfo]
ToolsInfoDict = Dict[str, ModuleToolsInfo]


class RouterBase(ABC):
    """
    Abstract base class for environment routers.

    All router implementations must inherit from this class and implement
    the abstract methods to provide different routing strategies.
    """

    def __init__(
        self,
        env_modules: list[EnvBase],
        max_steps: int = 10,
        max_llm_call_retry: int = 3,
        replay_writer: Optional["ReplayWriter"] = None,
    ):
        """
        Initialize the router.

        Args:
            env_modules: The list of environment modules to use for the router.
            max_steps: The maximum number of execution steps.
            max_llm_call_retry: The maximum number of times to retry LLM calls.
            replay_writer: Optional ReplayWriter for storing simulation state. Can also be set via set_replay_writer().
        """
        # Get router and model names from environment-based configuration
        # ReAct/codegen uses coder model
        self.coder_router, self.codegen_model_name = get_llm_router_and_model("coder")
        # Summary uses default model
        self.summary_router, self.summary_model_name = get_llm_router_and_model(
            "default"
        )

        self.env_modules = env_modules
        self.max_steps = max_steps
        self.max_llm_call_retry = max(max_llm_call_retry, 1)
        self._replay_writer = replay_writer
        if replay_writer is not None:
            for env_module in env_modules:
                env_module.set_replay_writer(replay_writer)

        # Current datetime
        self.t = datetime.now()

        # Token usage statistics: {model_name: TokenUsageStats}
        self._token_usage_stats: Dict[str, TokenUsageStats] = {}

        # Pydantic model collector for collecting BaseModel types from tool functions
        self._pydantic_collector = PydanticModelCollector()

        # World Description
        self._world_description = None
        self._generate_world_description_lock = asyncio.Lock()

    def _add_current_time_to_ctx(self, ctx: dict) -> None:
        """
        Add current time information to the context dictionary.
        This ensures all generated code and tool calls have access to the current time.

        Args:
            ctx: The context dictionary to update (modified in place).
        """
        ctx["current_time"] = {
            "datetime": self.t.isoformat(),
            "formatted": self.t.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": self.t.strftime("%A"),
            "timestamp": self.t.timestamp(),
        }

    @abstractmethod
    async def ask(
        self, ctx: dict, instruction: str, readonly: bool = False, template_mode: bool = False
    ) -> Tuple[dict, str]:
        """
        The unified interface for the interaction between the agent and the environment.

        Args:
            ctx: A dict-like object that contains the context, the environment can get some variables from it.
                 In template mode, ctx should contain a 'variables' key with variable values.
            instruction: The instruction to ask the environment. In template mode, this is treated as a
                         template instruction where variables are substituted using {variable_name} syntax.
            readonly: The readonly flag to pass to the environment modules.
            template_mode: Whether to enable template mode. When True, the instruction is treated as a
                          template instruction where variables from ctx['variables'] are substituted.

        Returns:
            A tuple of (ctx, answer)
                - ctx: A dict-like object that contains the context, the environment can update some variables in it.
                - answer: The answer from the environment.
        """
        raise NotImplementedError

    async def init(self, start_datetime: datetime):
        """
        Initialize the router with the start datetime.

        Args:
            start_datetime: The start datetime of the simulation.
        """
        self.t = start_datetime
        for env_module in self.env_modules:
            await env_module.init(start_datetime)

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step for all simulation modules.

        Args:
            tick: The number of ticks of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        self.t = t
        tasks = []
        for env_module in self.env_modules:
            tasks.append(env_module.step(tick, self.t))
        await asyncio.gather(*tasks)

    async def close(self):
        """
        Close the router.
        """
        for env_module in self.env_modules:
            await env_module.close()

    def get_tool_call_history(self) -> List[Dict[str, Any]]:
        """
        Get the tool call history from all environment modules, sorted by timestamp.

        Returns:
            A list of call history entries, sorted by timestamp (oldest first).
            Each call history entry contains:
            - module_name: str (name of the environment module)
            - function_name: str
            - args: list (serialized)
            - kwargs: dict (serialized)
            - return_value: Any (serialized, None if exception occurred)
            - exception_occurred: bool
            - exception_info: dict | None (type and message if exception occurred)
            - timestamp: str (ISO format)
        """
        all_history: List[Dict[str, Any]] = []
        for env_module in self.env_modules:
            module_name = env_module.name
            module_history = env_module.get_tool_call_history()
            # Add module_name to each call record
            for call_record in module_history:
                call_record_with_module = call_record.copy()
                call_record_with_module["module_name"] = module_name
                all_history.append(call_record_with_module)

        # Sort by timestamp (oldest first)
        all_history.sort(key=lambda x: x.get("timestamp", ""))
        return all_history

    def reset_tool_call_history(self):
        """
        Reset the tool call history for all environment modules.
        """
        for env_module in self.env_modules:
            env_module.reset_tool_call_history()

    # ==================== Replay Data Methods ====================

    def set_replay_writer(self, writer: "ReplayWriter") -> None:
        """Set the replay data writer for all environment modules.

        Args:
            writer: The ReplayWriter instance to use for storing replay data.
        """
        for env_module in self.env_modules:
            env_module.set_replay_writer(writer)

    async def get_agent_position(self, agent_id: int) -> Tuple[Optional[float], Optional[float]]:
        """Get the position of an agent from any environment module that supports it.

        Iterates through all environment modules and returns the first
        non-None position found. Only modules that implement get_agent_position
        (e.g., MobilitySpace) will be queried.

        Args:
            agent_id: The ID of the agent.

        Returns:
            A tuple of (longitude, latitude), or (None, None) if not available
            from any environment module.
        """
        for env_module in self.env_modules:
            # Only call if module implements get_agent_position
            if hasattr(env_module, 'get_agent_position'):
                lng, lat = await env_module.get_agent_position(agent_id)
                if lng is not None and lat is not None:
                    return lng, lat
        return None, None

    def get_system_prompt(self) -> str:
        """
        构建系统提示词，参考EAPRouter的风格。
        系统提示词只包含通用的指导原则，不包含具体的用户任务指令。

        Args:
            ctx: 上下文字典
            readonly: 是否只读模式

        Returns:
            系统提示词字符串
        """

        prompt = """You are an AI assistant that helps LLM agents accomplish tasks in a virtual world simulation environment.

## Guidelines

- This is a VIRTUAL WORLD simulation. ONLY USE tools and features explicitly provided by the environment modules rather than trying to access the real world.
- REJECT and report any requests that require functionality not provided by the environment modules as `fail` status.
"""

        return prompt

    @overload
    async def acompletion(
        self,
        model: Literal["coder", "summary"],
        messages: list[AllMessageValues],
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> ModelResponse: ...

    @overload
    async def acompletion(
        self,
        model: Literal["coder", "summary"],
        messages: list[AllMessageValues],
        stream: Literal[True] = True,
        **kwargs: Any,
    ) -> CustomStreamWrapper: ...

    async def acompletion(
        self,
        model: Literal["coder", "summary"],
        messages: list[AllMessageValues],
        stream: bool = False,
        max_retries: int | None = None,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        **kwargs: Any,
    ):
        """
        Wrapper for LLM router acompletion that records token usage statistics.
        System prompt is always prepended to messages.
        Includes retry logic for handling rate limit errors.

        Args:
            model: The model name to use for completion.
            messages: The messages to send to the LLM.
            stream: Whether to stream the response.
            max_retries: Maximum number of retry attempts. If None, uses self.max_llm_call_retry.
            base_delay: Base delay in seconds for exponential backoff when 429 error occurs (default: 1.0).
            max_delay: Maximum delay in seconds for exponential backoff (default: 60.0).
            **kwargs: Additional arguments to pass to the router acompletion.

        Returns:
            The response from the LLM router.
        
        Raises:
            ValueError: If the response cannot be retrieved after all retries.
        """
        logger = get_logger()
        
        if max_retries is None:
            max_retries = self.max_llm_call_retry
        else:
            max_retries = max(max_retries, 1)
        
        if model == "coder":
            router = self.coder_router
            model_name = self.codegen_model_name
        elif model == "summary":
            router = self.summary_router
            model_name = self.summary_model_name
        else:
            raise ValueError(f"Invalid model: {model}")

        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await router.acompletion(
                    model=model_name,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                )

                # Record token usage (only for non-streaming responses)
                if not stream:
                    # Type check: response should be ModelResponse when stream=False
                    if isinstance(response, ModelResponse):
                        usage = getattr(response, "usage", None)
                        if usage is not None:
                            if model not in self._token_usage_stats:
                                self._token_usage_stats[model] = TokenUsageStats()

                            stats = self._token_usage_stats[model]
                            stats.call_count += 1
                            stats.input_tokens += getattr(usage, "prompt_tokens", 0)
                            stats.output_tokens += getattr(usage, "completion_tokens", 0)

                return response
                
            except RateLimitError as e:
                # If this is the last attempt, raise the error
                if attempt >= max_retries:
                    raise ValueError(
                        f"Failed to get valid response after {max_retries + 1} attempts. Last error: {str(e)}"
                    )

                # For 429 errors, use exponential backoff
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"Rate limit error (429) detected for model '{model}' (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying after {delay:.2f} seconds with exponential backoff."
                )
                await asyncio.sleep(delay)
                last_error = e
                
            except Exception as e:
                # If this is the last attempt, raise the error
                if attempt >= max_retries:
                    raise ValueError(
                        f"Failed to get valid response after {max_retries + 1} attempts. Last error: {str(e)}"
                    )

                # For other errors, retry immediately
                logger.warning(
                    f"Request failed for model '{model}' (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying immediately. Error: {str(e)}"
                )
                last_error = e

        # This should never be reached, but just in case
        raise ValueError(
            f"Failed to get valid response after {max_retries + 1} attempts. Last error: {str(last_error)}"
        )

    async def acompletion_with_system_prompt(
        self,
        model: Literal["coder", "summary"],
        messages: list[AllMessageValues],
        **kwargs: Any,
    ):
        """
        Send a completion request to the router's LLM with the system prompt.

        Args:
            model: The model name to use for completion.
            messages: The messages to send to the LLM.
            **kwargs: Additional arguments to pass to the router acompletion.

        Returns:
            The response from the LLM router.
        """
        system_prompt = self.get_system_prompt()
        request_messages: list[AllMessageValues] = [
            {"role": "system", "content": system_prompt}
        ] + messages.copy()  # type: ignore
        response = await self.acompletion(
            model=model,
            messages=request_messages,
            stream=False,
            **kwargs,
        )
        return response

    async def acompletion_with_pydantic_validation(
        self,
        model: Literal["coder", "summary"],
        model_type: Type[T],
        messages: list[AllMessageValues],
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        error_feedback_prompt: str | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Send a completion request to the router's LLM and validate the response against a Pydantic model.
        Supports multi-turn conversation to provide error feedback to LLM for correction.

        This function will:
        1. Send the initial request to LLM with system prompt
        2. Extract JSON from the response
        3. Attempt to validate against the Pydantic model
        4. If validation fails, provide error feedback to LLM and retry immediately
        5. If a 429 (rate limit) error occurs, retry with binary exponential backoff
        6. Return the validated Pydantic model instance

        Args:
            model: The model name to use for completion ("coder" or "summary").
            model_type: The Pydantic model type to validate against.
            messages: The messages to send to the LLM.
            max_retries: Maximum number of retry attempts (default: 3).
            base_delay: Base delay in seconds for exponential backoff when 429 error occurs (default: 1.0).
                       Only used for 429 rate limit errors. Other errors retry immediately.
            max_delay: Maximum delay in seconds for exponential backoff (default: 60.0).
            error_feedback_prompt: Optional custom prompt template for error feedback.
                                  If None, a default prompt will be used.
                                  The template should contain {error_message} placeholder.
            **kwargs: Additional arguments to pass to the router acompletion.

        Returns:
            The validated Pydantic model instance.

        Raises:
            ValueError: If the response cannot be parsed or validated after all retries.
            AssertionError: If router is not initialized.

        Note:
            Binary exponential backoff is only applied when a 429 (rate limit) error is detected.
            For validation errors and other non-rate-limit errors, the function retries immediately
            without delay to provide faster feedback to the LLM.

        Example:
            ```python
            class MyModel(BaseModel):
                name: str
                age: int

            result = await router.acompletion_with_pydantic_validation(
                model="coder",
                model_type=MyModel,
                messages=[{"role": "user", "content": "Generate a person"}],
            )
            print(result.name, result.age)
            ```
        """
        logger = get_logger()

        # Get JSON schema for the model
        model_schema = model_type.model_json_schema()

        # Default error feedback prompt
        default_error_prompt = """The previous response failed validation. Please correct the following errors:

{error_message}

Please provide a corrected response in JSON format that matches the required schema:
```json
{model_schema}
```

Your corrected response:
```json
"""

        error_prompt_template = (
            error_feedback_prompt if error_feedback_prompt else default_error_prompt
        )

        conversation_messages = messages.copy()
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Add system prompt
                system_prompt = self.get_system_prompt()
                request_messages: list[AllMessageValues] = [
                    {"role": "system", "content": system_prompt}
                ] + conversation_messages.copy()  # type: ignore

                # Send request to LLM
                response = await self.acompletion(
                    model=model,
                    messages=request_messages,
                    stream=False,
                    **kwargs,
                )

                content = response.choices[0].message.content  # type: ignore
                if content is None:
                    raise ValueError("LLM returned empty content")
                conversation_messages.append({"role": "assistant", "content": content})

                # Extract JSON from response
                json_str = extract_json(content)
                if json_str is None:
                    raise ValueError("Failed to extract JSON from LLM response")

                # Repair JSON if needed
                try:
                    parsed_data = json_repair.loads(json_str)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON: {str(e)}")

                # Validate against Pydantic model
                try:
                    validated_instance = model_type.model_validate(parsed_data)
                    return validated_instance
                except ValidationError as e:
                    # Collect validation errors
                    error_messages = []
                    for error in e.errors():
                        error_path = " -> ".join(str(loc) for loc in error["loc"])
                        error_msg = error["msg"]
                        error_type = error["type"]
                        error_messages.append(
                            f"- Field '{error_path}': {error_msg} (type: {error_type})"
                        )

                    error_message = "\n".join(error_messages)
                    last_error = e

                    # If this is the last attempt, raise the error
                    if attempt >= max_retries:
                        raise ValueError(
                            f"Failed to validate response after {max_retries + 1} attempts. Last error: {error_message}"
                        )

                    # Prepare error feedback message
                    error_feedback = error_prompt_template.format(
                        error_message=error_message, model_schema=model_schema
                    )

                    # Add error feedback to conversation
                    conversation_messages.append(
                        {"role": "user", "content": error_feedback}
                    )

                    # For validation errors, retry immediately without delay
                    logger.warning(
                        f"Validation failed (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying immediately. Error: {error_message}"
                    )
                    # No delay for validation errors

            except RateLimitError as e:
                # If this is the last attempt, raise the error
                if attempt >= max_retries:
                    raise ValueError(
                        f"Failed to get valid response after {max_retries + 1} attempts. Last error: {str(e)}"
                    )

                # For 429 errors, use exponential backoff
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"Rate limit error (429) detected (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying after {delay:.2f} seconds with exponential backoff."
                )
                await asyncio.sleep(delay)
                # delete the last assistant message
                if (
                    conversation_messages
                    and conversation_messages[-1]["role"] == "assistant"
                ):
                    conversation_messages.pop()

                # record the error
                last_error = e

            except Exception as e:
                # If this is the last attempt, raise the error
                if attempt >= max_retries:
                    raise ValueError(
                        f"Failed to get valid response after {max_retries + 1} attempts. Last error: {str(e)}"
                    )

                # For other errors (ValueError, etc.), prepare error feedback and retry immediately
                error_message = str(e)
                error_feedback = error_prompt_template.format(
                    error_message=error_message, model_schema=model_schema
                )

                # Add error feedback to conversation
                conversation_messages.append(
                    {"role": "user", "content": error_feedback}
                )

                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying immediately. Error: {error_message}"
                )
                # No delay for non-429 errors

                last_error = e

        # This should never be reached, but just in case
        raise ValueError(
            f"Failed to get valid response after {max_retries + 1} attempts. Last error: {str(last_error)}"
        )

    def get_token_usages(self):
        return self._token_usage_stats.copy()

    def reset_token_usages(self):
        self._token_usage_stats.clear()

    async def dump(self) -> dict:
        """
        Dump router state to a serializable dict.
        Subclasses should override this to include router-specific state.
        """
        modules_dump: list[dict] = []
        for m in self.env_modules:
            try:
                d = await m.dump()
            except Exception:
                continue
            # Skip MCP modules
            if isinstance(d, dict) and d.get("mcp") is True:
                continue
            modules_dump.append(
                {
                    "class": m.__class__.__name__,
                    "dump": d,
                }
            )
        return {
            "t": self.t.isoformat(),
            "modules": modules_dump,
        }

    async def load(self, dump_data: dict):
        """
        Load router state from a dict produced by dump().
        Subclasses should override this to restore router-specific state.
        """
        try:
            from datetime import datetime as _dt

            t_str = dump_data.get("t")
            if isinstance(t_str, str) and len(t_str) > 0:
                self.t = _dt.fromisoformat(t_str)
        except Exception:
            pass

        items = dump_data.get("modules") or []
        if not isinstance(items, list):
            return

        # Map existing envs by class name
        name2env: dict[str, EnvBase] = {}
        for e in self.env_modules:
            name2env.setdefault(e.__class__.__name__, e)

        for item in items:
            try:
                cls_name = item.get("class")
                d = item.get("dump") or {}
                env = name2env.get(cls_name)
                if env is None:
                    continue
                # Skip MCP dumps
                if isinstance(d, dict) and d.get("mcp") is True:
                    continue
                await env.load(d)
            except Exception:
                continue

    @staticmethod
    def get_status_descriptions() -> Dict[str, str]:
        """
        Get standard status descriptions.

        The status indicates whether the user's instruction has been effectively
        completed in the environment modules, or needs to wait for some time
        before the user actively checks completion.

        Returns:
            Dictionary mapping status values to their descriptions:
            - success: The task has been completed successfully. All required operations finished without errors.
            - in_progress: The task is still being executed or more steps are needed. The agent need to check whether it is done in the next steps.
            - fail: The task could not be completed (e.g., unsupported instruction, missing data, invalid input). Include detailed reason in results.
            - error: An error occurred during code execution. Must include error details in results['error'].
            - unknown: Execution status unknown
        """
        return {
            "success": "The task has been completed successfully. All required operations finished without errors.",
            "in_progress": "The task is still being executed or more steps are needed. The agent need to check whether it is done in the next steps.",
            "fail": "The task could not be completed (e.g., unsupported instruction, missing data, invalid input). Include detailed reason in results.",
            "error": "An error occurred during code execution. Must include error details in results['error'].",
            "unknown": "Execution status unknown",
        }

    async def get_world_description(self) -> str:
        """
        Get the world description.
        """
        if self._world_description is None:
            async with self._generate_world_description_lock:
                if self._world_description is None:
                    self._world_description = await self.generate_world_description_from_tools()
        return self._world_description

    async def generate_world_description_from_tools(self) -> str:
        """
        根据环境路由器的所有工具信息（包括可写和只读工具），使用LLM生成世界描述文本。

        这个函数将工具信息转换为对PersonAgent有用的world_description，帮助agent理解：
        1. 环境中可用的模块和工具
        2. 如何通过自然语言指令与router交互
        3. 每个工具的功能和用途
        4. 如何正确使用这些工具来实现目标

        Returns:
            str: 生成的世界描述文本，可直接作为PersonAgent的world_description参数
        """
        logger = get_logger()
        logger.info("\n【生成世界描述】从环境工具信息生成world_description...")

        try:
            # 收集所有工具信息（包括可写和只读工具）
            all_tools_info = self._collect_tools_info()

            # 使用 pyi 格式构建工具信息
            tools_pyi = self._format_tools_pyi(all_tools_info, 0)

            # 构建LLM prompt
            prompt = f"""# Task: Generate World Description for PersonAgent

You are tasked with generating a comprehensive world description that will help a PersonAgent understand how to interact with the environment through a code generation router.

## Context

The PersonAgent is an autonomous agent that:
1. Makes decisions based on its needs (satiety, energy, safety, social)
2. Generates intentions and plans to fulfill its goals
3. Interacts with the environment by sending natural language instructions to a router
4. The router uses code generation to call environment module tools based on these instructions

## Available Environment Tools

The following Python code (similar to .pyi stub files) contains all available environment modules and their tools:

```python
{tools_pyi}
```

## Requirements

Generate a comprehensive world description that:

1. **Describes the Environment**: Explain what kind of world/simulation environment the agent is operating in, based on the available modules and tools.

2. **Explains Available Capabilities**: Clearly describe what actions the agent can take in this world, organized by module and tool. For each major capability, explain:
   - What the tool does
   - When and why the agent might want to use it
   - What kind of natural language instruction would trigger its use

3. **Provides Interaction Guidelines**: Explain how the agent should interact with the router:
   - The agent sends natural language instructions (e.g., "I want to move to location X", "Find nearby restaurants")
   - The router interprets these instructions and calls appropriate tools
   - The router returns results that the agent can use to make decisions
   - Instructions should be clear, specific, and actionable

4. **Guides Decision Making**: Help the agent understand:
   - How different tools relate to satisfying different needs (satiety, energy, safety, social)
   - What kinds of activities are possible in this environment
   - How to combine multiple tools to achieve complex goals
   - What constraints or limitations exist in the environment

5. **Maintains Consistency**: The description should be written from the agent's perspective, as if the agent is learning about its world. Use natural, descriptive language that helps the agent understand its capabilities and environment.

## Output Format

Generate a clear, well-structured but short world description text. The text should:
- Be comprehensive but concise
- Use clear sections and formatting
- Be written in a way that helps the agent understand its world and capabilities
- Focus on practical guidance for interacting with the environment
- Not include code or technical implementation details (focus on what the agent can do, not how the router works)

Your generated world description:"""

            # 调用LLM生成描述
            dialog: list[AllMessageValues] = [{"role": "user", "content": prompt}]

            router, model_name = get_llm_router_and_model("coder")
            last_error = None
            max_retries = self.max_llm_call_retry
            
            for attempt in range(max_retries + 1):
                try:
                    response = await router.acompletion(
                        model=model_name,  # 使用coder模型生成描述性文本
                        messages=dialog,
                        stream=False,
                    )
                    world_description = response.choices[0].message.content or ""  # type: ignore
                    break
                except RateLimitError as e:
                    if attempt >= max_retries:
                        raise e
                    delay = min(1.0 * (2**attempt), 60.0)
                    logger.warning(
                        f"Rate limit error (429) detected when generating world description (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying after {delay:.2f} seconds with exponential backoff."
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                except Exception as e:
                    if attempt >= max_retries:
                        raise e
                    logger.warning(
                        f"Request failed when generating world description (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying immediately. Error: {str(e)}"
                    )
                    last_error = e
            logger.info(f"  ✓ 生成世界描述的response: {world_description}")

            if not world_description:
                logger.warning("  ⚠ LLM返回空描述，使用默认描述")
                world_description = (
                    "You are operating in a simulated environment with various tools and capabilities. "
                    "Use natural language instructions to interact with the environment router to accomplish your goals."
                )

            logger.info(f"  ✓ 成功生成世界描述（长度: {len(world_description)} 字符）")

            return world_description.strip()

        except Exception as e:
            logger.error(f"  ❌ 生成世界描述时出错: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            # 返回一个基本的默认描述
            return (
                "You are operating in a simulated environment. "
                "Use natural language instructions to interact with the environment router to accomplish your goals."
            )

    def _collect_tools_info(self) -> ToolsInfoDict:
        """
        收集所有环境模块的工具信息，按模块组织。
        返回所有工具，不进行任何过滤。
        同时收集所有工具函数的参数类型和返回值类型中的 Pydantic BaseModel。

        Returns:
            按模块组织的工具信息字典，使用 Pydantic 模型定义
        """
        parser = FunctionParser()
        # 重置 collector，准备收集新的模型
        self._pydantic_collector.reset()
        modules_info: ToolsInfoDict = {}

        for module in self.env_modules:
            # 收集所有工具（包括可写和只读）
            all_module_tools = module._llm_tools
            registered_tools = getattr(module.__class__, "_registered_tools", {})
            tool_kinds_dict = getattr(module.__class__, "_tool_kinds", {})
            readonly_tools_dict = getattr(module.__class__, "_readonly_tools", {})

            tools_list: List[ToolInfo] = []
            for tool in all_module_tools:
                func_info = tool["function"]
                tool_name = func_info["name"]

                # 获取工具的实际readonly状态和kind
                tool_readonly = readonly_tools_dict.get(tool_name, False)
                tool_kind = tool_kinds_dict.get(tool_name)

                # 使用 FunctionParser 解析函数
                tool_obj = registered_tools.get(tool_name)
                if not (tool_obj and hasattr(tool_obj, "fn")):
                    continue

                fn = tool_obj.fn._original_func
                function_parts = parser.parse_function(fn)

                # 如果解析失败，跳过该工具
                if function_parts is None:
                    get_logger().debug(
                        f"Failed to parse function for tool {tool_name}, skipping"
                    )
                    continue

                self._pydantic_collector.collect_from_function(fn)

                tool_info = ToolInfo(
                    function_parts=function_parts,
                    name=tool_name,
                    description=func_info.get("description", ""),
                    readonly=tool_readonly,
                    kind=tool_kind,
                )
                tools_list.append(tool_info)

            if tools_list:  # 只添加有工具的模块
                modules_info[module.name] = ModuleToolsInfo(
                    description=module.description,
                    tools=tools_list,
                )

        return modules_info

    def _filter_tools_info(
        self, tools_info: ToolsInfoDict, readonly: bool | None = None, kind: str | None = None
    ) -> ToolsInfoDict:
        """
        根据 readonly 和 kind 过滤工具信息。

        Args:
            tools_info: 完整的工具信息字典
            readonly: 是否只读模式（None 表示不过滤）
            kind: 工具类型筛选（None 表示不过滤），如 "observe" 或 "statistics"

        Returns:
            过滤后的工具信息字典
        """
        filtered_info: ToolsInfoDict = {}

        for module_name, module_data in tools_info.items():
            filtered_tools: List[ToolInfo] = []

            for tool_info in module_data.tools:
                # 根据 readonly 过滤
                if readonly is not None and tool_info.readonly != readonly:
                    continue

                # 根据 kind 过滤
                if kind is not None and tool_info.kind != kind:
                    continue

                filtered_tools.append(tool_info)

            if filtered_tools:  # 只添加有工具的模块
                filtered_info[module_name] = ModuleToolsInfo(
                    description=module_data.description,
                    tools=filtered_tools,
                )

        return filtered_info

    def get_collected_pydantic_models(self) -> Dict[Type[BaseModel], str]:
        """
        获取已收集的所有 Pydantic BaseModel 类型及其源代码。

        这些模型是在调用 _collect_tools_info() 时从工具函数的参数类型和返回值类型中收集的。

        Returns:
            字典，key 是 BaseModel 类型，value 是其源代码
        """
        return self._pydantic_collector.get_collected_models()

    def _format_tools_pyi(self, modules_info: ToolsInfoDict, max_body_code_lines: int) -> str:
        """
        将工具信息格式化为类似 pyi 文件的 Python 代码格式。

        格式：
        1. 上方：所有相关的 pydantic BaseModel 的完整代码
        2. 下方：模块类的 class XXX 及其 docstring（描述变成 docstring）
        3. 在 class 中包含相关函数的签名和 docstring

        Args:
            modules_info: 按模块组织的工具信息字典（使用新的 Pydantic 模型）

        Returns:
            格式化的 Python 代码字符串（类似 pyi 文件）
        """
        # 收集所有相关的 pydantic BaseModel
        pydantic_models = self.get_collected_pydantic_models()

        # 构建输出
        lines: List[str] = []
        lines.append("# Type definitions for environment modules")
        lines.append("from pydantic import BaseModel, Field")
        lines.append("from typing import Any, Optional, Union, List, Dict, Literal, Tuple")
        lines.append("from datetime import datetime")
        lines.append("")

        # 添加所有 pydantic BaseModel 的完整代码
        if pydantic_models:
            lines.append("# Pydantic BaseModel definitions")
            for model_class, source_code in pydantic_models.items():
                lines.append("")
                lines.append(source_code)
            lines.append("")
            lines.append("")

        if len(modules_info) > 0:
            lines.append("# Environment modules")
            lines.append("# These are what you can call to interact with the environment.")
        else:
            lines.append("# No environment modules")
            lines.append("# You can't interact with the environment.")
        lines.append("")

        # 为每个模块生成类定义
        for module_name, module_data in modules_info.items():
            # 模块类定义和 docstring
            lines.append(f"class {module_name}:")
            description = module_data.description
            if description:
                lines.append(f'    """{description}"""')
            lines.append("")

            # 添加每个工具函数
            for tool_info in module_data.tools:
                # 直接使用 function_parts，已经解析好了
                function_parts = tool_info.function_parts
                sig_str = function_parts.signature

                lines.append(f"    {sig_str}")
                lines.append(f'        """{function_parts.docstring}"""')
                # 以代码注释形式呈现body_code
                for line in function_parts.body_code[:max_body_code_lines]:
                    lines.append(f"        # {line}")
                lines.append("        ...")
                lines.append("")

        pyi_code = "\n".join(lines)

        # 使用 black 格式化 pyi 代码
        try:
            # 使用 black 格式化代码
            formatted_code = black.format_str(pyi_code, mode=black.Mode())
            return formatted_code
        except Exception as e:
            # 如果格式化失败，记录警告并返回原始代码
            get_logger().warning(f"Failed to format pyi code with black: {e}")
            return pyi_code

    async def generate_final_answer(
        self,
        ctx: dict,
        instruction: str,
        results: Dict[str, Any],
        process_text: str | None = None,
        status: str = "unknown",
        error: str | None = None,
    ) -> Tuple[str, str]:
        """
        根据用户输入和Router输出生成最终答案。

        这个方法接收用户输入（ctx, instruction）和Router输出（results和过程文本），
        使用LLM总结为一个answer，并确定执行状态。
        使用JSON格式返回，通过Pydantic model验证。

        Args:
            ctx: 上下文字典，包含环境变量等信息
            instruction: 用户的原始指令
            results: Router执行的结果字典
            process_text: 过程文本（可选），描述执行过程的文本信息
            status: 初步的执行状态（可能被LLM更新）
            error: 错误信息（如果有）

        Returns:
            Tuple[str, str]: (final_answer, determined_status)
                - final_answer: LLM生成的最终答案（summary文本）
                - determined_status: 确定的执行状态（success/in_progress/fail/error）
        """
        logger = get_logger()
        # logger.debug(
        #     f"RouterBase: Generating final answer - instruction: {instruction[:100]}..., "
        #     f"preliminary status: {status}, results keys: {list(results.keys())}"
        # )

        # 构建结果字符串
        error_str = error or "None"

        # 构建过程文本部分
        process_section = ""
        if process_text:
            process_section = f"""## Execution Process
{process_text}"""

        # 构建status描述
        status_descriptions = self.get_status_descriptions()
        status_desc = status_descriptions.get(status, f"状态: {status}")

        # 使用多行f-string构建prompt
        prompt = f"""# Summarize the Final Answer

Based on the agent input (`ctx` and `instruction`), execution process and results, provide a structured JSON answer to the agent.
The answer should be scenario-based instead of technical details.
But `id` related information should be included in the summary to help the agent identify.

## Agent Input

Instruction: {instruction}

Context:
```python
{ctx}
```

{process_section}

## Results

Status: {status.upper()}

Description: {status_desc}

Results:
```python
{results}
```

{f"Error: {error_str if error_str else 'None'}" if error_str else ""}

## Format

Your answer should be a JSON object with the following structure:
```json
{{
    "status": "success|fail|in_progress|error",
    "summary": "Summarize what was accomplished (or what went wrong). Include relevant information from the execution results. Reference the semantic process information to explain the process"
}}
```

Guidelines:
1. **status**: Use one of: "success", "fail", "in_progress", "error"
    - If the returned status is "unknown", SELECT ONE OF THESE STATUSES based on the execution results
    - **success**: The task has been completed successfully. All required operations finished without errors.
    - **in_progress**: The task is still being executed or more steps are needed. The agent need to check whether it is done in the next steps.
    - **fail**: The task could not be completed (e.g., unsupported instruction, missing data, invalid input). Include detailed reason.
    - **error**: An error occurred during code execution. Must include error details.
2. **summary**: Be short, clear and helpful. Should be scenario-based instead of technical details.
    - If the execution results contain time information (e.g., from ctx['current_time']), include it in the summary when relevant.
    - Summarize what was accomplished (or what went wrong)
    - Include relevant information from the execution results
    - Reference the process information to explain the process
    - If status is 'in_progress', mentions that more steps may be needed
    - If status is 'fail' or 'error', explains why it failed

## Important Notes
- Analyze the semantic meaning of the called environment functions and their results
- Consider whether the task requires waiting for completion when choosing the status (e.g., scheduled tasks)
- The status should reflect the actual completion state in the environment modules, not just whether more tool calls are needed

Return ONLY valid JSON, nothing else.

Final Answer:"""

        # logger.debug(f"RouterBase: Built final answer prompt - length: {len(prompt)}")
        # logger.info("--------------------------------")
        # logger.info(prompt)
        # logger.info("--------------------------------")

        dialog: List[AllMessageValues] = [{"role": "user", "content": prompt}]

        response = await self.acompletion_with_pydantic_validation(
            model="summary",
            model_type=FinalAnswerResponse,
            messages=dialog,
        )

        # 增加时间标签到summary (包含星期)
        time_tag = f"[{self.t.strftime('%A')}, {self.t.strftime('%Y-%m-%d %H:%M:%S')}]"
        response.summary = f"{time_tag} {response.summary}"
        return response.summary, response.status
