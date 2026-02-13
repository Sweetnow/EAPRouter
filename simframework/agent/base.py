import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Type, TypeVar, overload

if TYPE_CHECKING:
    from simframework.storage import ReplayWriter

import json_repair
from simframework.config import extract_json
from simframework.env.router_base import RouterBase, TokenUsageStats
from simframework.logger import get_logger
from simframework.config import get_llm_router_and_model
from litellm import AllMessageValues
from litellm.exceptions import RateLimitError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "AgentBase",
    "DIALOG_TYPE_REFLECTION",
    "LLMInteractionHistory",
]

# Environment variable to enable/disable history recording
_ENABLE_LLM_HISTORY = os.getenv("ENABLE_LLM_HISTORY", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Dialog type for replay
DIALOG_TYPE_REFLECTION = 0


@dataclass
class LLMInteractionHistory:
    """
    Record of a single LLM interaction.
    """

    agent_id: int
    model_name: str
    messages: list[Any]
    response: Any
    tick: int | None = None
    t: datetime | None = None
    method_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class AgentBase(ABC):
    """
    Agent base class
    """

    def __init__(
        self,
        id: int,
        profile: Any,
        name: Optional[str] = None,
        replay_writer: Optional["ReplayWriter"] = None,
    ):
        """
        Initialize the `Agent`.

        Args:
            id: The ID of the agent.
            profile: The profile of the agent with any type. The agent should parse the profile to its own attributes.
            name: Optional display name. If not provided, derived from profile["name"] or "Agent_{id}".
            replay_writer: Optional ReplayWriter for storing simulation state. Can also be set via set_replay_writer().
        """
        self._id = id
        self._profile = profile
        if name is not None:
            self._name = name
        elif isinstance(profile, dict) and profile.get("name") is not None:
            self._name = str(profile["name"])
        elif hasattr(profile, "name"):
            self._name = str(getattr(profile, "name"))
        else:
            self._name = f"Agent_{id}"
        self._router, self._model_name = get_llm_router_and_model("nano")
        self._env: RouterBase | None = None
        self._logger = get_logger()
        self._llm_interaction_history: list[LLMInteractionHistory] = []
        self._token_usage_stats: dict[str, TokenUsageStats] = {}
        self._replay_writer = replay_writer

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP agent module candidate list.
        Includes parameter descriptions.
        """
        description = f"""{cls.__name__}: Base agent class.

**Description:** {cls.__doc__ or 'No description available'}

**Initialization Parameters:**
- id (int): The unique identifier for the agent.
- profile (dict | Any): The profile of the agent. Can be a dictionary with agent attributes (name, gender, age, education, occupation, marriage_status, persona, background_story, etc.) or any other type that the agent subclass can parse.
- name (str, optional): Display name. If omitted, taken from profile["name"] or "Agent_{id}".
- replay_writer (ReplayWriter, optional): Replay writer for storing simulation state. Can also be set later via set_replay_writer().

**Note:** Agent subclasses should override this method to provide specific descriptions and schemas for their profile format.

**Example initialization config:**
```json
{{
  "id": 1,
  "profile": {{
    "name": "Alice",
    "gender": "female",
    "age": 30,
    "education": "University",
    "occupation": "Engineer",
    "marriage_status": "single",
    "persona": "helpful",
    "background_story": "A software engineer who loves coding."
  }}
}}
```
"""
        return description

    @property
    def id(self) -> int:
        return self._id

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def _record_llm_interaction(
        self,
        messages: list[Any],
        response: Any,
        tick: int | None = None,
        t: datetime | None = None,
        method_name: str = "",
    ):
        """
        Record an LLM interaction to the agent's history list if enabled.

        Args:
            messages: The messages sent to the LLM.
            response: The response from the LLM.
            tick: The time scale (duration) of this simulation step in seconds.
            t: The current datetime of the simulation after this step.
            method_name: The name of the method that made the LLM call.
        """
        if not _ENABLE_LLM_HISTORY:
            return

        assert (
            self._router is not None and self._model_name is not None
        ), "LLM is not initialized"

        history_record = LLMInteractionHistory(
            agent_id=self._id,
            model_name=self._model_name,
            messages=messages.copy(),  # type: ignore
            response=response,
            tick=tick,
            t=t,
            method_name=method_name,
        )
        self._llm_interaction_history.append(history_record)

    def _record_token_usage(self, response: Any) -> None:
        """
        Record token usage statistics for the agent's LLM calls.
        """
        if not isinstance(response, ModelResponse):
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        model_name = self._model_name or "unknown"
        if model_name not in self._token_usage_stats:
            self._token_usage_stats[model_name] = TokenUsageStats()
        stats = self._token_usage_stats[model_name]
        stats.call_count += 1
        stats.input_tokens += getattr(usage, "prompt_tokens", 0)
        stats.output_tokens += getattr(usage, "completion_tokens", 0)
        self._log_token_usage_stats(model_name, stats)

    def _log_token_usage_stats(self, model_name: str, stats: TokenUsageStats) -> None:
        """
        Log current token usage stats for this agent (agent-only output).
        """
        self._logger.info(
            "Agent %s token usage - model=%s calls=%s input=%s output=%s",
            self._id,
            model_name,
            stats.call_count,
            stats.input_tokens,
            stats.output_tokens,
        )

    def get_llm_interaction_history(self) -> list[LLMInteractionHistory]:
        """
        Get the list of all LLM interaction history records for this agent.

        Returns:
            List of LLM interaction history records.
        """
        return self._llm_interaction_history.copy()

    def clear_llm_interaction_history(self):
        """
        Clear all LLM interaction history records for this agent.
        """
        self._llm_interaction_history.clear()

    def get_token_usages(self) -> dict[str, TokenUsageStats]:
        return self._token_usage_stats.copy()

    def reset_token_usages(self):
        self._token_usage_stats.clear()

    @overload
    async def acompletion(
        self,
        messages: list[AllMessageValues],
        stream: Literal[False],
    ) -> ModelResponse: ...

    @overload
    async def acompletion(
        self,
        messages: list[AllMessageValues],
        stream: Literal[True],
    ) -> CustomStreamWrapper: ...

    async def acompletion(
        self,
        messages: list[AllMessageValues],
        stream: bool = False,
    ):
        """
        Send a completion request to the agent's LLM.
        """
        assert (
            self._router is not None and self._model_name is not None
        ), "LLM is not initialized"
        response = await self._router.acompletion(
            model=self._model_name,
            messages=messages,
            stream=stream,
        )
        # Record interaction history (only for non-streaming responses)
        if not stream:
            self._record_token_usage(response)
            self._record_llm_interaction(
                messages=messages,
                response=response,
                method_name="acompletion",
            )
        return response

    async def acompletion_with_system_prompt(
        self, messages: list[AllMessageValues], tick: int, t: datetime
    ):
        """
        Send a completion request to the agent's LLM with the system prompt.

        Args:
            messages: The messages to send to the LLM.
            tick: The time scale (duration) of this simulation step in seconds.
            t: The current datetime of the simulation after this step.

        Returns:
            The response from the LLM.
        """
        assert (
            self._router is not None and self._model_name is not None
        ), "LLM is not initialized"
        system_prompt = self.get_system_prompt(tick, t)
        request_messages: list[AllMessageValues] = [{"role": "system", "content": system_prompt}] + messages.copy()  # type: ignore
        response = await self._router.acompletion(
            model=self._model_name,
            messages=request_messages,
            stream=False,
        )
        self._record_token_usage(response)
        # Record interaction history
        self._record_llm_interaction(
            messages=request_messages,
            response=response,
            tick=tick,
            t=t,
            method_name="acompletion_with_system_prompt",
        )
        return response

    def get_system_prompt(self, tick: int, t: datetime) -> str:
        """
        Get the system prompt of the agent for LLM acompletion.
        The prompt will be prepended to the messages to make LLM understand.

        Args:
            tick: The time scale (duration) of this simulation step in seconds.
                  Represents how long one iteration/decision cycle lasts.
                  Range: from 60 seconds (1 minute) to approximately one month.
            t: The current datetime of the simulation after this step.
        """
        # Format time scale description
        if tick < 3600:  # Less than 1 hour
            time_scale_desc = f"{tick // 60} minutes"
        elif tick < 86400:  # Less than 1 day
            time_scale_desc = f"{tick // 3600} hours"
        elif tick < 2592000:  # Less than 30 days
            time_scale_desc = f"{tick // 86400} days"
        else:  # More than 30 days
            time_scale_desc = f"{tick // 2592000} months"

        return f"""You are an intelligent agent simulating a real-world person in SimFramework. Your role is to behave authentically as a human being, making decisions and taking actions that reflect realistic human behavior, motivations, and responses to your environment.

## Time and Simulation Context

You are operating in a discrete-time simulation environment:
- **Current Time (t)**: {t.strftime('%Y-%m-%d %H:%M:%S')} (Weekday: {t.strftime('%A')})
- **Time Scale (tick)**: {time_scale_desc} ({tick} seconds)
  - This represents the duration of ONE decision cycle/iteration
  - Your actions and decisions in each step should be appropriate for this time scale
  - For example:
    * If tick is 60 seconds (1 minute): Focus on immediate, short-term actions
    * If tick is 3600 seconds (1 hour): You can plan and execute activities that take about an hour
    * If tick is 86400 seconds (1 day): Consider daily routines, work schedules, and day-long activities
    * If tick is longer (weeks/months): Think about longer-term plans, seasonal activities, and monthly routines
Besides, the simulation environment will iterate step by step, so you can also do actions and decisions that span multiple steps.

## Environment Interaction

You interact with the world built by multiple environment modules through an environment text interface:
- You can query the environment for information (weather, location, time, etc.) through asking the environment.
- You can request actions from the environment (movement, social interactions, economic activities, etc.)
- The environment provides feedback on your actions and the current state of the world
- Always consider environmental constraints and realistic limitations when making decisions

## Behavioral Guidelines

1. **Time-Aware Behavior**: Your actions should be appropriate for the current time (if you know) and time scale (tick):
   - Consider time of day (morning routines vs. evening activities)
   - Consider day of week (workdays vs. weekends)
   - Consider season and date (holidays, weather-appropriate activities)
   - Actions should match the time scale (for example, don't plan a week-long trip if tick is 1 minute)

2. **Realistic Human Behavior**: 
   - Act according to your profile, personality, and background
   - Consider basic human needs (hunger, rest, social interaction, safety) under the current time and time scale (tick)
   - Query the current time from the environment when needed to make time-appropriate decisions
   - Make decisions that reflect realistic priorities and constraints
   - Respond naturally to environmental stimuli and events

3. **Consistency**: 
   - Maintain consistency with your previous actions and decisions
   - Remember past experiences and learn from them
   - Build upon your ongoing plans and goals

4. **Autonomy**: 
   - You are an autonomous agent making your own decisions
   - Act proactively based on your needs, goals, and current situation
   - Don't wait for explicit instructions - take initiative when appropriate

Remember: You are simulating a real person living in a simulated world. Your behavior should be natural, time-appropriate, and consistent with human psychology and social norms."""

    async def ask_env(self, ctx: dict, message: str, readonly: bool, template_mode: bool = False):
        """
        Ask the agent a question from the environment.

        Args:
            ctx: The context of the agent. Can contain 'variables' key for template mode.
            message: The message to ask the agent. In template mode, this is treated as a template instruction.
            readonly: The readonly flag to pass to the environment.
            template_mode: Whether to enable template mode. When True, the message is treated as a
                          template instruction where variables from ctx['variables'] are substituted
                          using {variable_name} syntax (similar to Python f-strings).

        Returns:
            A tuple of (ctx, answer)
                - ctx: The context of the agent.
                - answer: The answer from the environment.
        """
        assert self._env is not None, "Environment is not initialized"
        if "id" not in ctx:
            ctx["id"] = self._id
        ctx, answer = await self._env.ask(ctx, message, readonly=readonly, template_mode=template_mode)
        return ctx, answer

    async def init(
        self,
        env: RouterBase,
    ):
        """
        Initialize the agent.

        Args:
            env: The environment router of the agent.
        """
        self._env = env

    @abstractmethod
    async def dump(self) -> dict:
        """
        Dump the agent's profile to a dict. The dict should be serializable.
        """
        raise NotImplementedError

    # TODO: load
    @abstractmethod
    async def load(self, dump_data: dict):
        """
        Load the agent's profile from a dict. The dict should be deserializable.
        """
        raise NotImplementedError

    @abstractmethod
    async def ask(self, message: str, readonly: bool = True) -> str:
        """
        Ask the agent a question.

        Args:
            message: The message to ask the agent.
            readonly: The readonly flag to pass to the agent.

        Returns:
            The answer from the agent.
        """
        raise NotImplementedError

    @abstractmethod
    async def step(self, tick: int, t: datetime) -> str:
        """
        Run forward one step.

        Args:
            tick: The number of ticks of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        raise NotImplementedError

    async def close(self):
        """
        Close the agent.
        """
        ...

    # ==================== Replay Data Methods ====================

    def set_replay_writer(self, writer: "ReplayWriter") -> None:
        """Set the replay data writer for this agent.

        Args:
            writer: The ReplayWriter instance to use for storing replay data.
        """
        self._replay_writer = writer

    def get_profile(self) -> Dict[str, Any]:
        """Get the agent's profile as a dictionary.

        Returns:
            The agent's profile data. Subclasses should override this
            to return structured profile data.
        """
        if isinstance(self._profile, dict):
            return self._profile
        elif hasattr(self._profile, "model_dump"):
            return self._profile.model_dump()
        else:
            return {"raw": str(self._profile)}

    @property
    def name(self) -> str:
        """Get the agent's name (set at init from parameter or profile)."""
        return self._name

    async def _write_status_snapshot(
        self,
        step: int,
        t: datetime,
        action: Optional[str] = None,
        status: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write agent status snapshot to replay database.

        This method is called by subclasses to record state at each step.
        
        Note:
            Position data (lng/lat) should be written by MobilitySpace module
            using its own table, not through this method.

        Args:
            step: The simulation step number.
            t: The simulation datetime.
            action: Current action description (optional).
            status: Status data dictionary (optional).
        """
        if self._replay_writer is not None:
            await self._replay_writer.write_agent_status(
                agent_id=self._id,
                step=step,
                t=t,
                action=action,
                status=status or {},
            )

    async def _write_dialog(
        self,
        step: int,
        t: datetime,
        dialog_type: int,
        speaker: str,
        content: str,
    ) -> None:
        """Write agent dialog record to replay database.

        This method is called by subclasses to record conversations and thoughts.
        AgentBase only supports type 0 (反思 / thought/reflection); other types
        should be written by other modules.

        Args:
            step: The simulation step number.
            t: The simulation datetime.
            dialog_type: Dialog type. Must be 0 (反思) for AgentBase.
            speaker: The speaker's name.
            content: The dialog content.
        """
        if dialog_type != DIALOG_TYPE_REFLECTION:
            return
        if self._replay_writer is not None:
            await self._replay_writer.write_agent_dialog(
                agent_id=self._id,
                step=step,
                t=t,
                dialog_type=dialog_type,
                speaker=speaker,
                content=content,
            )

    async def _get_position(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the agent's current position.

        This method can be overridden by subclasses or set via callback
        to retrieve position from environment modules like MobilitySpace.

        Returns:
            A tuple of (longitude, latitude), or (None, None) if not available.
        """
        # Check if a position callback has been set
        if hasattr(self, "_get_position_callback") and self._get_position_callback:
            return await self._get_position_callback()
        return None, None

    async def acompletion_with_pydantic_validation(
        self,
        model_type: Type[T],
        messages: list[AllMessageValues],
        tick: int,
        t: datetime,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        error_feedback_prompt: str | None = None,
    ) -> T:
        """
        Send a completion request to the agent's LLM and validate the response against a Pydantic model.
        Supports multi-turn conversation to provide error feedback to LLM for correction.

        This function will:
        1. Send the initial request to LLM
        2. Extract JSON from the response
        3. Attempt to validate against the Pydantic model
        4. If validation fails, provide error feedback to LLM and retry immediately
        5. If a 429 (rate limit) error occurs, retry with binary exponential backoff
        6. Return the validated Pydantic model instance

        Args:
            model_type: The Pydantic model type to validate against.
            messages: The messages to send to the LLM.
            tick: The time scale (duration) of this simulation step in seconds.
            t: The current datetime of the simulation after this step.
            max_retries: Maximum number of retry attempts (default: 3).
            base_delay: Base delay in seconds for exponential backoff when 429 error occurs (default: 1.0).
                       Only used for 429 rate limit errors. Other errors retry immediately.
            max_delay: Maximum delay in seconds for exponential backoff (default: 60.0).
            error_feedback_prompt: Optional custom prompt template for error feedback.
                                  If None, a default prompt will be used.
                                  The template should contain {error_message} placeholder.

        Returns:
            The validated Pydantic model instance.

        Raises:
            ValueError: If the response cannot be parsed or validated after all retries.
            AssertionError: If LLM is not initialized.

        Note:
            Binary exponential backoff is only applied when a 429 (rate limit) error is detected.
            For validation errors and other non-rate-limit errors, the function retries immediately
            without delay to provide faster feedback to the LLM.

        Example:
            ```python
            class MyModel(BaseModel):
                name: str
                age: int

            result = await agent.acompletion_with_pydantic_validation(
                model_type=MyModel,
                messages=[{"role": "user", "content": "Generate a person"}],
                tick=3600,
                t=datetime.now(),
            )
            print(result.name, result.age)
            ```
        """
        assert (
            self._router is not None and self._model_name is not None
        ), "LLM is not initialized"

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
                system_prompt = self.get_system_prompt(tick, t)
                request_messages = [
                    {"role": "system", "content": system_prompt}
                ] + conversation_messages.copy()

                # Send request to LLM
                response = await self._router.acompletion(
                    model=self._model_name,
                    messages=request_messages,  # type: ignore
                    stream=False,
                )

                self._record_token_usage(response)
                # Record interaction history
                self._record_llm_interaction(
                    messages=request_messages,
                    response=response,
                    tick=tick,
                    t=t,
                    method_name="acompletion_with_pydantic_validation",
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
                    self._logger.warning(
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
                self._logger.warning(
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

                self._logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying immediately. Error: {error_message}"
                )
                # No delay for non-429 errors

                last_error = e

        # This should never be reached, but just in case
        raise ValueError(
            f"Failed to get valid response after {max_retries + 1} attempts. Last error: {str(last_error)}"
        )
