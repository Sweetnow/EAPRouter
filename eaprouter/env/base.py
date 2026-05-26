import functools
import inspect
import json
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple, TypeVar, overload

if TYPE_CHECKING:
    from eaprouter.storage import ReplayWriter

from mcp.server.fastmcp.tools.base import Tool
from mcp.server.fastmcp.tools.tool_manager import ToolManager
from openai.types.chat import ChatCompletionToolParam

__all__ = ["tool", "EnvBase"]


F = TypeVar("F", bound=Callable)


@overload
def tool(
    readonly: Literal[True],
    name: str | None = None,
    description: str | None = None,
    kind: Literal["observe", "statistic", "statistics"] = ...,
) -> Callable[[F], F]:
    """Overload for observe/statistic tools that must be readonly."""
    ...


@overload
def tool(
    readonly: bool,
    name: str | None = None,
    description: str | None = None,
    kind: None = None,
) -> Callable[[F], F]:
    """Overload for regular tools."""
    ...


def tool(
    readonly: bool,
    name: str | None = None,
    description: str | None = None,
    kind: Literal["observe", "statistic", "statistics"] | None = None,
) -> Callable[[F], F]:
    """
    Register a tool method to the environment module.

    Args:
        readonly: If the tool is readonly. **Must be True when kind is 'observe' or 'statistic'.**
        name: The name of the tool.
        description: The description of the tool.
        kind: The kind of tool. Can be "observe" (for observation functions that can accept
            at most one parameter besides self, typically agent_id) or "statistic" (for
            aggregate statistic functions that can only have self parameter). Default is None for
            regular tools.

            **Important**: When kind is 'observe' or 'statistic', readonly must be True.
            This is enforced at runtime and will raise a ValueError if violated.

    Returns:
        The decorator function.

    Raises:
        ValueError: If kind is 'observe' or 'statistic' but readonly is False.
    """

    def tool_decorator(func: F) -> F:
        normalized_kind = "statistic" if kind == "statistics" else kind

        # Validate readonly constraint for observe/statistic tools
        if normalized_kind in ("observe", "statistic") and not readonly:
            raise ValueError(
                f"Tool '{func.__name__}' with kind='{kind}' must have readonly=True. "
                f"Tools of kind 'observe' or 'statistic' are read-only by design."
            )

        # Validate kind parameter if provided
        if kind is not None:
            if kind not in ("observe", "statistic", "statistics"):
                raise ValueError(
                    f"Invalid kind '{kind}'. Must be 'observe', 'statistic', 'statistics', or None."
                )

            # Get function signature to validate parameters
            sig = inspect.signature(func)
            params = list(sig.parameters.values())

            # Exclude the first parameter (self) for class methods
            # For instance methods, the first parameter is typically 'self'
            non_self_params = params[1:] if len(params) > 0 else []

            if normalized_kind == "observe":
                # Observe functions can have at most one parameter besides self
                if len(non_self_params) > 1:
                    param_names = [p.name for p in non_self_params]
                    raise ValueError(
                        f"Tool '{func.__name__}' with kind='observe' can have at most one "
                        f"parameter besides self, but found {len(non_self_params)}: {param_names}. The only one allowed parameter is agent_id, id, or person_id"
                    )
            elif normalized_kind == "statistic":
                # Statistic functions can only have self parameter
                if len(non_self_params) > 0:
                    param_names = [p.name for p in non_self_params]
                    raise ValueError(
                        f"Tool '{func.__name__}' with kind='statistic' can only have self "
                        f"parameter, but found additional parameters: {param_names}"
                    )

        # Store the tool information in the function's attribute for Metaclass usage
        tool = Tool.from_function(
            func,
            name=name,
            description=description,
        )
        func._tool_info = {  # type: ignore
            "mcp.Tool": tool,
            "readonly": readonly,
            "kind": normalized_kind,
        }

        # Wrap the function to record call history
        original_func = func
        tool_name = name if name else func.__name__
        
        # Get function signature to convert args to kwargs
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        # Skip 'self' parameter for instance methods
        if param_names and param_names[0] == 'self':
            param_names = param_names[1:]

        def _normalize_to_kwargs(args, kwargs):
            """
            Convert args and kwargs to unified kwargs dict based on function signature.
            
            Args:
                args: Positional arguments (excluding self)
                kwargs: Keyword arguments
                
            Returns:
                Dict of arguments with parameter names as keys
            """
            normalized_kwargs = {}
            
            # Process positional args first (map to parameter names by position)
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                    # Only add if not already in kwargs (kwargs take precedence)
                    if param_name not in kwargs:
                        normalized_kwargs[param_name] = arg
            
            # Add all kwargs
            normalized_kwargs.update(kwargs)
            
            return normalized_kwargs

        def _create_call_record(
            args, kwargs, return_value, exception_occurred, exception_info
        ):
            """Helper function to create a call record."""
            # Convert args and kwargs to unified kwargs dict
            normalized_kwargs = _normalize_to_kwargs(args, kwargs)
            kwargs_repr = _serialize_to_literal(normalized_kwargs)
            return_value_repr = (
                _serialize_to_literal(return_value)
                if return_value is not None
                else None
            )

            return {
                "function_name": tool_name,
                "kwargs": kwargs_repr,
                "return_value": return_value_repr,
                "exception_occurred": exception_occurred,
                "exception_info": exception_info,
                "timestamp": datetime.now().isoformat(),
            }

        if inspect.iscoroutinefunction(func):
            async def async_wrapper(self, *args, **kwargs):
                exception_occurred = False
                exception_info = None
                return_value = None

                try:
                    return_value = await original_func(self, *args, **kwargs)
                except Exception as e:
                    exception_occurred = True
                    exception_info = {
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                    raise
                finally:
                    # Record the call history
                    if hasattr(self, "_tool_call_history"):
                        call_record = _create_call_record(
                            args,
                            kwargs,
                            return_value,
                            exception_occurred,
                            exception_info,
                        )
                        self._tool_call_history.append(call_record)
                return return_value

            wrapped_func = async_wrapper
        else:
            # Sync function wrapper using functools.wraps
            @functools.wraps(original_func)
            def sync_wrapper(self, *args, **kwargs):
                exception_occurred = False
                exception_info = None
                return_value = None

                try:
                    return_value = original_func(self, *args, **kwargs)
                except Exception as e:
                    exception_occurred = True
                    exception_info = {
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                    raise
                finally:
                    # Record the call history
                    if hasattr(self, "_tool_call_history"):
                        call_record = _create_call_record(
                            args,
                            kwargs,
                            return_value,
                            exception_occurred,
                            exception_info,
                        )
                        self._tool_call_history.append(call_record)
                return return_value

            wrapped_func = sync_wrapper

        # Set custom attributes
        wrapped_func._tool_info = func._tool_info  # type: ignore
        wrapped_func._original_func = original_func  # type: ignore

        return wrapped_func  # type: ignore

    return tool_decorator


def _serialize_to_literal(value: Any) -> Any:
    """
    Serialize a value to a JSON-serializable literal representation.

    Args:
        value: The value to serialize

    Returns:
        A JSON-serializable representation of the value
    """
    try:
        # Try to serialize directly
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        # If not directly serializable, convert to string representation
        try:
            return repr(value)
        except Exception:
            return str(type(value).__name__)


class EnvMeta(type):
    """
    Metaclass: Automatically collect the methods decorated with @tool and register them to tool_manager
    """

    def __new__(cls, name, bases, namespace, **kwargs):
        new_class = super().__new__(cls, name, bases, namespace, **kwargs)

        registered_tools: Dict[str, Any] = {}
        readonly_tools: Dict[str, bool] = {}
        tool_kinds: Dict[str, str | None] = {}
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and hasattr(attr_value, "_tool_info"):
                tool_info = attr_value._tool_info  # type: ignore
                # tool_info is a dict, containing "mcp.Tool", "readonly", and "kind"
                tool_obj = tool_info["mcp.Tool"]
                tool_obj.fn = attr_value  # Bind the actual function
                tool_key = tool_obj.name if tool_obj.name else attr_name
                registered_tools[tool_key] = tool_obj
                readonly_tools[tool_key] = tool_info.get("readonly", False)
                tool_kinds[tool_key] = tool_info.get("kind", None)
        new_class._registered_tools = registered_tools  # type: ignore
        new_class._readonly_tools = readonly_tools  # type: ignore
        new_class._tool_kinds = tool_kinds  # type: ignore
        return new_class


class EnvBase(metaclass=EnvMeta):
    def __init__(self):
        self.t = datetime.now()

        # Initialize tool call history storage
        self._tool_call_history: list[dict[str, Any]] = []

        # Replay writer for storing simulation state
        self._replay_writer: Optional["ReplayWriter"] = None

        tools = list(getattr(self.__class__, "_registered_tools", {}).values())
        self._tool_manager = ToolManager(tools=tools)
        self._readonly_llm_tools: list[ChatCompletionToolParam] = []
        self._llm_tools: list[ChatCompletionToolParam] = []
        """
        Tool Schema for LLM to call
        """
        for t in self._tool_manager.list_tools():
            parameters = deepcopy(t.parameters)
            # remove self
            if "properties" in parameters and "self" in parameters["properties"]:
                del parameters["properties"]["self"]
            if "required" in parameters and "self" in parameters["required"]:
                parameters["required"].remove("self")
            # convert format
            func: ChatCompletionToolParam = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": parameters,
                },
            }
            self._llm_tools.append(func)
            if self._readonly_tools.get(t.name, False):  # type: ignore
                self._readonly_llm_tools.append(func)

    def _set_llm_tools(self, tools: list[ChatCompletionToolParam]):
        self._llm_tools = tools

    @property
    def name(self):
        """Name of the environment module"""
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """Description of the environment module for router selection and function calling"""
        return """EnvBase is an abstract class for environment modules. DO NOT USE IT DIRECTLY.
It contains no functions or methods.
"""

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP environment module candidate list.
        This method should be overridden by subclasses to provide a description
        suitable for MCP server to list available environment modules.

        Returns:
            A string description of the environment module for MCP registration.
        """
        return f"{cls.__name__}: {cls.__doc__ or 'No description available'}"

    async def init(self, start_datetime: datetime):
        """
        Initialize the environment module.
        """
        self.t = start_datetime

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step.

        - **Args**:
            - tick: The number of ticks of this simulation step.
            - t: The current datetime of the simulation after this step with the ticks.
        """
        raise NotImplementedError

    async def close(self):
        """
        Close the environment module.
        """
        ...

    # ---- Dump & Load ----
    def _dump_state(self) -> dict:
        """
        Hook for subclasses to dump internal state. Override if needed.
        """
        return {}

    def _load_state(self, state: dict):
        """
        Hook for subclasses to load internal state. Override if needed.
        """
        return None

    async def dump(self) -> dict:
        """
        Dump the environment module state to a serializable dict.
        """
        return {
            "name": self.name,
            "t": self.t.isoformat(),
            "state": self._dump_state(),
        }

    async def load(self, dump_data: dict):
        """
        Load the environment module state from a dict produced by dump().
        """
        try:
            t_str = dump_data.get("t")
            if isinstance(t_str, str) and len(t_str) > 0:
                from datetime import datetime as _dt

                self.t = _dt.fromisoformat(t_str)
        except Exception:
            # keep current t on parse failure
            pass
        state = dump_data.get("state") or {}
        if isinstance(state, dict):
            self._load_state(state)

    def get_tool_call_history(self) -> list[dict[str, Any]]:
        """
        Get the tool call history for this environment module.

        Returns:
            A list of call records, each containing:
            - function_name: str
            - kwargs: dict (serialized, unified kwargs dict including both positional and keyword arguments)
            - return_value: Any (serialized, None if exception occurred)
            - exception_occurred: bool
            - exception_info: dict | None (type and message if exception occurred)
            - timestamp: str (ISO format)
            
        Note:
            The kwargs dict contains all arguments (both positional and keyword) unified into
            a single dict with parameter names as keys, according to the function signature.
        """
        return self._tool_call_history.copy()

    def reset_tool_call_history(self):
        """
        Reset the tool call history for this environment module.
        """
        self._tool_call_history.clear()

    # ==================== Replay Data Methods ====================

    def set_replay_writer(self, writer: "ReplayWriter") -> None:
        """Set the replay data writer for this environment module.

        Args:
            writer: The ReplayWriter instance to use for storing replay data.
        """
        self._replay_writer = writer
