"""
Create logger named EAPRouter as singleton for ray.
"""

import json
import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

__all__ = [
    "get_logger",
    "set_logger_level",
    "setup_litellm_logging",
    "setup_logging",
]


def get_logger():
    logger = logging.getLogger("EAPRouter")
    # check if there is already a handler, avoid duplicate output
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        # set propagate to False, avoid duplicate output
        logger.propagate = False
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_logger_level(level: str):
    """Set the logger level"""
    get_logger().setLevel(level)


class LiteLLMLogger:
    """
    Custom logger for LiteLLM that logs prompts, responses, tokens, and timing.
    
    This class implements the callback interface for LiteLLM to log:
    - Prompts at DEBUG level
    - Responses at INFO level
    - Token usage and timing at INFO level
    """

    def __init__(self):
        self.logger = get_logger()
        self._call_start_times: Dict[str, float] = {}

    def log_pre_api_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ):
        """
        Log before API call - records the prompt at DEBUG level.
        
        Args:
            model: Model name
            messages: List of messages (prompt)
            kwargs: Additional arguments
        """
        # Generate a unique call ID for tracking
        call_id = f"{model}_{int(time.time() * 1000000)}"
        self._call_start_times[call_id] = time.time()
        
        # Format messages for logging
        prompt_text = self._format_messages(messages)
        
        # Log prompt at DEBUG level
        self.logger.debug(
            f"[LiteLLM] Model: {model} | Prompt:\n{prompt_text}"
        )
        
        # Store call_id in kwargs for later retrieval
        kwargs["_litellm_call_id"] = call_id

    def log_post_api_call(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ):
        """
        Log after API call - records response, tokens, and timing at INFO level.
        
        Args:
            kwargs: Arguments passed to the API call
            response_obj: Response object from LiteLLM
            start_time: Start time of the call
            end_time: End time of the call
        """
        duration = end_time - start_time
        model = kwargs.get("model", "unknown")
        
        # Extract response content
        response_content = self._extract_response_content(response_obj)
        
        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        if hasattr(response_obj, "usage") and response_obj.usage:
            input_tokens = getattr(response_obj.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response_obj.usage, "completion_tokens", 0) or 0
            total_tokens = getattr(response_obj.usage, "total_tokens", 0) or 0
        elif isinstance(response_obj, dict):
            usage = response_obj.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
            total_tokens = usage.get("total_tokens", 0) or 0
        
        # Log response at INFO level
        self.logger.info(
            f"[LiteLLM] Model: {model} | Response:\n{response_content}"
        )
        
        # Log token usage and timing at INFO level
        self.logger.info(
            f"[LiteLLM] Model: {model} | Tokens - Input: {input_tokens}, "
            f"Output: {output_tokens}, Total: {total_tokens} | "
            f"Duration: {duration:.3f}s"
        )

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format messages list into a readable string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted string representation of messages
        """
        formatted_parts = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Handle different content types
            if isinstance(content, list):
                # Handle content array (e.g., multimodal)
                content_str = json.dumps(content, ensure_ascii=False, indent=2)
            elif isinstance(content, str):
                content_str = content
            else:
                content_str = str(content)
            
            formatted_parts.append(f"[{role}]: {content_str}")
            
            # Include tool_calls if present
            if "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                formatted_parts.append(f"  Tool calls: {json.dumps(tool_calls, ensure_ascii=False, indent=2)}")
        
        return "\n".join(formatted_parts)

    def _extract_response_content(self, response_obj: Any) -> str:
        """
        Extract response content from LiteLLM response object.
        
        Args:
            response_obj: Response object from LiteLLM
            
        Returns:
            Response content as string
        """
        try:
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "message"):
                    content = getattr(choice.message, "content", None)
                    if content:
                        return content
                    # Check for tool calls
                    if hasattr(choice.message, "tool_calls"):
                        tool_calls = choice.message.tool_calls
                        if tool_calls:
                            return f"[Tool calls: {len(tool_calls)}]"
            elif isinstance(response_obj, dict):
                choices = response_obj.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    if content:
                        return content
                    # Check for tool calls
                    tool_calls = message.get("tool_calls")
                    if tool_calls:
                        return f"[Tool calls: {len(tool_calls)}]"
        except Exception as e:
            return f"[Error extracting response: {str(e)}]"
        
        return "[Empty or unknown response]"


def setup_litellm_logging():
    """
    Setup LiteLLM logging by registering the custom logger callback.
    This should be called once during application initialization.
    """
    try:
        # Suppress Pydantic serialization warnings from LiteLLM/OpenAI Message and Choices types.
        # See: https://github.com/BerriAI/litellm/issues/11759
        # The fix is in open PR https://github.com/BerriAI/litellm/pull/16299
        warnings.filterwarnings(
            "ignore",
            message=".*PydanticSerializationUnexpectedValue.*",
            category=UserWarning,
        )
        import litellm
        
        # Create and register the custom logger
        litellm_logger = LiteLLMLogger()
        litellm.callbacks = [litellm_logger]
        
        # Configure litellm's internal logger to avoid duplicate logs
        litellm_logging = logging.getLogger("litellm")
        litellm_logging.handlers.clear()
        litellm_logging.propagate = False
        # Set to WARNING to avoid duplicate logs (we handle logging ourselves)
        litellm_logging.setLevel(logging.WARNING)
        
        get_logger().info("LiteLLM logging initialized")
    except ImportError:
        get_logger().warning("litellm not available, skipping LiteLLM logging setup")
    except Exception as e:
        get_logger().warning(f"Failed to setup LiteLLM logging: {e}")


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Setup comprehensive logging configuration for the application.
    
    This function configures:
    - Root logger
    - EAPRouter logger
    - LiteLLM logger integration
    
    Args:
        log_file: Optional path to log file. If provided, logs will be written to file.
                 If None, logs will only go to console.
        log_level: Logging level (default: logging.DEBUG)
        log_format: Custom log format string. If None, uses default format.
                   Default: "%(message)s" for file, full format for console
        console_output: Whether to output logs to console (default: True)
    
    Returns:
        Configured EAPRouter logger instance
    """
    # Default format
    if log_format is None:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        file_formatter = logging.Formatter(log_format)
        console_formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers.clear()
    
    # Add file handler if log_file is provided
    if log_file:
        # Create logs directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Configure EAPRouter logger
    logger = get_logger()
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False
    
    # Add handlers to EAPRouter logger
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Setup LiteLLM logging
    setup_litellm_logging()
    
    # Configure litellm internal logger to avoid duplicate logs
    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.handlers.clear()
    litellm_logger.propagate = False
    
    return logger


