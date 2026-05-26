"""
EAPRouter implementation.

This module grounds agent instructions and structured context into restricted
Python executable action programs (EAPs), validates them, executes them over
registered environment modules, and reuses them through predefined EAPs and a
template cache.
"""

import ast
import asyncio
import inspect
import json
import math
import os
import pickle
import random
import re
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from eaprouter.storage import ReplayWriter

import faiss
import numpy as np
from eaprouter.config import Config
from eaprouter.env.base import EnvBase
from eaprouter.env.benchmark import (
    EnvRouterBenchmarkData,
)
from eaprouter.env.router_base import RouterBase
from eaprouter.logger import get_logger
from litellm import AllMessageValues, aembedding

__all__ = ["AskContext", "EAPRouter", "CodeGenRouter"]


@dataclass
class CacheStats:
    """"""

    request_count: int = 0
    predefined_hit_count: int = 0
    cache_hit_count: int = 0  # 
    cache_miss_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    code_execution_success_count: int = 0  # 
    code_execution_failure_count: int = 0  # 
    total_code_retry_count: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Internal helper."""
        total = self.cache_hit_count + self.cache_miss_count
        return self.cache_hit_count / total if total > 0 else 0.0

    @property
    def code_execution_success_rate(self) -> float:
        """Internal helper."""
        total = self.code_execution_success_count + self.code_execution_failure_count
        return self.code_execution_success_count / total if total > 0 else 0.0

    @property
    def avg_input_tokens(self) -> float:
        """Internal helper."""
        return (
            self.total_input_tokens / self.request_count
            if self.request_count > 0
            else 0.0
        )

    @property
    def avg_output_tokens(self) -> float:
        """Internal helper."""
        return (
            self.total_output_tokens / self.request_count
            if self.request_count > 0
            else 0.0
        )

    @property
    def avg_retry_count(self) -> float:
        """"""
        return (
            self.total_code_retry_count / self.request_count
            if self.request_count > 0
            else 0.0
        )


@dataclass
class CacheEntry:
    """"""

    instruction_template: str  # 
    variable_keys: tuple[str, ...]  # 
    variable_types: dict[str, str]  #  {key: type_name}
    code: str
    embedding: Optional[np.ndarray] = None
    env_class_type: str = ""  # nv module classes
    entry_id: Optional[int] = None  # D
    success_count: int = 0  # 
    failure_count: int = 0  # 
    last_used: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)  # 

    @property
    def total_usage(self) -> int:
        """Internal helper."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Internal helper."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


OBSERVE_INSTRUCTION = (
    "Builtin observe has collected all readonly kind='observe' tools into result['observations']. "
    "Summarize the situational information for the agent in clear natural language."
)
STATISTIC_INSTRUCTION = "Collect environment statistics by calling all available statistic tools. Store all statistics results in result['statistics']."


@dataclass
class AskContext:
    """Internal helper."""

    # ===  ===
    ctx: dict
    instruction: str
    readonly: bool
    template_mode: bool

    # ===  ===
    variables: dict = field(default_factory=dict)
    instruction_stripped: str = ""
    is_observe_or_statistic: bool = False
    resolved_instruction: str = ""


    code: Optional[str] = None
    cache_entry: Optional["CacheEntry"] = None
    cache_miss_reason: Optional[str] = None
    code_source: Optional[str] = None  # "predefined" | "cache" | "llm" | "builtin"


    retry_count: int = 0
    previous_code: Optional[str] = None
    previous_errors: List[str] = field(default_factory=list)
    dialog_history: List[AllMessageValues] = field(default_factory=list)

    # ===  ===
    execution_result: Optional[Dict[str, Any]] = None
    execution_attempted: bool = False
    success_data: Optional[Dict[str, Any]] = (
        None  #  {ctx, instruction, results, process_text, status, error, code}
    )

    # ===  ===
    final_answer: str = ""
    results: dict = field(default_factory=dict)
    early_return: Optional[Tuple[dict, str]] = (
        None
    )
    token_usage_responses: List[Dict[str, int]] = field(
        default_factory=list
    )

    def __post_init__(self):
        stripped = self.instruction.strip()
        self.instruction_stripped = (
            "<statistic>" if stripped == "<statistics>" else stripped
        )
        self.is_observe_or_statistic = self.instruction_stripped in (
            "<observe>",
            "<statistic>",
        )
        self.variables = self.ctx.get("variables", {})
        self.resolved_instruction = self.instruction
        if self.instruction_stripped == "<observe>":
            self.resolved_instruction = OBSERVE_INSTRUCTION
        elif self.instruction_stripped == "<statistic>":
            self.resolved_instruction = STATISTIC_INSTRUCTION

    @property
    def is_observe_or_statistics(self) -> bool:
        """Backward-compatible alias for older internal observers."""
        return self.is_observe_or_statistic


def _get_debug_info(description: str = "") -> str:
    """Internal helper."""
    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_frame = frame.f_back
        filename = os.path.basename(caller_frame.f_code.co_filename)
        lineno = caller_frame.f_lineno
        return f"[{filename}:{lineno}] {description}"
    return description


async def clean_coroutines_from_results(results: dict) -> dict:
    """Internal helper."""
    visited: set[int] = set()

    async def clean_value(value: Any) -> Any:
        if inspect.iscoroutine(value):
            try:
                return await value
            except Exception as e:
                get_logger().warning(f"Failed to await coroutine: {e!s}")
                return f"<unawaited_coroutine: {type(value).__name__}>"
        elif isinstance(value, dict):
            obj_id = id(value)
            if obj_id in visited:
                return "<circular_reference>"
            visited.add(obj_id)
            try:
                return {k: await clean_value(v) for k, v in value.items()}
            finally:
                visited.remove(obj_id)
        elif isinstance(value, (list, tuple)):
            obj_id = id(value)
            if obj_id in visited:
                return "<circular_reference>"
            visited.add(obj_id)
            try:
                cleaned = [await clean_value(item) for item in value]
                return cleaned if isinstance(value, list) else tuple(cleaned)
            finally:
                visited.remove(obj_id)
        return value

    return await clean_value(results)


def _get_env_class_type_key(env_modules: list) -> str:
    """Internal helper."""
    keys = []
    for m in env_modules:
        cls = m.__class__
        full_name = f"{cls.__module__}.{cls.__qualname__}"
        keys.append((m.name, full_name))
    return json.dumps(sorted(keys), sort_keys=True, ensure_ascii=False)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def _compact_results_text(results: Dict[str, Any]) -> str:
    try:
        return json.dumps(
            results, ensure_ascii=False, separators=(",", ":"), default=str
        )
    except TypeError:
        return str(results)


def _build_deterministic_final_answer(
    router: "EAPRouter", success_data: Dict[str, Any]
) -> str:
    time_tag = f"[{router.t.strftime('%A')}, {router.t.strftime('%Y-%m-%d %H:%M:%S')}]"
    status = success_data.get("status", "unknown")
    results = success_data.get("results", {})
    reason = results.get("reason") or success_data.get("error")
    process_text = _strip_code_fence(success_data.get("process_text", ""))

    if status in {"fail", "error"} and reason:
        body = str(reason).strip()
    elif process_text and process_text != "":
        body = process_text
    elif reason:
        body = str(reason).strip()
    else:
        body = _compact_results_text(results)

    return f"{time_tag} {body}" if body else time_tag


class TemplateCacheDB:
    """Internal helper."""

    def __init__(
        self,
        cache_path: str,
        embedding_dims: int,
        max_size_per_env: int = 1000,
    ):
        self._cache_path = cache_path
        self._embedding_dims = embedding_dims
        self._max_size_per_env = max_size_per_env
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    def _load_data(self) -> dict:
        """Internal helper."""
        if not os.path.exists(self._cache_path):
            return {"next_id": 1, "by_env": {}}
        try:
            with open(self._cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            get_logger().warning(f"Failed to load cache from {self._cache_path}: {e}")
            return {"next_id": 1, "by_env": {}}

    def _save_data(self, data: dict) -> None:
        """Internal helper."""
        with open(self._cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_entries(
        self, env_class_type: str
    ) -> Tuple[List[CacheEntry], Optional["faiss.Index"], List[int]]:
        """Internal helper."""
        data = self._load_data()
        raw = data.get("by_env", {}).get(env_class_type, [])
        raw = sorted(raw, key=lambda e: e.last_used, reverse=True)[
            : self._max_size_per_env
        ]

        entries: List[CacheEntry] = []
        embeddings: List[np.ndarray] = []
        entry_indices: List[int] = []

        for i, e in enumerate(raw):
            e.env_class_type = env_class_type
            entries.append(e)
            if e.embedding is not None:
                embeddings.append(e.embedding)
                entry_indices.append(i)

        if not embeddings:
            return entries, None, []

        emb_array = np.asarray(embeddings, dtype=np.float32).copy()
        faiss.normalize_L2(emb_array)
        index = faiss.IndexFlatIP(len(embeddings[0]))
        index.add(emb_array)
        return entries, index, entry_indices

    def add_entry(
        self,
        env_class_type: str,
        entry: CacheEntry,
    ) -> int:
        """Internal helper."""
        data = self._load_data()
        next_id = data.get("next_id", 1)
        by_env = data.setdefault("by_env", {})
        lst = by_env.setdefault(env_class_type, [])

        entry.entry_id = next_id
        entry.env_class_type = env_class_type
        lst.append(entry)
        lst.sort(key=lambda e: e.last_used, reverse=True)
        by_env[env_class_type] = lst[: self._max_size_per_env]

        data["next_id"] = next_id + 1
        self._save_data(data)
        return next_id

    def update_entry(
        self,
        env_class_type: str,
        entry_id: int,
        success: bool,
        code: str,
    ) -> None:
        """Internal helper."""
        data = self._load_data()
        lst = data.get("by_env", {}).get(env_class_type, [])
        for e in lst:
            if e.entry_id == entry_id:
                e.last_used = datetime.now()
                e.code = code
                if success:
                    e.success_count += 1
                else:
                    e.failure_count += 1
                break
        self._save_data(data)

    def clear_env_cache(self, env_class_type: str) -> None:
        """Internal helper."""
        data = self._load_data()
        data.setdefault("by_env", {}).pop(env_class_type, None)
        self._save_data(data)

    def find_by_instruction(
        self,
        env_class_type: str,
        instruction_template: str,
    ) -> Optional[CacheEntry]:
        """Internal helper."""
        data = self._load_data()
        lst = data.get("by_env", {}).get(env_class_type, [])
        for e in lst:
            if e.instruction_template == instruction_template:
                return e
        return None






class AskObserver(Protocol):
    """Internal helper."""

    async def on_final(self, context: AskContext) -> None: ...


# ==================== Code  ====================


class CodeProvider(Protocol):
    """Code Predefined """

    @property
    def name(self) -> str:
        """ context.code_source"""
        ...

    async def get_code(
        self, context: AskContext, router: "EAPRouter"
    ) -> Optional[str]:
        """Internal helper."""
        ...





class PipelineStage(Protocol):
    """Internal helper."""

    async def process(
        self, context: AskContext, router: "EAPRouter"
    ) -> AskContext: ...





class InstructionLogObserver:
    """Internal helper."""

    def __init__(self, router: "EAPRouter"):
        self._router = router

    async def on_final(self, context: AskContext) -> None:

        ctx = (
            context.ctx
            if context.execution_attempted
            else await clean_coroutines_from_results(context.ctx)
        )
        log_entry = EnvRouterBenchmarkData(
            instruction=context.instruction,
            context=ctx,
            readonly=context.readonly,
        )
        async with self._router._instruction_log_lock:
            self._router._instruction_log.append(log_entry)
            try:
                with open(self._router._log_path, "wb") as f:
                    pickle.dump(self._router._instruction_log, f)
            except Exception as e:
                get_logger().warning(
                    f"Failed to pickle instruction log: {e!s}, skipping file write"
                )


class CacheStatsObserver:
    """Internal helper."""

    def __init__(self, router: "EAPRouter"):
        self._router = router

    async def on_final(self, context: AskContext) -> None:
        async with self._router._cache_stats_lock:
            self._router._cache_stats.request_count += 1
            if context.code_source in ("predefined", "builtin"):
                self._router._cache_stats.predefined_hit_count += 1
            elif context.cache_entry:
                self._router._cache_stats.cache_hit_count += 1
            elif context.template_mode and not context.is_observe_or_statistics:
                self._router._cache_stats.cache_miss_count += 1
            if context.execution_attempted:
                if context.success_data:
                    self._router._cache_stats.code_execution_success_count += 1
                else:
                    self._router._cache_stats.code_execution_failure_count += 1
                self._router._cache_stats.total_code_retry_count += context.retry_count
            for tu in context.token_usage_responses:
                self._router._cache_stats.total_input_tokens += tu.get(
                    "input_tokens", 0
                )
                self._router._cache_stats.total_output_tokens += tu.get(
                    "output_tokens", 0
                )


class CacheAddObserver:
    """Internal helper."""

    def __init__(self, router: "EAPRouter"):
        self._router = router

    @staticmethod
    async def _add_to_cache(
        router: "EAPRouter",
        instruction: str,
        variables: dict,
        code: str,
        success: bool = True,
    ) -> None:
        if not router._template_cache_enabled:
            return
        async with router._template_cache_lock:
            variable_keys = tuple(sorted(variables.keys()))
            variable_types = {k: type(v).__name__ for k, v in variables.items()}
            embedding = await CacheCodeProvider._compute_embedding(router, instruction)
            existing = router._cache_db.find_by_instruction(
                router._env_class_type_key, instruction
            )
            if existing and existing.entry_id is not None:
                router._cache_db.update_entry(
                    router._env_class_type_key, existing.entry_id, success, code
                )
                for e in router._cache_entries:
                    if e.instruction_template == instruction:
                        e.last_used = datetime.now()
                        e.code = code
                        if success:
                            e.success_count += 1
                        else:
                            e.failure_count += 1
                        break
                return
            entry = CacheEntry(
                instruction_template=instruction,
                variable_keys=variable_keys,
                variable_types=variable_types,
                code=code,
                embedding=embedding,
                env_class_type=router._env_class_type_key,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
            )
            new_id = router._cache_db.add_entry(router._env_class_type_key, entry)
            entry.entry_id = new_id
            idx = len(router._cache_entries)
            router._cache_entries.append(entry)
            if embedding is not None:
                router._cache_faiss_entry_indices.append(idx)
                emb = embedding.astype(np.float32).reshape(1, -1).copy()
                faiss.normalize_L2(emb)
                if router._cache_faiss_index is None:
                    router._cache_faiss_index = faiss.IndexFlatIP(emb.shape[1])
                assert router._cache_faiss_index is not None
                router._cache_faiss_index.add(emb)

    async def on_final(self, context: AskContext) -> None:
        if context.success_data is None:
            return
        get_logger().info(f"Try to cache: {context.instruction[:100]}...")
        get_logger().info(f"context.template_mode: {context.template_mode}")
        get_logger().info(f"context.cache_entry: {context.cache_entry}")
        get_logger().info(
            f"context.is_observe_or_statistics: {context.is_observe_or_statistics}"
        )
        sd = context.success_data
        if (
            context.template_mode
            and not context.cache_entry
            and not context.is_observe_or_statistics
        ):
            get_logger().info(f"Adding to cache: {context.instruction[:100]}...")
            await self._add_to_cache(
                self._router,
                context.instruction,
                context.variables,
                sd["code"],
                success=True,
            )





class PredefinedCodeProvider:
    """Pre-generated EAP provider for special requests such as <statistic>."""

    @property
    def name(self) -> str:
        return "predefined"

    async def get_code(
        self, context: AskContext, router: "EAPRouter"
    ) -> Optional[str]:
        if not context.is_observe_or_statistic:
            return None
        if context.instruction_stripped == "<statistic>":
            return router._statistics_code if router._statistics_code else None
        return None


class CacheCodeProvider:
    """Internal helper."""

    @property
    def name(self) -> str:
        return "cache"

    @staticmethod
    async def _compute_embedding(
        router: "EAPRouter", text: str
    ) -> Optional[np.ndarray]:
        try:
            async with router._embedding_cache_lock:
                if text in router._embedding_cache:
                    return router._embedding_cache[text]
            response = await aembedding(
                model=f"openai/{router._embedding_model}",
                input=[text],
                api_key=router._embedding_api_key,
                api_base=router._embedding_api_base,
            )
            if response and hasattr(response, "data") and len(response.data) > 0:
                emb = np.array(response.data[0]["embedding"], dtype=np.float32)
                async with router._embedding_cache_lock:
                    if len(router._embedding_cache) < 10000:
                        router._embedding_cache[text] = emb
                return emb
            return None
        except Exception as e:
            get_logger().warning(f"Failed to compute embedding: {e}")
            return None

    @staticmethod
    async def _lookup(
        router: "EAPRouter", instruction: str, variables: dict
    ) -> Tuple[Optional[CacheEntry], Optional[str]]:
        if not router._template_cache_enabled:
            return None, "template_cache_disabled"
        async with router._template_cache_lock:
            emb = await CacheCodeProvider._compute_embedding(router, instruction)
            if emb is None:
                return None, "embedding_unavailable"
            current_keys = set(variables.keys())
            best_match, best_sim = None, 0.0
            saw_compatible_candidate = False
            saw_key_incompatible_candidate = False
            if router._cache_faiss_index and router._cache_faiss_entry_indices:
                query = np.asarray([emb], dtype=np.float32).copy()
                faiss.normalize_L2(query)
                k = min(32, router._cache_faiss_index.ntotal)
                scores, indices = router._cache_faiss_index.search(query, k)
                for score, idx in zip(scores[0], indices[0], strict=False):
                    if idx < 0:
                        break
                    entry_idx = router._cache_faiss_entry_indices[idx]
                    entry = router._cache_entries[entry_idx]
                    if entry.env_class_type != router._env_class_type_key:
                        continue
                    cached_keys = set(entry.variable_keys)
                    if not (
                        current_keys.issubset(cached_keys)
                        or cached_keys.issubset(current_keys)
                    ):
                        saw_key_incompatible_candidate = True
                        continue
                    saw_compatible_candidate = True
                    sim = float(score)
                    if (
                        sim >= router._template_cache_similarity_threshold
                        and sim > best_sim
                    ):
                        best_sim, best_match = sim, entry
            if best_match:
                best_match.last_used = datetime.now()
                return best_match, None
            if saw_compatible_candidate:
                return None, "below_similarity_threshold"
            if saw_key_incompatible_candidate:
                return None, "variable_keys_incompatible"
            return None, "no_similar_entry"

    async def get_code(
        self, context: AskContext, router: "EAPRouter"
    ) -> Optional[str]:
        if (
            not router._template_cache_enabled
            or not context.template_mode
            or context.is_observe_or_statistic
        ):
            return None
        cache_entry, miss_reason = await self._lookup(
            router, context.instruction, context.variables
        )
        if cache_entry is not None:
            context.cache_entry = cache_entry
            return cache_entry.code
        context.cache_miss_reason = miss_reason
        get_logger().info(
            "Template cache miss: reason=%s instruction=%s",
            miss_reason,
            context.instruction[:100],
        )
        return None


class LLMCodegenProvider:
    """LLM  LLM """

    @property
    def name(self) -> str:
        return "llm"

    @staticmethod
    def _build_prompt(
        router: "EAPRouter",
        instruction: str,
        ctx: dict,
        readonly: bool,
        kind: str | None = None,
    ) -> str:
        key = (readonly, kind)
        tools_pyi = router._tools_pyi_dict[key]
        template_note = ""
        if isinstance(ctx.get("variables"), dict) and ctx["variables"]:
            template_note = (
                "\n## Template Mode Hint\n"
                "- Runtime values are available in context['variables'].\n"
                "- Prefer reading changing values from context['variables'] instead of hard-coding literals.\n"
                "- This keeps the EAP reusable and improves template-cache hits.\n"
            )
        return """# Executable Action Program Generation Task
You generate a restricted Python executable action program (EAP) that grounds the agent intent into valid environment-module action calls.

## Available Environment Modules and Tools
```python
{tools_pyi}
```
```python
modules = {router._modules!r}
```

## Agent Input
<instruction>{instruction}</instruction>
```python
context = {ctx!r}
```
{template_note}

## EAP Requirements
1. Generate Python code that accomplishes the instruction by calling appropriate module actions.
2. Use `context` for structured input and `modules` for environment modules.
3. Store the structured output in `result` and MUST set result['status'] at the end: 'success', 'in_progress', 'fail', or 'error'.
4. Use `print()` for textual feedback when useful.

## Important Notes
- Use `context`, `modules`, `result`, `print()`. `ctx` and `results` are legacy aliases, but prefer `context` and `result`.
- Allowed modules: collections, itertools, functools, operator, copy, decimal, fractions, statistics, string, re, datetime, json, math, random, numpy (as np).
- Do NOT use dangerous operations. ALWAYS USE `await` TO CALL TOOLS (ASYNC FUNCTIONS).
- NEVER forget to set result['status'] at the END of your code!

## CRITICAL: Status Handling
When calling environment tools:
- Check the return value for a 'status' field
- If the tool returns successfully, set result['status'] = 'success'
- If the tool indicates an error or failure, set result['status'] = 'fail'
- If the operation is ongoing, set result['status'] = 'in_progress'
- Example code pattern:
  response = await modules["ModuleName"].some_tool(arg1, arg2)
  print("Tool response:", response)
  result["response"] = response
  result["status"] = response.get("status", "success") if isinstance(response, dict) else "success"

## Output Format
Generate ONLY the Python code, without markdown. Start directly with Python statements.

Your EAP:""".format(
            tools_pyi=tools_pyi,
            router=router,
            instruction=instruction,
            ctx=ctx,
            template_note=template_note,
        )

    @staticmethod
    def _build_error_message(previous_errors: List[str]) -> str:
        errors_text = "\n".join(f"- {i+1}. {e}" for i, e in enumerate(previous_errors))
        return """The EAP I generated failed during execution. Here's what went wrong:

## Errors
{errors_text}

Please analyze the errors and fix the EAP. Common issues: incorrect module/action names, wrong parameter types, async/await usage, type mismatches, missing imports, logic errors.

Please generate the corrected EAP:""".format(errors_text=errors_text)

    @staticmethod
    async def _call_llm(
        router: "EAPRouter", context: AskContext
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        try:
            response = await router.acompletion_with_system_prompt(
                model="coder", messages=context.dialog_history
            )
            raw = response.choices[0].message.content or ""  # type: ignore[union-attr]
            pattern = r"```(?:python)?\s*\n(.*?)```"
            matches = re.findall(pattern, raw, re.DOTALL)
            code = matches[0].strip() if matches else raw.strip()
            usage = getattr(response, "usage", None)
            token_usage = None
            if usage is not None:
                token_usage = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                }
            return code, token_usage
        except Exception as e:
            get_logger().error(f"{_get_debug_info('LLM')} - {e}")
            return "", None

    async def get_code(
        self, context: AskContext, router: "EAPRouter"
    ) -> Optional[str]:
        if context.retry_count == 0:
            prompt = self._build_prompt(
                router,
                context.resolved_instruction,
                context.ctx,
                context.readonly,
                None,
            )
            context.dialog_history = [{"role": "user", "content": prompt}]
        else:
            if context.previous_code:
                context.dialog_history.append(
                    {"role": "assistant", "content": context.previous_code}
                )
            context.dialog_history.append(
                {
                    "role": "user",
                    "content": self._build_error_message(context.previous_errors),
                }
            )
        code, token_usage = await self._call_llm(router, context)
        if token_usage:
            context.token_usage_responses.append(token_usage)
        return code.strip() if code else None


# --- Pipeline  ---


class InitStage:
    """Internal helper."""

    async def process(self, context: AskContext, router: "EAPRouter") -> AskContext:
        router._add_current_time_to_ctx(context.ctx)
        if not router.env_modules:
            context.early_return = (
                context.ctx,
                "No environment modules available to handle the request.",
            )
        return context


class CodeStage:
    """Internal helper."""

    @staticmethod
    async def _acquire_code(
        router: "EAPRouter", context: AskContext, retry_only: bool
    ) -> bool:
        providers = (
            router._code_provider_chain[-1:]
            if retry_only
            else router._code_provider_chain
        )
        for provider in providers:
            code = await provider.get_code(context, router)
            if code is not None:
                context.code = code
                context.code_source = provider.name
                return True
        return False

    @staticmethod
    def _validate_code_safety(router: "EAPRouter", code: str) -> Tuple[bool, str]:
        violations = []
        try:
            tree = ast.parse(code, mode="exec")
            dangerous_functions = {
                "eval",
                "exec",
                "compile",
                "__import__",
                "open",
                "input",
            }

            def is_dangerous_module(name: str) -> bool:
                if name in router.ALLOWED_MODULES:
                    return False
                return name in router.DANGEROUS_MODULES or (
                    name.startswith("_") and name != "__future__"
                )

            def _is_truthy_constant(node: ast.expr) -> bool:
                """Check if an AST node is a compile-time truthy constant."""
                if isinstance(node, ast.Constant):
                    return bool(node.value)
                if isinstance(node, ast.Name) and node.id in {"True", "__debug__"}:
                    return True
                if isinstance(node, ast.Num):
                    return bool(node.n)
                return False

            def _loop_has_break(node: ast.AST) -> bool:
                """Check if a while/for loop body contains a reachable break."""
                for child in ast.walk(node):
                    if isinstance(child, ast.Break):
                        return True
                return False

            for node in ast.walk(tree):
                if type(node) in router.FORBIDDEN_AST_NODES:
                    violations.append(f"Forbidden AST node type: {type(node).__name__}")
                elif isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in dangerous_functions
                    ):
                        violations.append(f"Dangerous function call: {node.func.id}()")
                    elif isinstance(node.func, ast.Attribute) and node.func.attr in {
                        "eval",
                        "exec",
                        "compile",
                    }:
                        violations.append(f"Dangerous method call: {node.func.attr}()")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if is_dangerous_module(alias.name):
                            violations.append(f"Dangerous import: import {alias.name}")
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and is_dangerous_module(node.module)
                ):
                    violations.append(
                        f"Dangerous import: from {node.module} import ..."
                    )
                elif isinstance(node, ast.While) and _is_truthy_constant(node.test):
                    if not _loop_has_break(node):
                        violations.append(
                            "Potential infinite loop: while True without break"
                        )

            if violations:
                return (
                    False,
                    "Code safety check failed. Violations found:\n"
                    + "\n".join(f"- {v}" for v in violations),
                )
            return True, ""
        except SyntaxError as e:
            return False, f"Code syntax error: {e!s}"
        except Exception as e:
            return False, f"Code validation error: {e!s}"

    @staticmethod
    async def _execute_code(
        router: "EAPRouter", code: str, ctx: dict, readonly: bool
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        def coerce_result_dict(value: Any) -> Dict[str, Any]:
            if value is None:
                return result
            if isinstance(value, dict):
                return value
            return {"value": value}

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name not in router.ALLOWED_MODULES:
                return None
            return __import__(name, globals, locals, fromlist, level)

        import collections
        import copy
        import decimal
        import fractions
        import functools
        import itertools
        import operator
        import statistics
        import string

        allowed_modules = {
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "operator": operator,
            "copy": copy,
            "decimal": decimal,
            "fractions": fractions,
            "statistics": statistics,
            "string": string,
            "re": re,
            "datetime": datetime,
            "json": json,
            "math": math,
            "random": random,
            "numpy": np,
            "np": np,
        }
        restricted_builtins = {
            k: v for k, v in __builtins__.items() if k in router.ALLOWED_BUILTINS
        }
        restricted_builtins["__import__"] = safe_import
        exec_globals = {
            "__builtins__": restricted_builtins,
            "context": ctx,
            "ctx": ctx,
            "modules": types.MappingProxyType(router._modules),
            "result": result,
            "results": result,
            "print": print,
            **allowed_modules,
            "Exception": Exception,
            "RuntimeError": RuntimeError,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "SyntaxError": SyntaxError,
            "NameError": NameError,
            "AttributeError": AttributeError,
            "IndexError": IndexError,
            "KeyError": KeyError,
        }
        exec_locals = {}
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        try:
            is_async = "async" in code or "await" in code
            local_vars: Dict[str, Any] = {}

            if is_async:

                async def run_async():
                    indented = "\n".join(
                        "    " + line if line.strip() else ""
                        for line in code.split("\n")
                    )
                    async_code = (
                        f"async def _generated_main():\n{indented}\n"
                        "    return locals()"
                    )
                    exec(
                        compile(async_code, "<generated_async>", "exec"),
                        exec_globals,
                        exec_locals,
                    )
                    return await exec_locals["_generated_main"]()

                local_vars = await asyncio.wait_for(run_async(), timeout=10) or {}
                if not isinstance(local_vars, dict):
                    local_vars = {}
            else:
                deadline = time.monotonic() + 10

                def _timeout_tracer(frame, event, arg):
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "Code execution timeout: exceeded 10 seconds limit"
                        )
                    return _timeout_tracer

                compiled = compile(code, "<generated>", "exec")
                sys.settrace(_timeout_tracer)
                try:
                    exec(compiled, exec_globals, exec_locals)
                finally:
                    sys.settrace(None)
                local_vars = exec_locals
            structured_result = result
            if "results" in local_vars:
                legacy_result = coerce_result_dict(local_vars["results"])
                if legacy_result is not result and (
                    legacy_result or not structured_result
                ):
                    structured_result = legacy_result
            if "result" in local_vars:
                local_result = coerce_result_dict(local_vars["result"])
                if local_result is not result and (
                    local_result or not structured_result
                ):
                    structured_result = local_result
            output = captured_output.getvalue()
            print_outputs = [
                line.strip() for line in output.split("\n") if line.strip()
            ]
            return {
                "results": structured_result,
                "output": output,
                "print_outputs": print_outputs,
                "success": True,
            }
        except asyncio.TimeoutError:
            raise TimeoutError("Code execution timeout: exceeded 10 seconds limit") from None
        except TimeoutError:
            raise
        except Exception as e:
            return {
                "results": result,
                "output": captured_output.getvalue(),
                "print_outputs": [],
                "error": str(e),
                "success": False,
            }
        finally:
            sys.stdout = old_stdout

    async def process(self, context: AskContext, router: "EAPRouter") -> AskContext:
        if context.early_return:
            return context
        if context.instruction_stripped == "<observe>":
            context.execution_attempted = True
            context.code_source = "builtin"
            try:
                execution_result = await router._run_builtin_observe(context.ctx)
            except Exception as e:
                execution_result = {
                    "success": False,
                    "error": str(e),
                    "results": {},
                    "print_outputs": [],
                    "output": "",
                }
            context.execution_result = execution_result
            context.ctx = await clean_coroutines_from_results(context.ctx)
            if execution_result.get("results"):
                execution_result["results"] = await clean_coroutines_from_results(
                    execution_result["results"]
                )
            if not execution_result.get("success", False):
                err = execution_result.get("error", "observe failed")
                context.early_return = (context.ctx, err)
                return context
            results = execution_result.get("results", {})
            print_outputs = execution_result.get("print_outputs", [])
            proc_err = execution_result.get("error", "")
            if print_outputs:
                process_text = "\n".join(print_outputs)
            else:
                process_text = json.dumps(results, ensure_ascii=False, default=str)[
                    :8000
                ]
            if proc_err:
                process_text += f"\n\nError: {proc_err}"
            process_text = f"```\n{process_text}\n```"
            context.success_data = {
                "ctx": context.ctx,
                "instruction": context.resolved_instruction,
                "results": results,
                "process_text": process_text,
                "status": results.get("status", "unknown"),
                "error": proc_err,
                "code": "<builtin observe>",
            }
            context.results = results
            return context
        max_retries = router.max_llm_call_retry if context.code is None else 0
        while context.retry_count <= max_retries:
            if context.code is None:
                ok = await self._acquire_code(
                    router, context, retry_only=bool(context.previous_errors)
                )
                if not ok:
                    context.retry_count += 1
                    context.previous_errors.append("Failed to generate code from LLM.")
                    context.previous_code = None
                    if context.retry_count > max_retries:
                        context.early_return = (
                            {},
                            "Failed to generate code after retries.",
                        )
                        return context
                    continue
                if context.code_source == "llm":
                    context.dialog_history.append(
                        {"role": "assistant", "content": context.code}
                    )

                assert context.code is not None  # set by _acquire_code when ok=True
                is_safe, safety_violation = self._validate_code_safety(
                    router, context.code
                )
                if not is_safe:
                    context.retry_count += 1
                    context.previous_code = context.code
                    context.previous_errors.append(safety_violation)
                    if context.retry_count > max_retries:
                        context.early_return = (
                            {},
                            f"Generated EAP failed safety check after retries: {safety_violation}",
                        )
                        return context
                    context.code = None
                    continue

            code = context.code
            if code is None:
                return context
            context.execution_attempted = True
            try:
                async with router._execute_lock:
                    execution_result = await self._execute_code(
                        router, code, context.ctx, context.readonly
                    )
            except Exception as e:
                execution_result = {
                    "success": False,
                    "error": str(e),
                    "results": {},
                    "print_outputs": [],
                    "output": "",
                }

            context.execution_result = execution_result

            context.ctx = await clean_coroutines_from_results(context.ctx)
            if execution_result.get("results"):
                execution_result["results"] = await clean_coroutines_from_results(
                    execution_result["results"]
                )
            if not execution_result.get("success", False):
                error = execution_result.get("error", "Unknown error")
                context.retry_count += 1
                context.previous_code = code
                context.previous_errors.append(error)
                if context.retry_count > max_retries:
                    context.early_return = (
                        context.ctx,
                        f"EAP execution failed after retries: {error}",
                    )
                    return context
                context.code = None
                continue

            print_outputs = execution_result.get("print_outputs", [])
            results = execution_result.get("results", {})
            status = results.get("status", "unknown")
            error = execution_result.get("error", "")
            process_text = "\n".join(print_outputs) if print_outputs else "No output"
            if error:
                process_text += f"\n\nError: {error}"
            process_text = f"```\n{process_text}\n```"
            context.success_data = {
                "ctx": context.ctx,
                "instruction": context.resolved_instruction,
                "results": results,
                "process_text": process_text,
                "status": status,
                "error": error,
                "code": code,
            }
            context.results = results
            break
        return context


class SummaryStage:
    """Internal helper."""

    async def process(self, context: AskContext, router: "EAPRouter") -> AskContext:
        if context.early_return:
            return context
        if not context.success_data:
            context.early_return = (
                {},
                "Failed to generate and execute code after all retries.",
            )
            return context
        sd = context.success_data
        context.results = sd["results"]

        if router._final_summary_enabled:
            final_answer, determined_status = await router.generate_final_answer(
                sd["ctx"],
                sd["instruction"],
                sd["results"],
                sd["process_text"],
                sd["status"],
                sd["error"],
            )
        else:
            final_answer = _build_deterministic_final_answer(router, sd)
            determined_status = sd["status"]

        context.results["status"] = determined_status
        context.final_answer = final_answer
        if determined_status == "unknown":
            context.results["status"] = "fail"
            context.results["reason"] = (
                "Generated EAP did not set result['status'], which is mandatory"
            )
        return context


class ObserveFinalStage:
    """Internal helper."""

    async def process(self, context: AskContext, router: "EAPRouter") -> AskContext:
        await router._notify_observers_final(context)
        return context


class EAPRouter(RouterBase):
    """
    EAPRouter grounds LLM agent intents into executable action programs.

    Workflow:
    1. Aggregate action interfaces from registered environment modules.
    2. Prompt the LLM with compact Python-like interface descriptions.
    3. Generate a restricted Python EAP that calls module actions through modules.
    4. Validate EAP syntax and unsafe operations with AST checks.
    5. Execute the EAP with context/modules/result conventions.
    6. Return textual feedback and structured result data to the agent.
    """


    ALLOWED_BUILTINS: ClassVar[set[str]] = {
        "print",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "range",
        "enumerate",
        "zip",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "sorted",
        "reversed",
        "any",
        "all",
        "isinstance",
        "type",
        "getattr",
        "hasattr",
        "dir",
    }


    FORBIDDEN_AST_NODES: ClassVar[set[type]] = {
        ast.ClassDef,
        ast.Delete,  # del
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,  # with
        ast.Assert,  # assert
    }

    # 
    ALLOWED_MODULES: ClassVar[set[str]] = {
        "collections",
        "itertools",
        "functools",
        "operator",
        "copy",
        "decimal",
        "fractions",
        "statistics",
        "string",
        "re",
        "datetime",
        "json",
        "math",
        "random",
        "numpy",
        "np",
    }

    # 
    DANGEROUS_MODULES: ClassVar[set[str]] = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pickle",
        "marshal",
        "ctypes",
        "socket",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "__builtin__",
        "__builtins__",
        "builtins",
    }

    OBSERVE_INSTRUCTION = OBSERVE_INSTRUCTION
    STATISTIC_INSTRUCTION = STATISTIC_INSTRUCTION

    def __init__(
        self,
        env_modules: list[EnvBase],
        max_body_code_lines: int = 15,
        max_steps: int = 10,
        max_llm_call_retry: int = 10,
        log_path: str = "logs/instruction_log.pkl",
        replay_writer: Optional["ReplayWriter"] = None,
        final_summary_enabled: bool = True,
        code_format: str = "raw_code",
        # Template cache configuration
        template_cache_enabled: bool = True,  # 
        template_cache_similarity_threshold: float = 0.85,
        template_cache_max_size: int = 1000,
        template_cache_dir: Optional[
            str
        ] = None,  # None  {Config.HOME_DIR}/eap_router_cache
    ):
        super().__init__(
            env_modules=env_modules,
            max_steps=max_steps,
            max_llm_call_retry=max_llm_call_retry,
            replay_writer=replay_writer,
        )

        # Pre-generate all action interfaces in a dictionary: key is (readonly, kind).
        # kind can be None, "observe", "statistic", etc.
        self._tools_pyi_dict: Dict[Tuple[bool, str | None], str] = {}
        self._log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self._code_format = code_format
        self._max_body_code_lines = max_body_code_lines

        # Collect all tools info once and populate _tools_pyi_dict
        self._populate_tools_pyi_dict()

        self._modules = {module.name: module for module in self.env_modules}

        # Statistic EAP is generated during init; observe uses _run_builtin_observe.
        self._statistics_code = ""

        # Flag to track if LLM EAP generation has been attempted
        self._llm_code_generated = False

        # gentontext
        self._instruction_log: List[EnvRouterBenchmarkData] = []
        self._instruction_log_lock: asyncio.Lock = asyncio.Lock()

        # ==================== Template ====================
        self._final_summary_enabled = final_summary_enabled
        self._template_cache_enabled = template_cache_enabled
        self._template_cache_similarity_threshold = template_cache_similarity_threshold
        self._template_cache_max_size = template_cache_max_size


        self._env_class_type_key = _get_env_class_type_key(env_modules)


        cache_dir = template_cache_dir or os.path.join(
            Config.HOME_DIR, "eap_router_cache"
        )
        cache_path = os.path.join(cache_dir, "cache.pkl")
        self._cache_dir = cache_dir
        self._cache_db = TemplateCacheDB(
            cache_path=cache_path,
            embedding_dims=Config.EMBEDDING_DIMS,
            max_size_per_env=template_cache_max_size,
        )

        self._cache_stats_jsonl_path = os.path.join(cache_dir, "cache_stats.jsonl")
        self._prev_step_stats: Optional[CacheStats] = None
        self._step_index: int = 0
        self._run_id: Optional[str] = None

        #  env  DB  init() 
        self._cache_entries: List[CacheEntry] = []
        self._cache_faiss_index: Optional[faiss.Index] = None
        self._cache_faiss_entry_indices: List[int] = []
        self._template_cache_lock: asyncio.Lock = asyncio.Lock()

        # 
        self._cache_stats = CacheStats()

        # Embedding
        self._embedding_cache: Dict[str, np.ndarray] = {}


        self._embedding_model = Config.EMBEDDING_MODEL
        self._embedding_api_key = Config.EMBEDDING_API_KEY
        self._embedding_api_base = Config.EMBEDDING_API_BASE
        self._embedding_dims = Config.EMBEDDING_DIMS

        #  env odegenenerate_final_answer
        self._execute_lock: asyncio.Lock = asyncio.Lock()
        self._cache_stats_lock: asyncio.Lock = asyncio.Lock()
        self._embedding_cache_lock: asyncio.Lock = asyncio.Lock()

        # nstruction log
        self._observers: List[AskObserver] = [
            InstructionLogObserver(self),
            CacheStatsObserver(self),
            CacheAddObserver(self),
        ]

        # EAP provider chain: predefined -> template cache -> LLM.
        self._code_provider_chain: List[CodeProvider] = [
            PredefinedCodeProvider(),
            CacheCodeProvider(),
            LLMCodegenProvider(),
        ]

    def _get_tools_code(self, tools_info) -> str:
        if self._code_format == "raw_code":
            return self._format_tools_raw_code(tools_info)
        elif self._code_format == "trimmed":
            return self._format_tools_pyi_trimmed(
                tools_info, self._max_body_code_lines
            )
        elif self._code_format.startswith("ratio:"):
            ratio = float(self._code_format.split(":", 1)[1])
            return self._format_tools_pyi_ratio(tools_info, ratio)
        else:  # "pyi"
            return self._format_tools_pyi(tools_info, self._max_body_code_lines)

    def _populate_tools_pyi_dict(self) -> None:
        all_tools_info = self._collect_tools_info()
        all_tools_info = self._filter_tools_info(
            all_tools_info, readonly=None, kind=None
        )
        self._tools_pyi_dict[(False, None)] = self._get_tools_code(all_tools_info)

        readonly_tools_info = self._filter_tools_info(
            all_tools_info, readonly=True, kind=None
        )
        self._tools_pyi_dict[(True, None)] = self._get_tools_code(readonly_tools_info)

        readonly_observe_tools_info = self._filter_tools_info(
            all_tools_info, readonly=True, kind="observe"
        )
        self._tools_pyi_dict[(True, "observe")] = self._get_tools_code(
            readonly_observe_tools_info
        )

        readonly_statistics_tools_info = self._filter_tools_info(
            all_tools_info, readonly=True, kind="statistic"
        )
        self._tools_pyi_dict[(True, "statistic")] = self._get_tools_code(
            readonly_statistics_tools_info
        )

    @staticmethod
    def _resolve_subject_id_from_ctx(ctx: dict) -> Optional[int]:
        for k in ("id", "agent_id", "person_id"):
            if k not in ctx:
                continue
            v = ctx[k]
            if v is None or isinstance(v, bool):
                continue
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.strip():
                try:
                    return int(v.strip(), 10)
                except ValueError:
                    continue
        return None

    @staticmethod
    async def _call_single_observe_tool(
        module: Any, fn: Any, subject_id: Optional[int]
    ) -> Any:

        impl = getattr(fn, "_original_func", fn)
        sig = inspect.signature(impl)
        params = list(sig.parameters.values())
        non_self = [
            p
            for p in (params[1:] if params else [])
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if len(non_self) == 0:
            if inspect.iscoroutinefunction(fn):
                return await fn(module)
            return fn(module)
        if subject_id is None:
            raise ValueError(
                "missing subject id in ctx (need id, agent_id, or person_id; integer 0 is valid)"
            )
        pname = non_self[0].name
        kw = {pname: subject_id}
        if inspect.iscoroutinefunction(fn):
            return await fn(module, **kw)
        return fn(module, **kw)

    async def _run_builtin_observe(self, ctx: dict) -> Dict[str, Any]:
        observe_info = self._filter_tools_info(
            self._collect_tools_info(), readonly=True, kind="observe"
        )
        n_tools = sum(len(m.tools) for m in observe_info.values())
        if n_tools == 0:
            return {
                "success": False,
                "results": {"observations": {}, "status": "fail"},
                "print_outputs": [],
                "output": "",
                "error": "no observe tools registered",
            }
        subject_id = self._resolve_subject_id_from_ctx(ctx)
        observations: Dict[str, Any] = {}
        errors: List[str] = []
        for module_name, module_data in observe_info.items():
            module = self._modules.get(module_name)
            if module is None:
                errors.append(f"{module_name}: module not mounted")
                continue
            reg = getattr(module.__class__, "_registered_tools", {})
            for ti in module_data.tools:
                tname = ti.name
                tool_obj = reg.get(tname)
                fn = getattr(tool_obj, "fn", None) if tool_obj else None
                if not fn:
                    errors.append(f"{module_name}.{tname}: tool missing")
                    continue
                try:
                    out = await self._call_single_observe_tool(module, fn, subject_id)
                    observations[f"{module_name}.{tname}"] = out
                except Exception as e:
                    errors.append(f"{module_name}.{tname}: {e}")
        n_ok = len(observations)
        if n_ok and not errors:
            status = "success"
        elif n_ok and errors:
            status = "partial"
        else:
            status = "fail"
        results: Dict[str, Any] = {"observations": observations, "status": status}
        if errors:
            results["observe_errors"] = errors
        success = n_ok > 0
        return {
            "success": success,
            "results": results,
            "print_outputs": [],
            "output": "",
            "error": "; ".join(errors) if not success else "",
        }

    async def _notify_observers_final(self, context: AskContext) -> None:
        """Internal helper."""
        for obs in self._observers:
            await obs.on_final(context)

    async def ask(
        self,
        ctx: dict,
        instruction: str,
        readonly: bool = False,
        template_mode: bool = False,
    ) -> Tuple[dict, str]:
        """Internal helper."""
        context = AskContext(
            ctx=ctx,
            instruction=instruction,
            readonly=readonly,
            template_mode=template_mode,
        )
        stages: List[PipelineStage] = [
            InitStage(),
            CodeStage(),  #  +  + 
            SummaryStage(),
            ObserveFinalStage(),  #  early_return observer 
        ]
        for stage in stages:
            context = await stage.process(context, self)
        if context.early_return:
            return context.early_return
        return context.results, context.final_answer

    async def init(self, start_datetime: datetime):
        """Internal helper."""
        await super().init(start_datetime)

        # sync
        get_logger().debug("Initialized instruction log lock")


        if self._template_cache_enabled:
            self._load_cache_from_db()


        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate predefined EAPs using LLM if not already done.
        if not self._llm_code_generated:
            # Generate the statistic EAP with the same validation path as runtime EAPs.
            if self._tools_pyi_dict.get((True, "statistic")):
                llm_statistics_code = await self._generate_statistics_code()
                if llm_statistics_code:
                    self._statistics_code = llm_statistics_code
                    get_logger().info("Generated statistic EAP using LLM")
                else:
                    raise ValueError("Failed to generate statistic EAP")
            self._llm_code_generated = True

    async def step(self, tick: int, t: datetime):
        """Internal helper."""
        await super().step(tick, t)


        async with self._cache_stats_lock:
            current = CacheStats(
                request_count=self._cache_stats.request_count,
                predefined_hit_count=self._cache_stats.predefined_hit_count,
                cache_hit_count=self._cache_stats.cache_hit_count,
                cache_miss_count=self._cache_stats.cache_miss_count,
                total_input_tokens=self._cache_stats.total_input_tokens,
                total_output_tokens=self._cache_stats.total_output_tokens,
                code_execution_success_count=self._cache_stats.code_execution_success_count,
                code_execution_failure_count=self._cache_stats.code_execution_failure_count,
                total_code_retry_count=self._cache_stats.total_code_retry_count,
            )
        prev = self._prev_step_stats or CacheStats()
        delta = CacheStats(
            request_count=current.request_count - prev.request_count,
            predefined_hit_count=current.predefined_hit_count
            - prev.predefined_hit_count,
            cache_hit_count=current.cache_hit_count - prev.cache_hit_count,
            cache_miss_count=current.cache_miss_count - prev.cache_miss_count,
            total_input_tokens=current.total_input_tokens - prev.total_input_tokens,
            total_output_tokens=current.total_output_tokens - prev.total_output_tokens,
            code_execution_success_count=current.code_execution_success_count
            - prev.code_execution_success_count,
            code_execution_failure_count=current.code_execution_failure_count
            - prev.code_execution_failure_count,
            total_code_retry_count=current.total_code_retry_count
            - prev.total_code_retry_count,
        )
        exec_total = (
            delta.code_execution_success_count + delta.code_execution_failure_count
        )
        code_execution_success_rate = (
            delta.code_execution_success_count / exec_total if exec_total > 0 else None
        )
        record = {
            "run_id": self._run_id or "",
            "step": self._step_index,
            "request_count": delta.request_count,
            "predefined_hit_count": delta.predefined_hit_count,
            "cache_hit_count": delta.cache_hit_count,
            "cache_miss_count": delta.cache_miss_count,
            "cached_entries_count": len(self._cache_entries),
            "total_input_tokens": delta.total_input_tokens,
            "total_output_tokens": delta.total_output_tokens,
            "code_execution_success_rate": code_execution_success_rate,
        }
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            with open(self._cache_stats_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            get_logger().warning(f"Failed to write cache stats JSONL: {e}")

        self._prev_step_stats = current
        self._step_index += 1
        get_logger().info(self.get_cache_stats_summary())

    async def dump(self) -> dict:
        """
        Dump router state to a serializable dict, including instruction logs and cache stats.
        """

        base_dump = await super().dump()

        # 
        async with self._instruction_log_lock:
            base_dump["instruction_log"] = self._instruction_log.copy()

        # 
        async with self._cache_stats_lock:
            base_dump["cache_stats"] = {
                "request_count": self._cache_stats.request_count,
                "predefined_hit_count": self._cache_stats.predefined_hit_count,
                "cache_hit_count": self._cache_stats.cache_hit_count,
                "cache_miss_count": self._cache_stats.cache_miss_count,
                "total_input_tokens": self._cache_stats.total_input_tokens,
                "total_output_tokens": self._cache_stats.total_output_tokens,
                "code_execution_success_count": self._cache_stats.code_execution_success_count,
                "code_execution_failure_count": self._cache_stats.code_execution_failure_count,
                "total_code_retry_count": self._cache_stats.total_code_retry_count,
            }

        return base_dump

    async def load(self, dump_data: dict):
        """
        Load router state from a dict produced by dump().
        """

        await super().load(dump_data)

        # 
        try:
            instruction_log = dump_data.get("instruction_log", [])
            if isinstance(instruction_log, list):
                async with self._instruction_log_lock:
                    self._instruction_log = instruction_log.copy()
                get_logger().debug(
                    f"Loaded {len(instruction_log)} instruction log entries"
                )
        except Exception as e:
            get_logger().warning(f"Failed to load instruction log: {e!s}")

    async def _generate_initialization_code_with_retry(
        self,
        instruction: str,
        ctx: dict,
        kind: str,
    ) -> str:
        """Generate predefined observe/statistic EAPs through the normal EAP path."""
        llm_provider = LLMCodegenProvider()
        code_stage = CodeStage()
        prompt = llm_provider._build_prompt(self, instruction, ctx, True, kind)
        dialog_history: List[AllMessageValues] = [{"role": "user", "content": prompt}]
        previous_code: Optional[str] = None
        previous_errors: List[str] = []
        retry_count = 0

        while retry_count <= self.max_llm_call_retry:

            tmp_ctx = AskContext(
                ctx=ctx, instruction=instruction, readonly=True, template_mode=False
            )
            tmp_ctx.dialog_history = dialog_history
            tmp_ctx.retry_count = retry_count
            tmp_ctx.previous_code = previous_code
            tmp_ctx.previous_errors = previous_errors

            code, token_usage = await llm_provider._call_llm(self, tmp_ctx)
            if token_usage:
                async with self._cache_stats_lock:
                    self._cache_stats.total_input_tokens += token_usage.get(
                        "input_tokens", 0
                    )
                    self._cache_stats.total_output_tokens += token_usage.get(
                        "output_tokens", 0
                    )

            if not code:
                previous_errors.append("Failed to generate code from LLM.")
                previous_code = None
                if retry_count >= self.max_llm_call_retry:
                    raise ValueError(f"Failed to generate {kind} code after retries.")
                retry_count += 1
                if previous_code:
                    dialog_history.append(
                        {"role": "assistant", "content": previous_code}
                    )
                dialog_history.append(
                    {
                        "role": "user",
                        "content": llm_provider._build_error_message(previous_errors),
                    }
                )
                continue

            dialog_history.append({"role": "assistant", "content": code})
            is_safe, safety_violation = code_stage._validate_code_safety(self, code)
            if not is_safe:
                previous_errors.append(safety_violation)
                previous_code = code
                if retry_count >= self.max_llm_call_retry:
                    raise ValueError(
                        f"Generated {kind} code failed safety check after retries: {safety_violation}"
                    )
                retry_count += 1
                dialog_history.append(
                    {
                        "role": "user",
                        "content": llm_provider._build_error_message(previous_errors),
                    }
                )
                continue

            try:
                async with self._execute_lock:
                    execution_result = await code_stage._execute_code(
                        self, code, ctx, True
                    )
                if not execution_result.get("success", False):
                    previous_errors.append(
                        f"Execution failed: {execution_result.get('error', 'Unknown error')}"
                    )
                    previous_code = code
                    if retry_count >= self.max_llm_call_retry:
                        raise ValueError(
                            f"Generated {kind} code failed execution after retries."
                        )
                    retry_count += 1
                    dialog_history.append(
                        {
                            "role": "user",
                            "content": llm_provider._build_error_message(
                                previous_errors
                            ),
                        }
                    )
                    continue
            except Exception as e:
                previous_errors.append(f"Execution exception: {e!s}")
                previous_code = code
                if retry_count >= self.max_llm_call_retry:
                    raise ValueError(
                        f"Generated {kind} code failed execution after retries: {e!s}"
                    ) from e
                retry_count += 1
                dialog_history.append(
                    {
                        "role": "user",
                        "content": llm_provider._build_error_message(previous_errors),
                    }
                )
                continue

            return code.strip()
        raise ValueError(f"Failed to generate {kind} code after retries.")

    async def _generate_statistics_code(self) -> str:
        """
        Generate an EAP that calls all actions marked as kind="statistic".
        It uses the same generation, validation, and execution path as runtime EAPs.

        :returns: generated Python EAP, or an empty string if generation fails.
        """
        get_logger().debug(f"{_get_debug_info('start generating statistic EAP')}")

        if not self._tools_pyi_dict.get((True, "statistic")):
            get_logger().debug(
                f"{_get_debug_info('no statistic actions')} - skip EAP generation"
            )
            return ""

        instruction = self.STATISTIC_INSTRUCTION
        ctx = {}  # Minimal context for validation execution.
        return await self._generate_initialization_code_with_retry(
            instruction=instruction, ctx=ctx, kind="statistic"
        )

    def _load_cache_from_db(self) -> None:
        """Internal helper."""
        entries, index, indices = self._cache_db.load_entries(self._env_class_type_key)
        self._cache_entries = entries
        self._cache_faiss_index = index
        self._cache_faiss_entry_indices = indices
        env_preview = self._env_class_type_key[:80] + (
            "..." if len(self._env_class_type_key) > 80 else ""
        )
        get_logger().info(
            f"Loaded {len(entries)} cache entries (env={env_preview}), FAISS index size: {len(indices)}"
        )

    def get_cache_stats(self) -> CacheStats:
        """Internal helper."""

        return self._cache_stats

    def get_cache_stats_summary(self) -> str:
        """Internal helper."""
        stats = self._cache_stats
        return f"""Cache Statistics Summary:
- Total Requests: {stats.request_count}
- Predefined Hits (observe/statistic): {stats.predefined_hit_count}
- Cache Hits: {stats.cache_hit_count}
- Cache Misses: {stats.cache_miss_count}
- Cache Hit Rate: {stats.cache_hit_rate:.2%}
- Total Input Tokens: {stats.total_input_tokens}
- Total Output Tokens: {stats.total_output_tokens}
- Average Input Tokens: {stats.avg_input_tokens:.2f}
- Average Output Tokens: {stats.avg_output_tokens:.2f}
- Code Execution Success: {stats.code_execution_success_count}
- Code Execution Failure: {stats.code_execution_failure_count}
- Code Execution Success Rate: {stats.code_execution_success_rate:.2%}
- Total Code Retries: {stats.total_code_retry_count}
- Average Code Retries per Request: {stats.avg_retry_count:.2f}
- Cache Size: {len(self._cache_entries)} entries (env={self._env_class_type_key[:60] + ('...' if len(self._env_class_type_key) > 60 else '')})"""

    async def clear_cache(self) -> None:
        """Internal helper."""
        async with self._template_cache_lock:
            self._cache_db.clear_env_cache(self._env_class_type_key)
            self._cache_entries = []
            self._cache_faiss_index = None
            self._cache_faiss_entry_indices = []
            self._embedding_cache.clear()
            get_logger().info("Template cache cleared")


# Backward compatibility for older code and experiment scripts.
CodeGenRouter = EAPRouter
