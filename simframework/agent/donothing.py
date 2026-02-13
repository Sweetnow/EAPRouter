from datetime import datetime
from typing import Any

from simframework.agent.base import AgentBase
from simframework.config import Config
from mem0 import AsyncMemory
from simframework.env import RouterBase

__all__ = ["DoNothingAgent"]


class DoNothingAgent(AgentBase):
    """
    A minimal agent with simple memory (mem0). On each step, it asks the
    environment two questions in English and stores the answers:
    - "What time is it now?"
    - "How is the environment now?"
    """

    def __init__(self, id: int, profile: Any | dict):
        """
        Initialize the DoNothingAgent.

        Args:
            id: The unique identifier for the agent.
            profile: The profile of the agent. Can be a dict or any other type.
        """
        super().__init__(id=id, profile=profile)
        self._memory_user_id = f"agent-{id}"
        self._memory: AsyncMemory | None = None

        # Initialize memory from config if provided
        self._memory = AsyncMemory(config=Config.get_mem0_config(str(id)))

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP agent module candidate list.
        Includes parameter descriptions.
        """
        description = f"""{cls.__name__}: A minimal agent with simple memory (mem0).

**Description:** {cls.__doc__}

**Behavior:** On each step, it asks the environment two questions in English and stores the answers:
- "What time is it now?"
- "How is the environment now?"

**Initialization Parameters:**
- id (int): The unique identifier for the agent.
- profile (dict | Any): The profile of the agent. Can be a dictionary with agent attributes or any other type. Common profile fields include: name, gender, age, education, occupation, marriage_status, persona, background_story.

**Example initialization config:**
```json
{{
  "id": 1,
  "profile": {{
    "name": "Agent 1",
    "gender": "unknown",
    "age": 25,
    "education": "Unknown",
    "occupation": "Unknown",
    "marriage_status": "Unknown"
  }}
}}
```

**Note:** If memory_config is null or not provided, the agent will still function but memory-related operations (search, add) will be disabled. The memory_config will be automatically converted to an AsyncMemory instance in the agent's __init__ method.
"""
        return description

    async def init(self, env: RouterBase):
        await super().init(env)

    async def dump(self) -> dict:
        # collect all memories via mem0
        memories: list[Any] = []
        try:
            if self._memory is not None:
                res: Any = await self._memory.get_all(
                    user_id=self._memory_user_id,
                )
                items: list[Any] = []
                if isinstance(res, dict):
                    for key in ("memories", "data", "results", "items"):
                        v = res.get(key)
                        if isinstance(v, list):
                            items = v
                            break
                elif isinstance(res, list):
                    items = res
                memories.extend(items)
        except Exception as e:
            # ignore mem0 dump errors, keep minimal dump
            print(f"Warning: Failed to dump memories for agent {self._id}: {e}")

        return {
            "profile": self._profile,
            "memories": memories,
        }

    async def load(self, dump_data: dict):
        try:
            self._profile = dump_data.get("profile")
        except Exception as e:
            print(f"Warning: Failed to load profile for agent {self._id}: {e}")

        # Restore memories to mem0
        try:
            memories = dump_data.get("memories", [])
            if memories and self._memory is not None:
                for mem_item in memories:
                    # Extract the memory text from various possible formats
                    memory_text = None
                    if isinstance(mem_item, dict):
                        memory_text = mem_item.get("memory") or mem_item.get("text")
                    elif isinstance(mem_item, str):
                        memory_text = mem_item

                    if memory_text:
                        await self._memory.add(
                            memory_text, user_id=self._memory_user_id
                        )
        except Exception as e:
            print(f"Warning: Failed to load memories for agent {self._id}: {e}")

    async def ask(self, message: str, readonly: bool = True) -> str:
        assert self._memory is not None, "Memory is not initialized"
        results = await self._memory.search(
            message, user_id=self._memory_user_id, limit=1
        )
        memory_text = ""
        for result in results["results"]:
            memory_text += f"- {result['memory']}\n"

        if memory_text:
            prompt = (
                f"You are a DoNothingAgent. Here is some related memory: {memory_text}\n"
                f"Answer the question: {message}"
            )
        else:
            prompt = f"You are a DoNothingAgent. Answer the question: {message}"

        response = await self.acompletion(
            [
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return str(response.choices[0].message.content)  # type: ignore

    async def step(self, tick: int, t: datetime) -> str:
        q1 = "What time is it now?"
        q2 = "How is the environment now?"
        ctx = {"id": self._id}
        result, ans1 = await self.ask_env(ctx, q1, readonly=True)
        result, ans2 = await self.ask_env(result, q2, readonly=True)

        ts = t.isoformat()
        assert self._memory is not None, "Memory is not initialized"
        await self._memory.add(f"[{ts}] {q1} -> {ans1}", user_id=self._memory_user_id)
        await self._memory.add(f"[{ts}] {q2} -> {ans2}", user_id=self._memory_user_id)

        return f"time: {ans1}; env: {ans2}"

    async def close(self):
        return None
