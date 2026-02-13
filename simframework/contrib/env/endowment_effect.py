"""
Endowment Effect Experiment Environment
Environment for Endowment Effect experiment based on V2 framework
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from simframework.env import EnvBase, tool


# Response models for tool functions
class SubmitWTAWTPResponse(BaseModel):
    """Response model for submit_wta_wtp() function"""

    agent_id: int = Field(..., description="Agent ID")
    item: str = Field(..., description="Item name")
    wta: float = Field(..., description="Willingness to Accept price")
    wtp: float = Field(..., description="Willingness to Pay price")
    status: str = Field(..., description="Status: 'submitted' or 'completed'")


class GetMyEvaluationsResponse(BaseModel):
    """Response model for get_my_evaluations() function"""

    agent_id: int = Field(..., description="Agent ID")
    evaluations: Dict[str, Dict[str, float]] = Field(
        ..., description="Dictionary of item -> {'wta': float, 'wtp': float}"
    )
    completed_items: List[str] = Field(..., description="List of completed items")
    remaining_items: List[str] = Field(..., description="List of remaining items")


class EndowmentEffectEnv(EnvBase):
    """Environment for Endowment Effect experiment based on V2 framework"""

    # Valid items for the experiment
    VALID_ITEMS = ["pen", "plate", "glass", "doll"]

    def __init__(self, agent_ids: List[int]):
        """
        Initialize the Endowment Effect environment.

        Args:
            agent_ids: List of agent IDs participating in the experiment
        """
        super().__init__()

        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)

        # Store evaluations: {agent_id: {item: {"wta": float, "wtp": float}}}
        self._evaluations: Dict[int, Dict[str, Dict[str, float]]] = {
            agent_id: {} for agent_id in agent_ids
        }

        self._lock = asyncio.Lock()

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP environment module candidate list.
        Includes parameter descriptions and JSON schemas for data models.
        """
        description = f"""{cls.__name__}: Endowment Effect experiment environment module.

**Description:** Manages an Endowment Effect experiment where agents evaluate their willingness to accept (WTA) and willingness to pay (WTP) for different items.

**Initialization Parameters (excluding llm):**
- agent_ids (List[int]): List of agent IDs participating in the experiment

**Example initialization config:**
```json
{{
  "agent_ids": [101, 102, 103]
}}
```
"""
        return description

    @property
    def description(self):
        """Description of the environment module for router selection and function calling"""
        return f"""You are an Endowment Effect experiment environment module specialized in collecting WTA and WTP evaluations.

**Experiment Overview:** {self.num_agents} agents are participating in an Endowment Effect experiment to evaluate their willingness to accept (WTA) and willingness to pay (WTP) for different items.

**Experiment Task:**
You need to evaluate 4 items: pen, plate, glass, and doll. For each item, you need to provide two values:
- **WTA (Willingness to Accept)**: The minimum price you would be willing to accept to sell this item (if you owned it)
- **WTP (Willingness to Pay)**: The maximum price you would be willing to pay to buy this item

**Evaluation Instructions:**
- Both WTA and WTP should be non-negative numbers (can be 0 or positive)
- You should base your evaluations on your personal profile and psychological characteristics
- Consider the actual value and utility of each item when making your evaluations
- Think about the endowment effect: people often value items they own more than items they don't own

**Available Operations (you MUST use these within your plan):**
1. **submit_wta_wtp(agent_id, item, wta, wtp)**: Submit your WTA and WTP evaluation for an item
   - agent_id: Your agent ID (e.g., 101, 102)
   - item: Item name - must be one of: "pen", "plate", "glass", "doll"
   - wta: Your willingness to accept price (non-negative number)
   - wtp: Your willingness to pay price (non-negative number)
   - You can submit evaluations for multiple items, one at a time
   - You can update an evaluation by submitting again for the same item

2. **get_my_evaluations(agent_id)**: View your submitted evaluations
   - agent_id: Your agent ID
   - Returns: Your current evaluations, completed items, and remaining items

3. **get_all_evaluations()**: View all agents' evaluations (statistics)
   - Returns: Dictionary of all agents' evaluations
   - Use this to see how others are evaluating items

**CRITICAL CONSTRAINT:**
⚠️ You MUST submit evaluations for all 4 items (pen, plate, glass, doll) within your plan.
⚠️ Each evaluation requires both WTA and WTP values.
⚠️ Consider your psychological profile when making these evaluations - your personality traits, values, and preferences should influence your pricing decisions.
"""

    @tool(readonly=False)
    async def submit_wta_wtp(
        self, agent_id: int, item: str, wta: float, wtp: float
    ) -> SubmitWTAWTPResponse:
        """
        Submit WTA and WTP evaluation for an item.

        Args:
            agent_id: The agent's ID
            item: The item name (must be one of: "pen", "plate", "glass", "doll")
            wta: Willingness to Accept price (non-negative number)
            wtp: Willingness to Pay price (non-negative number)

        Returns:
            Response containing submission status.
        """
        async with self._lock:
            # Validate agent_id
            if agent_id not in self.agent_ids:
                raise ValueError(
                    f"Agent ID {agent_id} is not in the experiment. Valid IDs: {self.agent_ids}"
                )

            # Validate and normalize item name
            item = item.lower().strip()
            if item not in self.VALID_ITEMS:
                raise ValueError(
                    f"Invalid item '{item}'. Valid items are: {self.VALID_ITEMS}"
                )

            # Validate WTA and WTP (must be non-negative)
            if wta < 0:
                wta = 0.0
            if wtp < 0:
                wtp = 0.0

            # Store the evaluation
            if agent_id not in self._evaluations:
                self._evaluations[agent_id] = {}

            self._evaluations[agent_id][item] = {
                "wta": float(wta),
                "wtp": float(wtp),
            }
            
            # Check if all items are completed
            completed = len(self._evaluations[agent_id])
            all_completed = completed == len(self.VALID_ITEMS)
            
            status = "completed" if all_completed else "submitted"

            return SubmitWTAWTPResponse(
                agent_id=agent_id,
                item=item,
                wta=float(wta),
                wtp=float(wtp),
                status=status,
            )

    @tool(readonly=True, kind="observe")
    async def get_my_evaluations(self, agent_id: int) -> GetMyEvaluationsResponse:
        """
        Get evaluations submitted by a specific agent.

        Args:
            agent_id: The agent's ID

        Returns:
            Response containing current evaluations, completed items, and remaining items.
        """
        async with self._lock:
            if agent_id not in self.agent_ids:
                raise ValueError(
                    f"Agent ID {agent_id} is not in the experiment. Valid IDs: {self.agent_ids}"
                )

            evaluations = self._evaluations.get(agent_id, {}).copy()
            completed_items = list(evaluations.keys())
            remaining_items = [
                item for item in self.VALID_ITEMS if item not in completed_items
            ]
            
            return GetMyEvaluationsResponse(
                agent_id=agent_id,
                evaluations=evaluations,
                completed_items=completed_items,
                remaining_items=remaining_items,
            )

    @tool(readonly=True, kind="statistics")
    async def get_all_evaluations(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Get all agents' evaluations.

        Returns:
            Dictionary of all evaluations: {agent_id: {item: {"wta": float, "wtp": float}}}
        """
        async with self._lock:
            # Return a deep copy to prevent external modifications
            return {
                agent_id: {item: eval_data.copy() for item, eval_data in items.items()}
                for agent_id, items in self._evaluations.items()
            }

    async def init(self, start_datetime: datetime):
        """
        Initialize the environment module.
        """
        await super().init(start_datetime)

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step.

        Args:
            tick: The number of ticks (1 tick = 1 second) of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        async with self._lock:
            self.current_datetime = t

    def get_results(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Get all evaluation results (synchronous method for result extraction).

        Returns:
            Dictionary of all evaluations: {agent_id: {item: {"wta": float, "wtp": float}}}
        """
        return {
            agent_id: {item: eval_data.copy() for item, eval_data in items.items()}
            for agent_id, items in self._evaluations.items()
        }

    def _dump_state(self) -> dict:
        """Serialize state"""
        return {
            "agent_ids": self.agent_ids,
            "num_agents": self.num_agents,
            "evaluations": self._evaluations,
        }

    def _load_state(self, state: dict):
        """Deserialize state"""
        self.agent_ids = state.get("agent_ids", [])
        self.num_agents = state.get("num_agents", len(self.agent_ids))
        self._evaluations = state.get("evaluations", {})


__all__ = ["EndowmentEffectEnv"]

