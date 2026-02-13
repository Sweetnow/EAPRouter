"""
Self-Enhancement Experiment Environment
Environment for Self-Enhancement (SE) experiment based on V2 framework
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from simframework.env import EnvBase, tool


# Response models for tool functions
class SubmitRankingResponse(BaseModel):
    """Response model for submit_ranking() function"""

    agent_id: int = Field(..., description="Agent ID")
    dimension: str = Field(..., description="Dimension name")
    percentile: int = Field(..., description="Percentile ranking (0-100)")
    status: str = Field(..., description="Status: 'submitted' or 'completed'")


class GetRankingsResponse(BaseModel):
    """Response model for get_my_rankings() function"""

    agent_id: int = Field(..., description="Agent ID")
    rankings: Dict[str, int] = Field(..., description="Dictionary of dimension -> percentile")
    completed_dimensions: List[str] = Field(..., description="List of completed dimensions")
    remaining_dimensions: List[str] = Field(..., description="List of remaining dimensions")


# Valid dimensions for SE experiment
VALID_DIMENSIONS = [
    "INTELLIGENCE",
    "COOPERATION",
    "APPEARANCE",
    "MORALITY",
    "SOCIABILITY",
    "HEALTH",
    "HONESTY",
    "GENEROSITY",
]


class SelfEnhancementEnv(EnvBase):
    """Environment for Self-Enhancement (SE) experiment based on V2 framework"""

    def __init__(
        self,
        agent_ids: List[int],
    ):
        """
        Initialize the Self-Enhancement environment.

        Args:
            agent_ids: List of agent IDs participating in the experiment
        """
        super().__init__()

        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        
        # Store rankings: agent_id -> dimension -> percentile
        self._rankings: Dict[int, Dict[str, int]] = {
            agent_id: {} for agent_id in agent_ids
        }
        
        self._lock = asyncio.Lock()

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP environment module candidate list.
        Includes parameter descriptions and JSON schemas for data models.
        """
        description = f"""{cls.__name__}: Self-Enhancement experiment environment module.

**Description:** Manages a Self-Enhancement experiment where agents evaluate their percentile rankings (0-100) across 8 dimensions relative to Tsinghua University student population.

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
        return f"""You are a Self-Enhancement experiment environment module specialized in managing self-evaluation assessments.

**Experiment Overview:** {self.num_agents} agents are participating in a self-enhancement evaluation study.

**Experiment Task:**
You need to evaluate your percentile ranking (0-100) in 8 dimensions relative to the Tsinghua University student population. A percentile of 50 means you are at the median (average), 75 means you are better than 75% of students, 25 means you are better than 25% of students.

**8 Evaluation Dimensions:**
1. **INTELLIGENCE** - Your intellectual ability and cognitive capacity
2. **COOPERATION** - Your ability to work collaboratively with others
3. **APPEARANCE** - Your physical appearance and attractiveness
4. **MORALITY** - Your moral character and ethical behavior
5. **SOCIABILITY** - Your social skills and ability to interact with others
6. **HEALTH** - Your physical health and well-being
7. **HONESTY** - Your honesty and truthfulness
8. **GENEROSITY** - Your generosity and willingness to help others

**Evaluation Instructions:**
- Consider your psychological profile and characteristics when making evaluations
- Be honest but also consider your self-perception
- Each dimension should be evaluated independently
- Percentile rankings range from 0 to 100
- You must complete all 8 dimensions

**Available Operations (you MUST use these within your plan):**
1. **submit_ranking(agent_id, dimension, percentile)**: Submit your percentile ranking for a dimension
   - agent_id: Your agent ID (e.g., 101)
   - dimension: One of the 8 dimensions (INTELLIGENCE, COOPERATION, APPEARANCE, MORALITY, SOCIABILITY, HEALTH, HONESTY, GENEROSITY)
   - percentile: Your percentile ranking (0-100, integer)
   - You can submit rankings for multiple dimensions, but each dimension can only be submitted once
   - You MUST complete all 8 dimensions

2. **get_my_rankings(agent_id)**: View your submitted rankings
   - agent_id: Your agent ID
   - Returns: Your current rankings, completed dimensions, and remaining dimensions
   - Use this to track your progress and see which dimensions you still need to evaluate

**CRITICAL CONSTRAINT:**
⚠️ You MUST submit rankings for all 8 dimensions.
⚠️ Each dimension can only be submitted once - you cannot change your ranking after submission.
⚠️ Percentile must be an integer between 0 and 100.
⚠️ Consider your psychological profile when making evaluations - your self-perception is important.
"""

    @tool(readonly=False)
    async def submit_ranking(
        self, agent_id: int, dimension: str, percentile: int
    ) -> SubmitRankingResponse:
        """
        Submit percentile ranking for a dimension.

        Args:
            agent_id: The agent's ID
            dimension: The dimension name (must be one of: INTELLIGENCE, COOPERATION, APPEARANCE, MORALITY, SOCIABILITY, HEALTH, HONESTY, GENEROSITY)
            percentile: The percentile ranking (0-100, integer)

        Returns:
            Response containing submission status.
        """
        async with self._lock:
            # Validate agent_id
            if agent_id not in self.agent_ids:
                raise ValueError(f"Agent ID {agent_id} is not in the experiment")
            
            # Validate dimension
            dimension_upper = dimension.upper()
            if dimension_upper not in VALID_DIMENSIONS:
                raise ValueError(
                    f"Invalid dimension: {dimension}. Must be one of: {', '.join(VALID_DIMENSIONS)}"
                )
            
            # Validate percentile
            if not isinstance(percentile, int):
                percentile = int(round(percentile))
            percentile = max(0, min(100, percentile))  # Clamp to 0-100
            
            # Check if already submitted
            if dimension_upper in self._rankings[agent_id]:
                raise ValueError(
                    f"Ranking for dimension {dimension_upper} has already been submitted. "
                    f"Current ranking: {self._rankings[agent_id][dimension_upper]}"
                )
            
            # Store the ranking
            self._rankings[agent_id][dimension_upper] = percentile
            
            # Check if all dimensions are completed
            completed = len(self._rankings[agent_id])
            all_completed = completed == len(VALID_DIMENSIONS)
            
            status = "completed" if all_completed else "submitted"
            
            # Debug log
            import sys
            print(
                f"[ENV DEBUG] Agent {agent_id} submitted {dimension_upper}: {percentile} "
                f"({completed}/{len(VALID_DIMENSIONS)} completed)",
                file=sys.stderr
            )
            
            return SubmitRankingResponse(
                agent_id=agent_id,
                dimension=dimension_upper,
                percentile=percentile,
                status=status,
            )

    @tool(readonly=True)
    async def get_my_rankings(self, agent_id: int) -> GetRankingsResponse:
        """
        Get rankings for a specific agent.

        Args:
            agent_id: The agent's ID

        Returns:
            Response containing current rankings, completed dimensions, and remaining dimensions.
        """
        async with self._lock:
            if agent_id not in self.agent_ids:
                raise ValueError(f"Agent ID {agent_id} is not in the experiment")
            
            rankings = self._rankings[agent_id].copy()
            completed_dimensions = list(rankings.keys())
            remaining_dimensions = [
                dim for dim in VALID_DIMENSIONS if dim not in completed_dimensions
            ]
            
            return GetRankingsResponse(
                agent_id=agent_id,
                rankings=rankings,
                completed_dimensions=completed_dimensions,
                remaining_dimensions=remaining_dimensions,
            )

    @tool(readonly=True, kind="statistics")
    async def get_all_rankings(self) -> Dict[int, Dict[str, int]]:
        """
        Get all rankings for all agents (statistics function).

        Returns:
            Dictionary mapping agent_id to their rankings dictionary.
        """
        async with self._lock:
            return {
                agent_id: rankings.copy()
                for agent_id, rankings in self._rankings.items()
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

    def get_results(self) -> Dict[int, Dict[str, int]]:
        """
        Get all rankings results (synchronous method for result extraction).

        Returns:
            Dictionary mapping agent_id to their rankings dictionary.
        """
        return {
            agent_id: rankings.copy()
            for agent_id, rankings in self._rankings.items()
        }

    def _dump_state(self) -> dict:
        """Serialize state"""
        return {
            "agent_ids": self.agent_ids,
            "num_agents": self.num_agents,
            "rankings": self._rankings,
        }

    def _load_state(self, state: dict):
        """Deserialize state"""
        self.agent_ids = state.get("agent_ids", [])
        self.num_agents = state.get("num_agents", 0)
        self._rankings = state.get("rankings", {})


__all__ = ["SelfEnhancementEnv", "VALID_DIMENSIONS"]

