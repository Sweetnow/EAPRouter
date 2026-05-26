"""
Commons Tragedy Game Environment
Environment for Tragedy of the Commons game based on V2 framework
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from eaprouter.env import EnvBase, tool


# Response models
class GetPoolResourcesResponse(BaseModel):
    """Response model for get_pool_resources() function"""

    current_pool_resources: int = Field(..., description="Current resource pool size")
    initial_pool_resources: int = Field(..., description="Initial resource pool size")


class SubmitExtractionResponse(BaseModel):
    """Response model for submit_extraction() function"""

    agent_name: str = Field(..., description="Agent name")
    requested_extraction: int = Field(..., description="Requested extraction amount")
    status: str = Field(..., description="Status: 'submitted' or 'round_executed'")


class GetRoundHistoryResponse(BaseModel):
    """Response model for get_round_history() function"""

    round: int = Field(..., description="Round number")
    pool_before_round: int = Field(..., description="Pool before round")
    extractions: Dict[str, int] = Field(..., description="Extractions by agent name")
    pool_after_round: int = Field(..., description="Pool after round")
    payoffs: Dict[str, int] = Field(..., description="Payoffs by agent name")


class CommonsTragedyEnv(EnvBase):
    """Environment for Tragedy of the Commons game based on V2 framework"""

    def __init__(
        self,
        num_agents: int = 4,
        initial_pool_resources: int = 100,
        max_extraction_per_agent: int = 10,
    ):
        """Initialize environment
        
        Args:
            num_agents: Number of agents (default: 4)
            initial_pool_resources: Initial resource pool size (default: 100)
            max_extraction_per_agent: Maximum extraction per agent per round (default: 10)
        """
        super().__init__()

        self.num_agents = num_agents
        self.initial_pool_resources = initial_pool_resources
        self.max_extraction_per_agent = max_extraction_per_agent

        self.current_pool_resources = self.initial_pool_resources
        self.round_number = 0
        self.round_history: List[dict] = []
        
        # Pending extractions for current round (agent_name -> extraction)
        self._pending_extractions: Dict[str, int] = {}
        
        # Track which agents have submitted in current round
        self._agents_submitted_in_current_round: set = set()
        
        # Track the last time we executed a round
        self._last_round_executed: int = -1
        
        self._lock = asyncio.Lock()

    @classmethod
    def mcp_description(cls) -> str:
        """Return a description text for MCP environment module candidate list"""
        description = f"""{cls.__name__}: Commons Tragedy game environment module.

**Description:** Manages a Tragedy of the Commons game where agents extract resources from a shared pool. The pool is depletable and if total extraction exceeds available resources, allocations are proportional.

**Initialization Parameters:**
- num_agents (int): Number of agents (default: 4)
- initial_pool_resources (int): Initial resource pool size (default: 100)
- max_extraction_per_agent (int): Maximum extraction per agent per round (default: 10)

**Example initialization config:**
```json
{{
  "num_agents": 4,
  "initial_pool_resources": 100,
  "max_extraction_per_agent": 10
}}
```
"""
        return description

    @property
    def description(self):
        """Description of the environment module"""
        return f"""You are a Commons Tragedy game environment module specialized in managing resource extraction from a shared pool.

**Game Overview:** Tragedy of the Commons - A strategic resource extraction game with {self.num_agents} agents over 10 rounds.

**Game Rules:**
- Initial pool: {self.initial_pool_resources} units
- Each round: You must extract between 1-{self.max_extraction_per_agent} units (inclusive)
- Your payoff per round = actual extraction amount (1 point per unit)
- Pool is depletable: If total extraction > remaining pool, allocations are proportional to requests
- Extractions execute simultaneously at round end
- You won't see outcomes until next round

**Available Operations (you MUST use these within your plan):**
1. **submit_extraction(agent_name, requested_extraction)**: Submit your extraction decision
   - agent_name: Your full name (e.g., "Agent-1")
   - requested_extraction: 1 to {self.max_extraction_per_agent} units
   - You MUST submit exactly once per round before the round executes

2. **get_pool_resources()**: Query current pool state
   - Returns: current pool size and initial pool size
   - Use this to inform your extraction decisions

3. **get_round_history(round_num=None)**: View past results
   - Returns: All rounds or specific round data
   - Shows: extractions, actual allocations, payoffs by agent

**CRITICAL CONSTRAINT:**
⚠️ You MUST complete your extraction decision (call submit_extraction) within your plan for this round.
The environment executes the round when all agents submit, so you have limited time.
If you don't submit, other agents' submissions will be delayed.
"""

    @tool(readonly=True, kind="observe")
    async def get_pool_resources(self) -> GetPoolResourcesResponse:
        """
        Get current pool resources.
        
        Game Context: This is a Tragedy of the Commons game. You are participating with other agents 
        in extracting resources from a shared pool over 10 rounds. Each unit you extract gives you 1 point. 
        The pool is depletable - if total extractions exceed available resources, allocations are proportional.

        Returns:
            Response containing current and initial pool resources.
        """
        async with self._lock:
            return GetPoolResourcesResponse(
                current_pool_resources=self.current_pool_resources,
                initial_pool_resources=self.initial_pool_resources,
            )

    @tool(readonly=False)
    async def submit_extraction(
        self, agent_name: str, requested_extraction: int
    ) -> SubmitExtractionResponse:
        """
        Submit extraction decision for an agent.
        
        Game Context: This is a Tragedy of the Commons game. You are participating with other agents 
        in extracting resources from a shared pool over 10 rounds. Each unit you extract gives you 1 point. 
        The pool is depletable - if total extractions exceed available resources, allocations are proportional.
        Your goal is to maximize your personal resource extraction over all rounds.

        Args:
            agent_name: The agent's name (should be in format "Agent-{id}", e.g., "Agent-1")
            requested_extraction: The requested extraction amount (1 to max_extraction_per_agent)

        Returns:
            Response containing submission status.
        """
        async with self._lock:
            # Validate extraction amount
            if (
                not isinstance(requested_extraction, int)
                or requested_extraction < 1
                or requested_extraction > self.max_extraction_per_agent
            ):
                requested_extraction = 1

            # Store pending extraction (allows re-submission in same round, last one wins)
            # Agent name should be in format "Agent-{id}" as specified in the agent profile
            self._pending_extractions[agent_name] = requested_extraction
            # Mark this agent as submitted in current round
            self._agents_submitted_in_current_round.add(agent_name)

            # Note: Round execution is deferred to step() to ensure atomicity
            # All agents submit first, then the round is executed together
            status = "submitted"

            return SubmitExtractionResponse(
                agent_name=agent_name,
                requested_extraction=requested_extraction,
                status=status,
            )

    def _calculate_actual_extractions_sync(
        self, requested_extractions: dict, pool_before: int
    ) -> tuple[dict, int, int]:
        """Calculate actual extractions considering resource pool capacity limits"""
        actual_extractions = {
            agent_name: 0 for agent_name in requested_extractions.keys()
        }
        total_requested = sum(requested_extractions.values())
        total_actual_extracted = 0

        if total_requested > 0 and pool_before > 0:
            if total_requested > pool_before:
                # Insufficient resources, allocate proportionally
                scaling_factor = pool_before / total_requested

                for agent_name, requested in requested_extractions.items():
                    actual_extractions[agent_name] = int(requested * scaling_factor)
                    total_actual_extracted += actual_extractions[agent_name]

                # Handle remainder
                remainder = pool_before - total_actual_extracted
                if remainder > 0:
                    fractional_parts = [
                        (
                            requested_extractions[name] * scaling_factor
                            - actual_extractions[name],
                            name,
                        )
                        for name in requested_extractions.keys()
                    ]
                    fractional_parts.sort(key=lambda x: x[0], reverse=True)

                    for _ in range(int(remainder)):
                        if fractional_parts:
                            _, agent_name_to_add = fractional_parts.pop(0)
                            actual_extractions[agent_name_to_add] += 1
                            total_actual_extracted += 1
                        else:
                            break
            else:
                # Sufficient resources, use requested extractions directly
                actual_extractions = requested_extractions.copy()
                total_actual_extracted = total_requested

        remaining_pool = pool_before - total_actual_extracted
        if remaining_pool < 0:
            remaining_pool = 0

        return actual_extractions, total_actual_extracted, remaining_pool

    @tool(readonly=True)
    async def get_round_history(self, round_num: Optional[int] = None) -> List[dict]:
        """
        Get round history.
        
        Game Context: This is a Tragedy of the Commons game. You are participating with other agents 
        in extracting resources from a shared pool over 10 rounds. Each unit you extract gives you 1 point. 
        The pool is depletable - if total extractions exceed available resources, allocations are proportional.
        Reviewing history helps you understand past behaviors and make better decisions.

        Args:
            round_num: Optional round number. If None, returns all rounds.

        Returns:
            List of round summaries.
        """
        async with self._lock:
            if round_num is not None:
                return [
                    r for r in self.round_history if r.get("round") == round_num
                ]
            return self.round_history.copy()

    async def init(self, start_datetime: datetime):
        """Initialize the environment"""
        await super().init(start_datetime)
        # Reset environment state for a new game
        self.current_pool_resources = self.initial_pool_resources
        self.round_number = 0
        self.round_history.clear()
        self._pending_extractions.clear()
        self._agents_submitted_in_current_round.clear()
        self._last_round_executed = -1

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step.
        
        This method is called by the environment router after agents have submitted their decisions.
        All submissions for the current round are processed and the round is executed here,
        ensuring atomicity and consistent state for the next round.
        
        Args:
            tick: The number of ticks of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        async with self._lock:
            self.t = t
            
            # Execute the round if at least some agents have submitted
            # (Agents that haven't submitted will be treated as extracting 0 units)
            # This prevents deadlock when some agents are slow or fail to decide
            if len(self._pending_extractions) > 0:
                # Execute the round
                self.round_number += 1
                pool_before_this_round = self.current_pool_resources

                # Calculate actual extractions with proportional allocation
                actual_extractions, total_extracted, remaining_pool = (
                    self._calculate_actual_extractions_sync(
                        self._pending_extractions.copy(), pool_before_this_round
                    )
                )

                self.current_pool_resources = remaining_pool

                # Build round summary
                round_summary = {
                    "round": self.round_number,
                    "pool_before_round": pool_before_this_round,
                    "extractions": actual_extractions,
                    "pool_after_round": remaining_pool,
                    "payoffs": actual_extractions.copy(),  # Extraction equals payoff
                }

                self.round_history.append(round_summary)

                # Clear pending extractions for next round
                self._pending_extractions.clear()
                self._agents_submitted_in_current_round.clear()

    def _dump_state(self) -> dict:
        """Serialize state"""
        return {
            "num_agents": self.num_agents,
            "initial_pool_resources": self.initial_pool_resources,
            "max_extraction_per_agent": self.max_extraction_per_agent,
            "current_pool_resources": self.current_pool_resources,
            "round_number": self.round_number,
            "round_history": self.round_history,
            "pending_extractions": self._pending_extractions,
            "agents_submitted_in_current_round": list(self._agents_submitted_in_current_round),
            "last_round_executed": self._last_round_executed,
        }

    def _load_state(self, state: dict):
        """Deserialize state"""
        self.num_agents = state.get("num_agents", 4)
        self.initial_pool_resources = state.get("initial_pool_resources", 100)
        self.max_extraction_per_agent = state.get("max_extraction_per_agent", 10)
        self.current_pool_resources = state.get("current_pool_resources", 0)
        self.round_number = state.get("round_number", 0)
        self.round_history = state.get("round_history", [])
        self._pending_extractions = state.get("pending_extractions", {})
        self._agents_submitted_in_current_round = set(state.get("agents_submitted_in_current_round", []))
        self._last_round_executed = state.get("last_round_executed", -1)


__all__ = ["CommonsTragedyEnv"]
