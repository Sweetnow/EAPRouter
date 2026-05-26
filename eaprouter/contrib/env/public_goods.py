"""
Public Goods Game Environment
Environment for Public Goods Game based on V2 framework
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from eaprouter.env import EnvBase, tool


# Response models
class SubmitContributionResponse(BaseModel):
    """Response model for submit_contribution() function"""

    agent_name: str = Field(..., description="Agent name")
    contribution: int = Field(..., description="Contribution amount")
    status: str = Field(..., description="Status: 'submitted' or 'round_executed'")


class GetRoundResultResponse(BaseModel):
    """Response model for get_round_result() function"""

    round: int = Field(..., description="Round number")
    total_contribution: int = Field(..., description="Total contribution")
    public_pool_gain: float = Field(..., description="Public pool gain")
    gain_per_agent: float = Field(..., description="Gain per agent")
    contributions: Dict[str, int] = Field(..., description="Contributions by agent name")
    payoffs: Dict[str, float] = Field(..., description="Payoffs by agent name")


class PublicGoodsEnv(EnvBase):
    """Environment for Public Goods Game based on V2 framework"""

    def __init__(
        self,
        num_agents: int = 4,
        initial_endowment: int = 20,
        public_pool_multiplier: float = 1.6,
    ):
        """Initialize environment
        
        Args:
            num_agents: Number of agents (default: 4)
            initial_endowment: Initial coins per agent per round (default: 20)
            public_pool_multiplier: Multiplier for public pool contributions (default: 1.6)
        """
        super().__init__()

        self.num_agents = num_agents
        self.initial_endowment = initial_endowment
        self.public_pool_multiplier = public_pool_multiplier

        self.round_number = 0
        self.round_history: List[dict] = []
        
        # Pending contributions for current round (agent_name -> contribution)
        self._pending_contributions: Dict[str, int] = {}
        
        # Track which agents have submitted in current round
        self._agents_submitted_in_current_round: set = set()
        
        self._lock = asyncio.Lock()

    @classmethod
    def mcp_description(cls) -> str:
        """Return a description text for MCP environment module candidate list"""
        description = f"""{cls.__name__}: Public Goods Game environment module.

**Description:** Manages a Public Goods Game where agents contribute to a public fund. Contributions are multiplied and divided equally among all players.

**Initialization Parameters:**
- num_agents (int): Number of agents (default: 4)
- initial_endowment (int): Initial coins per agent per round (default: 20)
- public_pool_multiplier (float): Multiplier for public pool contributions (default: 1.6)

**Example initialization config:**
```json
{{
  "num_agents": 4,
  "initial_endowment": 20,
  "public_pool_multiplier": 1.6
}}
```
"""
        return description

    @property
    def description(self):
        """Description of the environment module"""
        return f"""You are a Public Goods Game environment module specialized in managing contributions and collective payoffs.

**Game Overview:** {self.num_agents} agents cooperatively contribute to a shared public fund that benefits everyone.

**Game Rules:**
- Each round: You receive {self.initial_endowment} coins
- You choose: Contribute 0 to {self.initial_endowment} coins to public fund
- Public fund calculation: Total contributions × {self.public_pool_multiplier} multiplier
- Your payoff = (coins kept) + (share of multiplied fund / {self.num_agents})
- Example: If you keep 10 and total fund is 40: payoff = 10 + (40×{self.public_pool_multiplier}/{self.num_agents})
- All agents' past contributions and payoffs are visible
- Goal: Maximize cumulative coins over 10 rounds

**Available Operations (you MUST use these within your plan):**
1. **submit_contribution(agent_name, contribution)**: Submit your contribution
   - agent_name: Your full name (e.g., "Agent-1")
   - contribution: 0 to {self.initial_endowment} coins
   - You MUST submit exactly once per round before round executes
   - Contributing more benefits everyone, but reduces your private coins

2. **get_round_history(round_num=None)**: View past results
   - Returns: All rounds or specific round data
   - Shows: Each agent's contribution, total fund, gain per agent, individual payoffs
   - Use to analyze others' strategies and plan accordingly

**CRITICAL CONSTRAINT:**
⚠️ You MUST submit your contribution (call submit_contribution) within your plan for this round.
The environment executes the round when all agents submit.
If you don't submit, the round execution is delayed.
Submissions are final - cannot change contribution after submitting.
"""

    @tool(readonly=False)
    async def submit_contribution(
        self, agent_name: str, contribution: int
    ) -> SubmitContributionResponse:
        """
        Submit contribution decision for an agent in the Public Goods Game.
        
        Game Context: This is a Public Goods Game where agents contribute to a public fund.
        Each round, agents receive coins and can contribute 0 to their endowment to the public fund.
        The total public fund is multiplied and divided equally among all players.
        Each agent's round gain = (coins not contributed) + (share of multiplied public fund).

        Args:
            agent_name: The agent's name (should be in format "Agent-{id}", e.g., "Agent-1")
            contribution: The contribution amount (0 to initial_endowment)

        Returns:
            Response containing submission status.
        """
        async with self._lock:
            # Validate contribution
            if (
                not isinstance(contribution, int)
                or contribution < 0
                or contribution > self.initial_endowment
            ):
                contribution = 0

            # Store pending contribution (allows re-submission in same round, last one wins)
            self._pending_contributions[agent_name] = contribution
            # Mark this agent as submitted in current round
            self._agents_submitted_in_current_round.add(agent_name)

            # Note: Round execution is deferred to step() to ensure atomicity
            status = "submitted"

            return SubmitContributionResponse(
                agent_name=agent_name,
                contribution=contribution,
                status=status,
            )

    @tool(readonly=True)
    async def get_round_history(self, round_num: Optional[int] = None) -> List[dict]:
        """
        Get round history for the Public Goods Game.
        
        Game Context: This is a Public Goods Game where agents contribute to a public fund.
        Each round summary contains: round number, total contribution, public pool gain, and payoffs for each agent.
        History helps agents understand past contributions and outcomes to make better decisions.

        Args:
            round_num: Optional round number. If None, returns all rounds.

        Returns:
            List of round summaries. Each summary contains:
            - round: Round number
            - total_contribution: Total coins contributed by all agents
            - public_pool_gain: Total public fund after multiplication
            - payoffs: Dictionary mapping agent names to their round gains
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
        self.round_number = 0
        self.round_history.clear()
        self._pending_contributions.clear()
        self._agents_submitted_in_current_round.clear()

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
            
            # Execute the round if all agents have submitted
            if len(self._agents_submitted_in_current_round) >= self.num_agents:
                # Execute the round
                self.round_number += 1

                # Calculate total contribution and gains
                total_contribution = sum(self._pending_contributions.values())
                public_pool_gain = total_contribution * self.public_pool_multiplier
                gain_per_agent = public_pool_gain / self.num_agents

                # Calculate payoffs for each agent
                payoffs = {}
                for agent_name, contribution in self._pending_contributions.items():
                    private_savings = self.initial_endowment - contribution
                    total_gain = private_savings + gain_per_agent
                    payoffs[agent_name] = total_gain

                # Build round summary
                round_summary = {
                    "round": self.round_number,
                    "total_contribution": total_contribution,
                    "public_pool_gain": public_pool_gain,
                    "contributions": self._pending_contributions.copy(),  # Store individual contributions
                    "payoffs": payoffs,
                }

                self.round_history.append(round_summary)

                # Clear pending contributions for next round
                self._pending_contributions.clear()
                self._agents_submitted_in_current_round.clear()

    def _dump_state(self) -> dict:
        """Serialize state"""
        return {
            "num_agents": self.num_agents,
            "initial_endowment": self.initial_endowment,
            "public_pool_multiplier": self.public_pool_multiplier,
            "round_number": self.round_number,
            "round_history": self.round_history,
            "pending_contributions": self._pending_contributions,
            "agents_submitted_in_current_round": list(self._agents_submitted_in_current_round),
        }

    def _load_state(self, state: dict):
        """Deserialize state"""
        self.num_agents = state.get("num_agents", 4)
        self.initial_endowment = state.get("initial_endowment", 20)
        self.public_pool_multiplier = state.get("public_pool_multiplier", 1.6)
        self.round_number = state.get("round_number", 0)
        self.round_history = state.get("round_history", [])
        self._pending_contributions = state.get("pending_contributions", {})
        self._agents_submitted_in_current_round = set(state.get("agents_submitted_in_current_round", []))


__all__ = ["PublicGoodsEnv"]
