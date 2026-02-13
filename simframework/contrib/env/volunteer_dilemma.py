"""
Volunteer's Dilemma Game Environment
Environment for Volunteer's Dilemma game based on V2 framework
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from simframework.env import EnvBase, tool


# Response models
class SubmitChoiceResponse(BaseModel):
    """Response model for submit_choice() function"""

    agent_name: str = Field(..., description="Agent name")
    choice: str = Field(..., description="Choice (Volunteer/Stand by)")
    status: str = Field(..., description="Status: 'submitted' or 'round_executed'")


class VolunteerDilemmaEnv(EnvBase):
    """Environment for Volunteer's Dilemma game based on V2 framework"""

    def __init__(
        self,
        num_agents: int = 4,
        benefit_b: int = 100,
        cost_c: int = 40,
    ):
        """Initialize environment
        
        Args:
            num_agents: Number of agents (default: 4)
            benefit_b: Benefit for everyone if someone volunteers (default: 100)
            cost_c: Cost for a volunteer (default: 40)
        """
        super().__init__()

        self.num_agents = num_agents
        self.benefit_b = benefit_b
        self.cost_c = cost_c

        self.round_number = 0
        self.round_history: List[dict] = []
        
        # Pending choices for current round (agent_name -> choice)
        self._pending_choices: Dict[str, str] = {}
        
        self._lock = asyncio.Lock()

    @classmethod
    def mcp_description(cls) -> str:
        """Return a description text for MCP environment module candidate list"""
        description = f"""{cls.__name__}: Volunteer's Dilemma game environment module.

**Description:** Manages a Volunteer's Dilemma game where agents choose to volunteer or stand by. If at least one agent volunteers, all agents receive benefit, but volunteers pay a cost.

**Initialization Parameters:**
- num_agents (int): Number of agents (default: 4)
- benefit_b (int): Benefit for everyone if someone volunteers (default: 100)
- cost_c (int): Cost for a volunteer (default: 40)

**Example initialization config:**
```json
{{
  "num_agents": 4,
  "benefit_b": 100,
  "cost_c": 40
}}
```
"""
        return description

    @property
    def description(self):
        """Description of the environment module"""
        return f"""You are a Volunteer's Dilemma game environment module specialized in managing group volunteer decisions.

**Game Overview:** {self.num_agents} agents face a dilemma: someone must volunteer for a common cause, but volunteering has personal cost.

**Game Rules:**
- Each round: You choose "Volunteer" or "Stand by"
- Payoffs:
  * If at least 1 agent volunteers: Everyone gets benefit of {self.benefit_b}, volunteers pay cost of {self.cost_c}
    - Volunteer: {self.benefit_b} - {self.cost_c} = {self.benefit_b - self.cost_c} points
    - Stand by: {self.benefit_b} points (free rider benefit)
  * If NO one volunteers: Everyone gets 0 points (collective failure)
- Decisions are simultaneous
- Everyone benefits if someone volunteers, creating incentive to free-ride
- Goal: Maximize personal payoff while hoping others volunteer

**Available Operations (you MUST use these within your plan):**
1. **submit_choice(agent_name, choice)**: Make your decision
   - agent_name: Your full name (e.g., "Agent-1")
   - choice: "Volunteer" or "Stand by"
   - You MUST submit exactly once per round - this is your only action
   - No second chances - decision is final once submitted

2. **get_round_history(round_num=None)**: View past results
   - Returns: All rounds or specific round data
   - Shows: Who volunteered, total volunteers, everyone's payoffs
   - Use to evaluate group behavior and adjust future decisions

**CRITICAL CONSTRAINT:**
⚠️ You MUST submit your choice (call submit_choice with "Volunteer" or "Stand by") within your plan.
⚠️ This is your ONLY action per round - do it immediately, don't overcomplicate.
⚠️ The environment executes when all agents submit, so timely submission is crucial.
⚠️ Your choice determines the outcome: volunteer to ensure success, or hope others volunteer.
"""

    @tool(readonly=False)
    async def submit_choice(
        self, agent_name: str, choice: str
    ) -> SubmitChoiceResponse:
        """
        Submit choice decision for an agent.

        Args:
            agent_name: The agent's name
            choice: The choice ("Volunteer" or "Stand by")

        Returns:
            Response containing submission status.
        """
        async with self._lock:
            # Validate choice
            if choice in ["Volunteer", "Stand by"]:
                validated_choice = choice
            else:
                validated_choice = "Stand by"

            self._pending_choices[agent_name] = validated_choice

            # 记录提交日志用于调试
            import sys
            print(f"[ENV DEBUG] {agent_name} submitted: {validated_choice} (pending: {len(self._pending_choices)}/{self.num_agents})", file=sys.stderr)

            return SubmitChoiceResponse(
                agent_name=agent_name,
                choice=validated_choice,
                status="submitted",
            )

    @tool(readonly=True)
    async def get_round_history(self, round_num: Optional[int] = None) -> List[dict]:
        """
        Get round history.

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

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step.
        
        Executes a round if all agents have submitted their choices.
        
        Args:
            tick: The number of ticks of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        async with self._lock:
            self.t = t
            
            # 调试：打印待处理的选择
            import sys
            print(f"[ENV DEBUG STEP] Pending choices: {self._pending_choices}", file=sys.stderr)
            
            # Check if we have enough submissions to execute a round
            if len(self._pending_choices) >= self.num_agents:
                # Execute the round
                self.round_number += 1

                # Calculate number of volunteers
                num_volunteers = sum(
                    1 for choice in self._pending_choices.values()
                    if choice == "Volunteer"
                )
                is_someone_volunteering = num_volunteers > 0

                # Calculate payoffs for each agent
                payoffs = {}
                for agent_name, choice in self._pending_choices.items():
                    if is_someone_volunteering:
                        if choice == "Volunteer":
                            payoff = self.benefit_b - self.cost_c
                        else:  # "Stand by"
                            payoff = self.benefit_b
                    else:  # No one volunteered
                        payoff = 0
                    payoffs[agent_name] = float(payoff)

                # Build round summary with debug info
                round_summary = {
                    "round": self.round_number,
                    "choices": self._pending_choices.copy(),
                    "num_volunteers": num_volunteers,
                    "is_someone_volunteering": is_someone_volunteering,
                    "payoffs": payoffs,
                    "num_agents_submitted": len(self._pending_choices),
                    "benefit_b": self.benefit_b,
                    "cost_c": self.cost_c,
                }

                self.round_history.append(round_summary)

                # Clear pending choices for next round
                self._pending_choices.clear()

    def _dump_state(self) -> dict:
        """Serialize state"""
        return {
            "num_agents": self.num_agents,
            "benefit_b": self.benefit_b,
            "cost_c": self.cost_c,
            "round_number": self.round_number,
            "round_history": self.round_history,
            "pending_choices": self._pending_choices,
        }

    def _load_state(self, state: dict):
        """Deserialize state"""
        self.num_agents = state.get("num_agents", 4)
        self.benefit_b = state.get("benefit_b", 100)
        self.cost_c = state.get("cost_c", 40)
        self.round_number = state.get("round_number", 0)
        self.round_history = state.get("round_history", [])
        self._pending_choices = state.get("pending_choices", {})


__all__ = ["VolunteerDilemmaEnv"]
