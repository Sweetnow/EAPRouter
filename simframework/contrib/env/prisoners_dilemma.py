"""
Prisoner's Dilemma Game Environment
Environment for Prisoner's Dilemma game based on V2 framework
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from simframework.env import EnvBase, tool


# Response models
class SubmitActionResponse(BaseModel):
    """Response model for submit_action() function"""

    agent_name: str = Field(..., description="Agent name")
    action: str = Field(..., description="Action (Yes/No)")
    status: str = Field(..., description="Status: 'submitted' or 'round_executed'")


class GetPayoffMatrixResponse(BaseModel):
    """Response model for get_payoff_matrix() function"""

    payoff_cc: int = Field(..., description="Payoff when both cooperate")
    payoff_cd: int = Field(..., description="Payoff when cooperate but opponent defects")
    payoff_dc: int = Field(..., description="Payoff when defect but opponent cooperates")
    payoff_dd: int = Field(..., description="Payoff when both defect")


class PrisonersDilemmaEnv(EnvBase):
    """Environment for Prisoner's Dilemma game based on V2 framework"""

    def __init__(
        self,
        payoff_cc: int = 3,
        payoff_cd: int = 0,
        payoff_dc: int = 5,
        payoff_dd: int = 1,
    ):
        """Initialize environment
        
        Args:
            payoff_cc: Payoff when both cooperate (default: 3)
            payoff_cd: Payoff when cooperate but opponent defects (default: 0)
            payoff_dc: Payoff when defect but opponent cooperates (default: 5)
            payoff_dd: Payoff when both defect (default: 1)
        """
        super().__init__()

        self.payoff_cc = payoff_cc
        self.payoff_cd = payoff_cd
        self.payoff_dc = payoff_dc
        self.payoff_dd = payoff_dd

        self.round_number = 0
        self.round_history: List[dict] = []
        
        # Pending actions for current round (agent_name -> action)
        self._pending_actions: Dict[str, str] = {}
        
        self._lock = asyncio.Lock()

    @classmethod
    def mcp_description(cls) -> str:
        """Return a description text for MCP environment module candidate list"""
        description = f"""{cls.__name__}: Prisoner's Dilemma game environment module.

**Description:** Manages a Prisoner's Dilemma game where two agents simultaneously choose to cooperate (Yes) or defect (No), with payoffs determined by the payoff matrix.

**Initialization Parameters:**
- payoff_cc (int): Payoff when both cooperate (default: 3)
- payoff_cd (int): Payoff when cooperate but opponent defects (default: 0)
- payoff_dc (int): Payoff when defect but opponent cooperates (default: 5)
- payoff_dd (int): Payoff when both defect (default: 1)

**Example initialization config:**
```json
{{
  "payoff_cc": 3,
  "payoff_cd": 0,
  "payoff_dc": 5,
  "payoff_dd": 1
}}
```
"""
        return description

    @property
    def description(self):
        """Description of the environment module"""
        return f"""You are a Prisoner's Dilemma game environment module specialized in managing simultaneous decisions.

**Game Overview:** Two agents make simultaneous decisions to cooperate or defect, with payoff outcomes based on mutual decisions.

**Game Rules:**
- Two players: Agent-A and Agent-B
- Each round: Choose "Yes" (cooperate) or "No" (defect)
- Payoffs determined by payoff matrix:
  * Both cooperate (Yes, Yes): {self.payoff_cc} points each
  * You cooperate, opponent defects (Yes, No): {self.payoff_cd} for you, {self.payoff_dc} for opponent
  * You defect, opponent cooperates (No, Yes): {self.payoff_dc} for you, {self.payoff_cd} for opponent
  * Both defect (No, No): {self.payoff_dd} points each
- Decisions are simultaneous and executed together
- You won't see opponent's choice until round ends

**Available Operations (you MUST use these within your plan):**
1. **submit_action(agent_name, action)**: Submit your decision
   - agent_name: Your full name ("Agent-A" or "Agent-B")
   - action: "Yes" (cooperate) or "No" (defect)
   - You MUST submit exactly once per round

2. **get_payoff_matrix()**: View the payoff structure
   - Returns: All payoff values for different outcome combinations
   - Use this to make informed strategic decisions

3. **get_round_history(round_num=None)**: View past results
   - Returns: All rounds or specific round data
   - Shows: Both players' actions and payoffs per round

**CRITICAL CONSTRAINT:**
⚠️ You MUST submit your action decision (call submit_action) within your plan for this round.
The environment executes the round when both agents submit.
If you don't submit, the round execution is delayed.
"""

    def _calculate_payoff(self, action1: str, action2: str) -> Tuple[int, int]:
        """Calculate payoffs based on actions"""
        action1 = action1.capitalize()
        action2 = action2.capitalize()

        if action1 == "Yes" and action2 == "Yes":
            return self.payoff_cc, self.payoff_cc
        elif action1 == "Yes" and action2 == "No":
            return self.payoff_cd, self.payoff_dc
        elif action1 == "No" and action2 == "Yes":
            return self.payoff_dc, self.payoff_cd
        else:  # Both No
            return self.payoff_dd, self.payoff_dd

    @tool(readonly=False)
    async def submit_action(self, agent_name: str, action: str) -> SubmitActionResponse:
        """
        Submit action decision for an agent.

        Args:
            agent_name: The agent's name ("Agent A" or "Agent B")
            action: The action ("Yes" or "No")

        Returns:
            Response containing submission status.
        """
        async with self._lock:
            # Normalize and validate action
            action = action.capitalize()
            if action not in ["Yes", "No"]:
                action = "No"

            self._pending_actions[agent_name] = action

            return SubmitActionResponse(
                agent_name=agent_name,
                action=action,
                status="submitted",
            )

    @tool(readonly=True, kind="statistics")
    async def get_payoff_matrix(self) -> GetPayoffMatrixResponse:
        """
        Get the payoff matrix.

        Returns:
            Response containing all payoff values.
        """
        async with self._lock:
            return GetPayoffMatrixResponse(
                payoff_cc=self.payoff_cc,
                payoff_cd=self.payoff_cd,
                payoff_dc=self.payoff_dc,
                payoff_dd=self.payoff_dd,
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
        
        Executes a round if at least one agent has submitted their action.
        If only one agent submitted, the other agent defaults to 'No' (defect).
        
        Args:
            tick: The number of ticks of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        async with self._lock:
            self.t = t
            
            # Check if we have at least one agent's action to execute a round
            # (Other agents default to 'No' if they haven't submitted)
            if len(self._pending_actions) >= 1:
                # Execute the round
                self.round_number += 1

                # Get actions (assuming Agent A and Agent B)
                agent_names = list(self._pending_actions.keys())
                
                # Handle cases where one or both agents have submitted
                if len(agent_names) >= 1:
                    # Get first agent's action (or use default)
                    agent_a_name = agent_names[0]
                    agent_a_action = self._pending_actions[agent_a_name]
                    
                    # Get second agent's action or default to "No" (defect)
                    if len(agent_names) >= 2:
                        agent_b_name = agent_names[1]
                        agent_b_action = self._pending_actions[agent_b_name]
                    else:
                        # Only one agent submitted, the other defaults to "No" (defect)
                        agent_b_name = "Agent-B"  # Placeholder name
                        agent_b_action = "No"

                    # Calculate payoffs
                    payoff_a, payoff_b = self._calculate_payoff(
                        agent_a_action, agent_b_action
                    )

                    # Build round summary
                    round_summary = {
                        "round": self.round_number,
                        "agent_a_name": agent_a_name,
                        "agent_a_action": agent_a_action,
                        "agent_b_name": agent_b_name,
                        "agent_b_action": agent_b_action,
                        "agent_a_payoff": payoff_a,
                        "agent_b_payoff": payoff_b,
                    }

                    self.round_history.append(round_summary)

                    # Clear pending actions for next round
                    self._pending_actions.clear()

    def _dump_state(self) -> dict:
        """Serialize state"""
        return {
            "payoff_cc": self.payoff_cc,
            "payoff_cd": self.payoff_cd,
            "payoff_dc": self.payoff_dc,
            "payoff_dd": self.payoff_dd,
            "round_number": self.round_number,
            "round_history": self.round_history,
            "pending_actions": self._pending_actions,
        }

    def _load_state(self, state: dict):
        """Deserialize state"""
        self.payoff_cc = state.get("payoff_cc", 3)
        self.payoff_dc = state.get("payoff_dc", 5)
        self.payoff_cd = state.get("payoff_cd", 0)
        self.payoff_dd = state.get("payoff_dd", 1)
        self.round_number = state.get("round_number", 0)
        self.round_history = state.get("round_history", [])
        self._pending_actions = state.get("pending_actions", {})


__all__ = ["PrisonersDilemmaEnv"]
