"""
Public Goods Game Agent
Agent for Public Goods Game based on V2 framework
"""
import json
import re
from datetime import datetime
from typing import Any

from simframework.agent.base import AgentBase


class PublicGoodsAgent(AgentBase):
    """Agent for Public Goods Game based on V2 framework"""

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP agent module candidate list.
        Includes parameter descriptions.
        """
        description = f"""{cls.__name__}: Agent for Public Goods Game.

**Description:** {cls.__doc__ or 'No description available'}

**Initialization Parameters:**
- id (int): The unique identifier for the agent.
- name (str): The name of the agent.
- num_rounds (int, optional): Number of rounds in the game (default: 10).
- num_agents (int, optional): Number of agents in the game (default: 4).
- initial_endowment (int, optional): Initial coins per agent per round (default: 20).
- public_pool_multiplier (float, optional): Multiplier for public pool contributions (default: 1.6).

**Game Rules:**
This agent participates in a multi-round Public Goods Game. Each round, players receive an endowment and can contribute 0 to all coins to a public fund. The total public fund is multiplied and equally shared among all players. Players aim to maximize cumulative coins while balancing individual benefit and collective cooperation.

**Example initialization config:**
```json
{{
  "id": 1,
  "name": "Agent A",
  "num_rounds": 10,
  "num_agents": 4,
  "initial_endowment": 20,
  "public_pool_multiplier": 1.6
}}
```
"""
        return description

    def __init__(self, id: int, name: str, num_rounds: int = 10, num_agents: int = 4, 
                 initial_endowment: int = 20, public_pool_multiplier: float = 1.6):
        """Initialize PublicGoodsAgent
        
        Args:
            id: Agent ID
            name: Agent name
            num_rounds: Number of rounds in the game (default: 10)
            num_agents: Number of agents in the game (default: 4)
            initial_endowment: Initial coins per agent per round (default: 20)
            public_pool_multiplier: Multiplier for public pool contributions (default: 1.6)
        """
        profile = {"name": name}
        super().__init__(id=id, profile=profile)
        self._name = name
        self.history = []  # Store history records
        self.num_rounds = num_rounds
        self.num_agents = num_agents
        self.initial_endowment = initial_endowment
        self.public_pool_multiplier = public_pool_multiplier
        self.max_contribution = initial_endowment  # Maximum contribution amount
        self.min_contribution = 0  # Minimum contribution amount

    @property
    def name(self):
        """Return agent name"""
        return self._name

    async def dump(self) -> dict:
        """Export agent state"""
        return {
            "id": self._id,
            "name": self._name,
            "profile": self._profile,
            "history": self.history,
        }

    async def load(self, dump_data: dict):
        """Load agent state"""
        self._id = dump_data.get("id", self._id)
        self._name = dump_data.get("name", self._name)
        self._profile = dump_data.get("profile", self._profile)
        self.history = dump_data.get("history", [])

    async def ask(self, message: str, readonly: bool = True) -> str:
        """Answer questions"""
        profile_str = self._build_profile_string()
        prompt = f"{profile_str}\n\n{message}"
        try:
            response = await self.acompletion(
                [{"role": "user", "content": prompt}], stream=False
            )
            if response and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and choice.message:
                    return choice.message.content or ""  # type: ignore
            return "[错误] LLM返回空响应"
        except Exception as e:
            error_message = f"LLM调用失败: {type(e).__name__} - {str(e)}"
            self._logger.error(f"[{self.name}] {error_message}")
            return f"[错误] {error_message}"

    async def step(self, tick: int, t: datetime) -> str:
        """Execute one step - make contribution decision and submit to environment"""
        if self._env is None:
            return f"[{self.name}] Environment not initialized"
        
        try:
            # Step 1: Get round history from environment
            ctx = {}
            ctx, history_response = await self.ask_env(
                ctx,
                "Please call get_round_history() to get the round history.",
                readonly=True
            )
            self._logger.debug(f"[{self.name}] History response: {history_response}")
            
            # Parse and update local history
            round_history = self._parse_round_history(history_response)
            self._sync_history(round_history)
            
            # Determine current round number (next round = len(history) + 1)
            current_round = len(self.history) + 1
            
            # Extract all agent names from history if available
            all_agent_names = self._extract_agent_names()
            
            # Step 2: Decide contribution amount using LLM
            contribution, explanation = await self._decide_contribution(
                current_round, all_agent_names
            )
            
            # Step 3: Submit contribution to environment
            ctx, submit_response = await self.ask_env(
                ctx,
                f"Please call submit_contribution(agent_name='{self.name}', contribution={contribution}) to submit my contribution decision.",
                readonly=False
            )
            self._logger.info(
                f"[{self.name}] Round {current_round}: Submitted contribution={contribution}, "
                f"explanation={explanation[:50]}..."
            )
            
            return f"[{self.name}] Round {current_round}: Submitted contribution {contribution}"
            
        except Exception as e:
            error_message = f"Step execution failed: {type(e).__name__} - {str(e)}"
            self._logger.error(f"[{self.name}] {error_message}")
            return f"[{self.name}] [ERROR] {error_message}"

    async def _decide_contribution(
        self, round_num: int, all_agent_names: list
    ) -> tuple[int, str]:
        """Decide contribution amount using LLM"""
        # Build history string
        history_str = self._build_history_string(all_agent_names)

        # Build prompt
        profile_str = self._build_profile_string()
        prompt = (
            f"{profile_str}\n"
            f"Current Game State:\n"
            f"This is round {round_num} of {self.num_rounds}.\n"
            f"You have {self.initial_endowment} coins.\n"
            f"Public fund contributions are multiplied by {self.public_pool_multiplier} and divided equally among all {self.num_agents} players.\n\n"
            f"{history_str}\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, determine your contribution amount.\n"
            f"Your decision must be an integer between {self.min_contribution} and {self.max_contribution} (inclusive).\n"
            "State the integer amount first, followed by a brief 1-2 sentence explanation."
        )

        contribution = 0  # Default contribution (free-riding strategy)
        explanation = "LLM call or parsing failed"

        try:
            response = await self.acompletion(
                [{"role": "user", "content": prompt}], stream=False
            )

            if not response or not response.choices or len(response.choices) == 0:
                raise ValueError("LLM returned empty response")

            choice = response.choices[0]
            if not hasattr(choice, "message") or not choice.message:
                raise ValueError("LLM returned empty response")

            content = choice.message.content or ""  # type: ignore
            self._logger.debug(f"[{self.name}] [DEBUG] Raw response: {content}")

            if not content or content.isspace():
                raise ValueError("LLM returned empty response")

            # Use regex to match number and explanation
            match = re.search(r"^\s*(\d+)\s*[-–—]?\s*(.*)$", content, re.DOTALL)

            if match:
                parsed_contribution = int(match.group(1))

                if self.min_contribution <= parsed_contribution <= self.max_contribution:
                    contribution = parsed_contribution
                else:
                    self._logger.warning(
                        f"[{self.name}] [WARNING] Contribution value out of range: {parsed_contribution}, using default 0"
                    )
                    contribution = 0

                if match.group(2):
                    explanation = match.group(2).strip()
                else:
                    explanation = "No explanation provided"

                self._logger.info(
                    f"[{self.name}] [SUCCESS] Final parsed: contribution={contribution}, explanation={explanation[:50]}..."
                )
            else:
                # Fallback: search entire response for number
                keyword_match = re.search(r"\b(\d+)\b", content)
                if keyword_match:
                    parsed_contribution = int(keyword_match.group(1))
                    if (
                        self.min_contribution
                        <= parsed_contribution
                        <= self.max_contribution
                    ):
                        contribution = parsed_contribution
                    else:
                        contribution = 0
                    explanation = f"Extracted keyword from response: {contribution}"
                    self._logger.info(
                        f"[{self.name}] [KEYWORD] Keyword match successful: contribution={contribution}"
                    )
                else:
                    raise ValueError(
                        f"Failed to parse valid contribution, content:\n{content[:200]}"
                    )

        except Exception as e:
            error_message = f"Parsing/call failed: {type(e).__name__} - {str(e)}"
            self._logger.error(f"[{self.name}] [ERROR] {error_message}")

            contribution = 0
            explanation = f"[CRITICAL FAILURE] {error_message}, using default selection: 0"

        self._logger.debug(f"[{self.name}] [DEBUG] Final selection: {contribution}")
        return contribution, explanation

    def _build_history_string(self, all_agent_names: list) -> str:
        """Build history string"""
        if not self.history:
            return "No previous rounds have been played."

        history_lines = []
        history_lines.append("History of previous rounds:")

        cumulative_payoffs = {name: 0 for name in all_agent_names}

        for round_summary in self.history:
            r = round_summary["round"]
            total_contrib = round_summary["total_contribution"]
            public_gain = round_summary["public_pool_gain"]
            payoffs = round_summary["payoffs"]

            for name in all_agent_names:
                cumulative_payoffs[name] += payoffs.get(name, 0)

            history_lines.append(f"Round {r}:")
            history_lines.append(f"  Total contributed to public fund: {total_contrib} coins.")
            history_lines.append(f"  Public fund gain: {public_gain:.2f} coins.")

            payoff_details = [
                f"{name} gained {payoffs.get(name, 0):.2f} coins"
                for name in all_agent_names
            ]
            history_lines.append(f"  " + ", ".join(payoff_details) + ".")

            cumulative_details = [
                f"{name} has {cumulative_payoffs[name]:.2f} coins total"
                for name in all_agent_names
            ]
            history_lines.append(f"  Cumulative: " + ", ".join(cumulative_details) + ".")

        return "\n".join(history_lines)

    def _build_profile_string(self) -> str:
        """Build profile string with complete game rules matching baseline"""
        # Calculate example values for the profile
        total_contribution = self.num_agents * self.initial_endowment
        public_gain = total_contribution * self.public_pool_multiplier
        per_player_gain = public_gain / self.num_agents
        
        return (
            f"You are {self._name}, a participant in a {self.num_rounds}-round Public Goods Game.\n"
            f"Your goal is to maximize your personal cumulative coins.\n"
            f"Rules: Each round, you receive {self.initial_endowment} coins.\n"
            f"- You can contribute 0 to {self.initial_endowment} coins to a public fund.\n"
            f"- The total public fund is multiplied by {self.public_pool_multiplier} and equally shared among all players.\n"
            f"- Your round gain = (coins not contributed) + (total public fund * {self.public_pool_multiplier} / {self.num_agents}).\n"
            f"- You know all past contributions and gains.\n\n"
            f"Strategic Note: While you can choose to contribute 0 coins (free-ride), consider that if everyone contributes, you'll all gain more.\n"
            f"For example, if all {self.num_agents} players contribute all {self.initial_endowment} coins:\n"
            f"- Total contribution: {self.num_agents} × {self.initial_endowment} = {total_contribution}\n"
            f"- Public fund gain: {total_contribution} × {self.public_pool_multiplier} = {public_gain}\n"
            f"- Each player gets: {public_gain} / {self.num_agents} = {per_player_gain} coins from the public fund\n"
            f"- Your total gain: {per_player_gain} coins (vs. {self.initial_endowment} coins if everyone free-rides)"
        )

    def _parse_round_history(self, response: str) -> list:
        """Parse round history from environment response"""
        try:
            # Try to parse as JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Return empty list if parsing fails
        return []

    def _sync_history(self, round_history: list):
        """Sync local history with environment history"""
        # Update local history to match environment history
        # Only add new rounds that we don't have yet
        existing_rounds = {r.get("round") for r in self.history}
        for round_data in round_history:
            round_num = round_data.get("round")
            if round_num and round_num not in existing_rounds:
                self.history.append(round_data)

    def _extract_agent_names(self) -> list:
        """Extract all agent names from history"""
        agent_names = set()
        for round_summary in self.history:
            payoffs = round_summary.get("payoffs", {})
            agent_names.update(payoffs.keys())
        
        # If no history, return default names
        if not agent_names:
            return [f"Agent {chr(65 + i)}" for i in range(self.num_agents)]
        
        return sorted(list(agent_names))

