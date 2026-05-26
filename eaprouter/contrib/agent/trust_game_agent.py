"""
Trust Game Agent
Agent for Trust Game based on V2 framework
"""
import json
import re
from datetime import datetime
from typing import Literal

from eaprouter.agent.base import AgentBase


class TrustGameAgent(AgentBase):
    """Agent for Trust Game based on V2 framework"""

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP agent module candidate list.
        Includes parameter descriptions.
        """
        description = f"""{cls.__name__}: Agent for Trust Game.

**Description:** {cls.__doc__ or 'No description available'}

**Initialization Parameters:**
- id (int): The unique identifier for the agent.
- name (str): The name of the agent.
- role (Literal["Trustor", "Trustee"]): The role of the agent - either "Trustor" (investor) or "Trustee" (receiver).
- num_rounds (int, optional): Number of rounds in the game (default: 10).
- initial_funds (int, optional): Initial coins per Trustor per round (default: 10).
- multiplication_factor (int, optional): Investment multiplication factor (default: 3).
- partner_name (str, optional): Partner's name (will be set by environment).

**Game Rules:**
This agent participates in a multi-round Trust Game with two roles:
- **Trustor**: Starts with initial_funds coins per round, sends 0-initial_funds to Trustee. Amount is multiplied before Trustee receives it. Payoff = initial_funds - sent + returned.
- **Trustee**: Receives (sent amount × multiplication_factor) coins, returns 0-received to Trustor. Payoff = received - returned.

Both roles aim to maximize cumulative coins while considering trust and reciprocity.

**Example initialization config:**
```json
{{
  "id": 1,
  "name": "Trustor A",
  "role": "Trustor",
  "num_rounds": 10,
  "initial_funds": 10,
  "multiplication_factor": 3,
  "partner_name": ""
}}
```
"""
        return description

    def __init__(
        self,
        id: int,
        name: str,
        role: Literal["Trustor", "Trustee"],
        num_rounds: int = 10,
        initial_funds: int = 10,
        multiplication_factor: int = 3,
        partner_name: str = "",
    ):
        """Initialize TrustGameAgent
        
        Args:
            id: Agent ID
            name: Agent name
            role: Role: "Trustor" or "Trustee"
            num_rounds: Number of rounds in the game (default: 10)
            initial_funds: Initial coins per Trustor per round (default: 10)
            multiplication_factor: Investment multiplication factor (default: 3)
            partner_name: Partner's name (will be set by environment)
        """
        profile = {"name": name}
        super().__init__(id=id, profile=profile)
        self._name = name
        self._role = role  # Role: "Trustor" or "Trustee"
        self.history = []  # Store history records
        self._current_funds = initial_funds  # Current funds
        self.num_rounds = num_rounds
        self.initial_funds = initial_funds
        self.multiplication_factor = multiplication_factor
        self._partner_name = partner_name  # Partner's name

        # Validate role
        if self._role not in ["Trustor", "Trustee"]:
            raise ValueError(f"Invalid role: {role}. Must be 'Trustor' or 'Trustee'")

    def set_partner_name(self, partner_name: str):
        """Set partner name (called by environment)"""
        self._partner_name = partner_name

    @property
    def name(self) -> str:
        """Return agent name"""
        return self._name

    @property
    def role(self) -> str:
        """Return agent role"""
        return self._role

    async def dump(self) -> dict:
        """Export agent state"""
        return {
            "id": self._id,
            "name": self._name,
            "role": self._role,
            "profile": self._profile,
            "history": self.history,
            "current_funds": self._current_funds,
        }

    async def load(self, dump_data: dict):
        """Load agent state"""
        self._id = dump_data.get("id", self._id)
        self._name = dump_data.get("name", self._name)
        self._role = dump_data.get("role", self._role)
        self._profile = dump_data.get("profile", self._profile)
        self.history = dump_data.get("history", [])
        self._current_funds = dump_data.get("current_funds", 10)

    async def ask(self, message: str, readonly: bool = True) -> str:
        """Answer questions"""
        prompt = f"{self._profile}\n\n{message}"
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
        """Execute one step - make investment or return decision and submit to environment"""
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
            
            # Extract partner name from history if available
            partner_name = self._extract_partner_name()
            
            if self._role == "Trustor":
                # Trustor: Decide investment and submit
                investment, explanation = await self._decide_investment(
                    current_round, partner_name
                )
                
                # Submit investment to environment
                ctx, submit_response = await self.ask_env(
                    ctx,
                    f"Please call submit_investment(trustor_name='{self.name}', investment={investment}) to submit my investment decision.",
                    readonly=False
                )
                self._logger.info(
                    f"[{self.name}] Round {current_round}: Submitted investment={investment}, "
                    f"explanation={explanation[:50]}..."
                )
                return f"[{self.name}] Round {current_round}: Submitted investment {investment}"
            else:
                # Trustee: Query pending investment, decide return and submit
                # First, get partner name (trustor) to query pending investment
                trustor_name = partner_name if partner_name else self._extract_partner_name_from_env()
                
                if trustor_name:
                    # Query pending investment
                    ctx, pending_response = await self.ask_env(
                        ctx,
                        f"Please call get_pending_investment(trustor_name='{trustor_name}') to get the pending investment amount.",
                        readonly=True
                    )
                    
                    # Parse pending investment
                    pending_investment = self._parse_pending_investment(pending_response)
                    
                    if pending_investment is not None and pending_investment > 0:
                        # Calculate received amount
                        received_amount = pending_investment * self.multiplication_factor
                        self._current_funds = received_amount
                        
                        # Decide return amount
                        return_amount, explanation = await self._decide_return(
                            current_round, partner_name, received_amount
                        )
                        
                        # Submit return to environment
                        ctx, submit_response = await self.ask_env(
                            ctx,
                            f"Please call submit_return(trustee_name='{self.name}', return_amount={return_amount}) to submit my return decision.",
                            readonly=False
                        )
                        self._logger.info(
                            f"[{self.name}] Round {current_round}: Submitted return={return_amount}, "
                            f"explanation={explanation[:50]}..."
                        )
                        return f"[{self.name}] Round {current_round}: Submitted return {return_amount}"
                    else:
                        # No pending investment yet, submit 0
                        ctx, submit_response = await self.ask_env(
                            ctx,
                            f"Please call submit_return(trustee_name='{self.name}', return_amount=0) to submit my return decision.",
                            readonly=False
                        )
                        return f"[{self.name}] Round {current_round}: No pending investment, submitted return 0"
                else:
                    # Cannot determine partner, submit 0
                    ctx, submit_response = await self.ask_env(
                        ctx,
                        f"Please call submit_return(trustee_name='{self.name}', return_amount=0) to submit my return decision.",
                        readonly=False
                    )
                    return f"[{self.name}] Round {current_round}: Cannot determine partner, submitted return 0"
            
        except Exception as e:
            error_message = f"Step execution failed: {type(e).__name__} - {str(e)}"
            self._logger.error(f"[{self.name}] {error_message}")
            return f"[{self.name}] [ERROR] {error_message}"

    async def _decide_investment(self, round_num: int, partner_name: str) -> tuple[int, str]:
        """Decide investment amount using LLM"""
        # Build history string
        history_str = self._build_history_string(partner_name)

        # Build prompt
        profile_str = self._build_profile_string()
        prompt = self._build_trustor_prompt(
            round_num,
            self.initial_funds,
            self.multiplication_factor,
            self.num_rounds,
            self.initial_funds,
            history_str,
        )

        investment = 0  # Default investment
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

            content = choice.message.content or ""  # type: ignore  # noqa: E501
            self._logger.debug(f"[{self.name}] Raw LLM response: {content[:200]}...")

            if not content or content.isspace():
                raise ValueError("LLM returned empty response")

            # Parse numerical value
            investment = self._parse_amount(content, self._role, self.initial_funds)

            # Extract explanation
            explanation = self._extract_explanation(content)

            self._logger.info(
                f"[{self.name}] Decision: {investment} coins, explanation: {explanation[:50]}..."
            )

        except Exception as e:
            error_msg = f"Decision making failed: {type(e).__name__} - {str(e)}"
            self._logger.error(f"[{self.name}] {error_msg}")
            investment = 0
            explanation = f"[CRITICAL FAILURE] {error_msg}, using default amount: 0"

        return investment, explanation

    async def _decide_return(
        self, round_num: int, partner_name: str, received_amount: int
    ) -> tuple[int, str]:
        """Decide return amount using LLM"""
        # Build history string
        history_str = self._build_history_string(partner_name)

        # Build prompt
        profile_str = self._build_profile_string()
        prompt = self._build_trustee_prompt(
            round_num,
            self.initial_funds,
            self.multiplication_factor,
            self.num_rounds,
            received_amount,
            history_str,
        )

        return_amount = 0  # Default return
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

            content = choice.message.content or ""  # type: ignore  # noqa: E501
            self._logger.debug(f"[{self.name}] Raw LLM response: {content[:200]}...")

            if not content or content.isspace():
                raise ValueError("LLM returned empty response")

            # Parse numerical value
            return_amount = self._parse_amount(content, self._role, received_amount)

            # Extract explanation
            explanation = self._extract_explanation(content)

            self._logger.info(
                f"[{self.name}] Decision: {return_amount} coins, explanation: {explanation[:50]}..."
            )

        except Exception as e:
            error_msg = f"Decision making failed: {type(e).__name__} - {str(e)}"
            self._logger.error(f"[{self.name}] {error_msg}")
            return_amount = 0
            explanation = f"[CRITICAL FAILURE] {error_msg}, using default amount: 0"

        return return_amount, explanation

    def update_history(self, round_summary: dict):
        """Update agent's history with current round summary"""
        self.history.append(round_summary)

    def _build_profile_string(self) -> str:
        """Build profile string with complete game rules matching baseline"""
        if isinstance(self._profile, dict):
            name = self._profile.get("name", f"Agent {self.id}")
        else:
            name = self._name
            
        if self._role == "Trustor":
            return (
                f"You are {name}, Trustor in a {self.num_rounds}-round Trust Game.\n"
                f"Goal: Maximize cumulative coins.\n"
                f"Rules: Start with {self.initial_funds} coins/round.\n"
                f"Send 0-{self.initial_funds} coins (×{self.multiplication_factor} for Trustee).\n"
                f"Trustee returns some coins. Your payoff: {self.initial_funds} - sent + returned.\n"
                "Consider past Trustee behavior."
            )
        else:
            return (
                f"You are {name}, Trustee in a {self.num_rounds}-round Trust Game.\n"
                f"Goal: Maximize cumulative coins.\n"
                f"Rules: Receive (sent amount)×{self.multiplication_factor} coins from Trustor.\n"
                "Return 0-received coins. Your payoff: received - returned.\n"
                "Consider past Trustor behavior."
            )

    def _parse_round_history(self, response: str) -> list:
        """Parse round history from environment response"""
        try:
            # Try to extract JSON array from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                rounds = json.loads(json_match.group(0))
                if isinstance(rounds, list):
                    return rounds
        except Exception as e:
            self._logger.warning(f"[{self.name}] Failed to parse round history: {e}")
        return []

    def _sync_history(self, round_history: list):
        """Sync local history with environment round history"""
        # Update local history from environment round history
        # Convert environment format to local format
        self.history = []
        for round_data in round_history:
            round_num = round_data.get("round", 0)
            trustor_investments = round_data.get("trustor_investments", {})
            trustee_returns = round_data.get("trustee_returns", {})
            payoffs = round_data.get("payoffs", {})
            
            if self._role == "Trustor":
                sent_amount = trustor_investments.get(self.name, 0)
                # Find partner trustee
                partner_name = self._extract_partner_name()
                if partner_name:
                    returned_amount = trustee_returns.get(partner_name, 0)
                else:
                    returned_amount = 0
                payoff = payoffs.get(self.name, 0)
                
                self.history.append({
                    "round": round_num,
                    "sent_amount": sent_amount,
                    "returned_amount": returned_amount,
                    "payoff": payoff,
                    "multiplication_factor": self.multiplication_factor,
                })
            else:  # Trustee
                # Find partner trustor
                partner_name = self._extract_partner_name()
                if partner_name:
                    sent_amount = trustor_investments.get(partner_name, 0)
                    received_amount = sent_amount * self.multiplication_factor
                else:
                    sent_amount = 0
                    received_amount = 0
                returned_amount = trustee_returns.get(self.name, 0)
                payoff = payoffs.get(self.name, 0)
                
                self.history.append({
                    "round": round_num,
                    "sent_amount": sent_amount,
                    "received_amount": received_amount,
                    "returned_amount": returned_amount,
                    "payoff": payoff,
                    "multiplication_factor": self.multiplication_factor,
                })

    def _extract_partner_name(self) -> str:
        """Extract partner name"""
        # Use stored partner name if available
        if self._partner_name:
            return self._partner_name
        
        # Try to extract from history by checking round data
        if self.history:
            # Partner name should be set by environment, but we can try to infer from history
            # This is a fallback - partner mapping should be set explicitly
            return ""
        
        return ""

    def _extract_partner_name_from_env(self) -> str:
        """Extract partner name - use stored partner name"""
        return self._partner_name

    def _parse_pending_investment(self, response: str) -> int:
        """Parse pending investment amount from environment response"""
        try:
            # Try to extract number from response
            match = re.search(r'\b(\d+)\b', response)
            if match:
                return int(match.group(1))
        except Exception as e:
            self._logger.warning(f"[{self.name}] Failed to parse pending investment: {e}")
        return 0

    def _build_history_string(self, partner_name: str) -> str:
        """Build history string"""
        if not self.history:
            return "No previous rounds have been played."

        history_lines = []
        history_lines.append("History of previous rounds:")

        for round_summary in self.history:
            r = round_summary["round"]

            if self._role == "Trustor":
                sent_amount = round_summary.get("sent_amount", 0)
                returned_amount = round_summary.get("returned_amount", 0)
                payoff = round_summary.get("payoff", 0)
                history_lines.append(
                    f"Round {r}: You sent {sent_amount} coins. {partner_name} returned {returned_amount} coins. Your payoff: {payoff:.2f} coins."
                )
            else:  # Trustee
                sent_amount = round_summary.get("sent_amount", 0)
                received_amount = round_summary.get("received_amount", 0)
                returned_amount = round_summary.get("returned_amount", 0)
                payoff = round_summary.get("payoff", 0)
                history_lines.append(
                    f"Round {r}: You received {received_amount} coins (from {partner_name}'s {sent_amount} coins * {round_summary.get('multiplication_factor', 3)}). You returned {returned_amount} coins. Your payoff: {payoff:.2f} coins."
                )

        return "\n".join(history_lines)

    def _build_trustor_prompt(
        self,
        round_num: int,
        initial_funds: int,
        multiplication_factor: int,
        num_rounds: int,
        current_funds: int,
        history_str: str,
    ) -> str:
        """Build prompt for Trustor"""
        return (
            f"Round {round_num} of {num_rounds}\n"
            f"Current funds: {current_funds}\n"
            f"Send 0-{initial_funds} coins to Trustee (amount × {multiplication_factor} for them)\n"
            f"Your goal: Maximize total coins (payoff = {initial_funds} - sent + returned)\n\n"
            f"{history_str}\n\n"
            "Decision format:\n"
            "Amount: [0-10]\n"
            "Explanation: [brief strategic reason]"
        )

    def _build_trustee_prompt(
        self,
        round_num: int,
        initial_funds: int,
        multiplication_factor: int,
        num_rounds: int,
        current_funds: int,
        history_str: str,
    ) -> str:
        """Build prompt for Trustee"""
        return (
            f"Round {round_num} of {num_rounds}\n"
            f"Received: {current_funds} coins (Trustor sent {current_funds / multiplication_factor})\n"
            f"Return 0-{current_funds} coins to Trustor\n"
            f"Your goal: Maximize total coins (payoff = {current_funds} - returned)\n\n"
            f"{history_str}\n\n"
            "Decision format:\n"
            f"Amount: [0-{current_funds}]\n"
            "Explanation: [brief strategic reason]"
        )

    def _parse_amount(self, content: str, role: str, max_amount: int) -> int:
        """Parse numerical amount from LLM response"""
        match = re.search(r"\b(\d+)\b", content)
        if not match:
            raise ValueError(
                f"Failed to parse valid numerical amount from response: {content[:200]}..."
            )

        parsed_val = int(match.group(0))

        # Validate amount range based on role
        if not (0 <= parsed_val <= max_amount):
            self._logger.warning(
                f"[{self._name}] Invalid {role.lower()} amount ({parsed_val}) detected, must be between 0 and {max_amount}"
            )
            return 0  # Use 0 as safe default

        return parsed_val

    def _extract_explanation(self, content: str) -> str:
        """Extract explanation from LLM response"""
        # Look for explanation after the amount
        match = re.search(r"\b\d+\b", content)
        if match:
            explanation = content[match.end() :].strip()
            return explanation if explanation else "No explanation provided"

        # If no amount found, return entire content as explanation
        return content.strip() if content.strip() else "No explanation provided"

