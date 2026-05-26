"""
Prisoner's Dilemma Game Agent
Agent for Prisoner's Dilemma game based on V2 framework
"""
import json
import re
from datetime import datetime
from typing import Any

from eaprouter.agent.base import AgentBase


class PrisonersDilemmaAgent(AgentBase):
    """Agent for Prisoner's Dilemma game based on V2 framework"""

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP agent module candidate list.
        Includes parameter descriptions.
        """
        description = f"""{cls.__name__}: Agent for Prisoner's Dilemma game.

**Description:** {cls.__doc__ or 'No description available'}

**Initialization Parameters:**
- id (int): The unique identifier for the agent.
- name (str): The name of the agent.

**Game Rules:**
This agent participates in a 10-round Prisoner's Dilemma game where two players simultaneously choose to cooperate (Yes) or defect (No). The payoff matrix is:
- Both cooperate: 3 points each
- One cooperates, one defects: 0 points (cooperator), 5 points (defector)
- Both defect: 1 point each

**Example initialization config:**
```json
{{
  "id": 1,
  "name": "Agent A"
}}
```
"""
        return description

    def __init__(self, id: int, name: str):
        """Initialize PrisonersDilemmaAgent
        
        Args:
            id: Agent ID
            name: Agent name
        """
        profile = {"name": name}
        super().__init__(id=id, profile=profile)
        self._name = name
        self.history = []  # Store history records

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
        """Execute one step - make action decision and submit to environment"""
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
            
            # Extract opponent name from history if available
            opponent_name = self._extract_opponent_name()
            
            # Step 2: Decide action using LLM
            action, explanation = await self._decide_action(
                current_round, opponent_name
            )
            
            # Step 3: Submit action to environment
            ctx, submit_response = await self.ask_env(
                ctx,
                f"Please call submit_action(agent_name='{self.name}', action='{action}') to submit my action decision.",
                readonly=False
            )
            self._logger.info(
                f"[{self.name}] Round {current_round}: Submitted action={action}, "
                f"explanation={explanation[:50]}..."
            )
            
            return f"[{self.name}] Round {current_round}: Submitted action {action}"
            
        except Exception as e:
            error_message = f"Step execution failed: {type(e).__name__} - {str(e)}"
            self._logger.error(f"[{self.name}] {error_message}")
            return f"[{self.name}] [ERROR] {error_message}"

    async def _decide_action(self, round_num: int, opponent_name: str) -> tuple[str, str]:
        """Decide action using LLM - determine cooperate or defect"""
        # Build history string
        history_str = self._build_history_string(opponent_name)

        # Build prompt
        profile_str = self._build_profile_string()
        prompt = (
            f"{profile_str}\n"
            f"This is round {round_num}.\n"
            f"{history_str}\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, determine your action.\n"
            "Your decision must be either **Yes (Cooperate)** or **No (Defect/Betray)**.\n"
            "YOU MUST FOLLOW THIS OUTPUT FORMAT EXACTLY:\n"
            "\n"
            "<output>\n"
            "  <action>Yes or No</action>\n"
            "  <explanation>Your 1-2 sentence reasoning here.</explanation>\n"
            "</output>"
            "\n"
            "IMPORTANT: Do not write any additional text before or after the XML structure. Your entire output must consist of exactly the XML format shown above."
        )

        action = "No"  # Default action (defect)
        explanation = "LLM调用或解析失败"

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
            self._logger.debug(f"[{self.name}] [DEBUG] 原始响应: {content}")

            if not content or content.isspace():
                raise ValueError("LLM返回空响应")

            # Extract Action using regex
            action_match = re.search(
                r"<action>\s*(Yes|No)\s*</action>", content, re.IGNORECASE
            )
            if action_match:
                action = action_match.group(1).capitalize()

                # Extract Explanation
                explanation_match = re.search(
                    r"<explanation>(.*?)</explanation>", content, re.DOTALL | re.IGNORECASE
                )
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                else:
                    explanation = f"选择了 '{action}'，但未在 XML 中提供解释"

                self._logger.info(
                    f"[{self.name}] [SUCCESS] 最终解析 (XML): action={action}, explanation={explanation[:50]}..."
                )
            else:
                # Fallback: single-line matching
                self._logger.warning(f"[{self.name}] [WARNING] XML 解析失败，尝试单行匹配...")
                lines = content.strip().split("\n")
                first_line = lines[0].strip()

                match = re.search(r"^\s*(yes|no)\s*[.!]?$", first_line, re.IGNORECASE)

                if match:
                    action = match.group(1).capitalize()
                    explanation = (
                        " ".join(lines[1:]).strip()
                        or "单行匹配成功，无详细解释"
                    )
                    self._logger.info(f"[{self.name}] [FALLBACK] 单行匹配成功: action={action}")
                else:
                    # Final fallback: keyword search
                    keyword_match = re.search(r"\b(yes|no)\b", content, re.IGNORECASE)
                    if keyword_match:
                        action = keyword_match.group(1).capitalize()
                        explanation = f"从响应中提取关键词：{action}"
                        self._logger.info(f"[{self.name}] [KEYWORD] 关键词匹配成功: action={action}")
                    else:
                        raise ValueError(f"无法解析有效动作，内容:\n{content[:200]}")

        except Exception as e:
            error_message = f"解析/调用失败: {type(e).__name__} - {str(e)}"
            self._logger.error(f"[{self.name}] [ERROR] {error_message}")

            action = "No"
            explanation = f"[CRITICAL FAILURE] {error_message}，使用默认选择: No"

        self._logger.debug(f"[{self.name}] [DEBUG] 最终选择: {action}")
        return action, explanation

    def _build_profile_string(self) -> str:
        """Build profile string from profile dict"""
        if isinstance(self._profile, dict):
            name = self._profile.get("name", f"Agent {self.id}")
            # Build complete game rules description matching baseline
            return (
                "You are a rational decision-maker. You are playing a game that is played in multiple rounds with another player. "
                "The game consists of 10 rounds, and you are allowed to remember the past actions of your opponent. "
                "In each round, both of you will simultaneously choose one of two actions: Yes (Cooperate) or No (Defect/Betray). "
                "Your goal is to maximize your benefits. "
                "The following is the payoff matrix that determines your benefits: "
                "- If both choose Yes: You get 3 points, your opponent gets 3 points "
                "- If you choose Yes and your opponent chooses No: You get 0 points, your opponent gets 5 points "
                "- If you choose No and your opponent chooses Yes: You get 5 points, your opponent gets 0 points "
                "- If both choose No: You get 1 point, your opponent gets 1 point "
                "Remember this matrix and make decisions to maximize your total points. You must output 1-2 sentences to explain your decision."
            )
        elif isinstance(self._profile, str):
            return self._profile
        else:
            # Fallback: build profile with game rules even if profile is not dict
            return (
                "You are a rational decision-maker. You are playing a game that is played in multiple rounds with another player. "
                "The game consists of 10 rounds, and you are allowed to remember the past actions of your opponent. "
                "In each round, both of you will simultaneously choose one of two actions: Yes (Cooperate) or No (Defect/Betray). "
                "Your goal is to maximize your benefits. "
                "The following is the payoff matrix that determines your benefits: "
                "- If both choose Yes: You get 3 points, your opponent gets 3 points "
                "- If you choose Yes and your opponent chooses No: You get 0 points, your opponent gets 5 points "
                "- If you choose No and your opponent chooses Yes: You get 5 points, your opponent gets 0 points "
                "- If both choose No: You get 1 point, your opponent gets 1 point "
                "Remember this matrix and make decisions to maximize your total points. You must output 1-2 sentences to explain your decision."
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
        # Convert environment format to local format: (my_action, opponent_action)
        self.history = []
        for round_data in round_history:
            round_num = round_data.get("round", 0)
            agent_a_action = round_data.get("agent_a_action", "")
            agent_b_action = round_data.get("agent_b_action", "")
            
            # Determine which action is mine based on agent name
            if agent_a_action and agent_b_action:
                # For now, assume we're Agent A (will be determined by actual agent name)
                # This will be fixed when we extract opponent name properly
                if self.name == "Agent A":
                    self.history.append((agent_a_action, agent_b_action))
                else:
                    self.history.append((agent_b_action, agent_a_action))

    def _extract_opponent_name(self) -> str:
        """Extract opponent name from history"""
        # For Prisoner's Dilemma, there's only one opponent
        # Check if we're Agent A or Agent B
        if self.name == "Agent A":
            return "Agent B"
        else:
            return "Agent A"

    def _build_history_string(self, opponent_name: str) -> str:
        """Build history string"""
        if not self.history:
            return "No previous rounds have been played."

        history_str = "History:\n"
        for i, (my_action, opponent_action) in enumerate(self.history):
            formatted_my_action = my_action.capitalize()
            formatted_opponent_action = opponent_action.capitalize()
            history_str += f"Round {i + 1}: Your choice={formatted_my_action}, {opponent_name}'s choice={formatted_opponent_action}\n"

        return history_str

