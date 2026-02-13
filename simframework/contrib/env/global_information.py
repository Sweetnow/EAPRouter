"""
Global Information Environment
This environment provides global information to the agent.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from simframework.env import (
    EnvBase,
    tool,
)


class GetGlobalInformationResponse(BaseModel):
    """Response model for get() function"""

    global_information: str = Field(..., description="The global information")


class SetGlobalInformationResponse(BaseModel):
    """Response model for set() function"""

    old_information: str = Field(..., description="The old information")
    new_information: str = Field(..., description="The new information")


class GlobalInformationEnv(EnvBase):
    def __init__(self):
        """
        Initialize the Global Information Environment.
        """
        super().__init__()
        self._global_information = "It's a normal day without any special events."

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP environment module candidate list.
        Includes parameter descriptions.
        """
        description = f"""{cls.__name__}: Global information environment module.

**Description:** Provides global information to the agent like weather, global news, etc.

**Initialization Parameters (excluding llm):**
No additional parameters required. This module only requires the llm parameter.

**Example initialization config:**
```json
{{}}
```
"""
        return description

    @property
    def description(self):
        """Description of the environment module for router selection and function calling"""
        return """You are a global information environment module specialized in providing global information to the agent.

Your task is to use the get and set functions with context paths to provide global information to the agent."""

    @tool(readonly=True, kind="observe")
    async def get(self) -> GetGlobalInformationResponse:
        """
        Get the global information.

        Returns:
            The global information.
        """
        return GetGlobalInformationResponse(global_information=self._global_information)

    @tool(readonly=False)
    async def set(self, prompt: str) -> SetGlobalInformationResponse:
        """
        Set the global information.

        Args:
            prompt: The global information.

        Returns:
            The global information.
        """
        old_information = self._global_information
        self._global_information = prompt
        return SetGlobalInformationResponse(
            old_information=old_information,
            new_information=prompt,
        )

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step.

        Args:
            tick: The number of ticks of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        self.t = t
