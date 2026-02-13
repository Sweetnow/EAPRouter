# SimFramework: A Composable Framework for LLM Agent Social Simulation

**SimFramework** is a composable and efficient framework for building LLM agent-based social simulation environments through generative code execution. It addresses the challenges of modularity, reusability, and efficiency in existing LLM agent simulation systems.

## Features

### 1. Unified Agent-Environment Interface

SimFramework provides a standardized interface that unifies agent-environment interactions using:
- **Input**: Natural language instructions + structured context (`ctx`)
- **Output**: Textual responses + structured results (`results`)

This design preserves expressiveness and accuracy while enabling strong generality across different simulation scenarios.

### 2. Environment Module Integration

Integrate existing code with minimal modifications using Python decorators and Pydantic models:

```python
from simframework.env import EnvBase, tool
from pydantic import BaseModel, Field

class SubmitActionResponse(BaseModel):
    """Response model for submit_action() function"""
    agent_name: str = Field(..., description="Agent name")
    action: str = Field(..., description="Action (Yes/No)")
    status: str = Field(..., description="Status: 'submitted' or 'round_executed'")

class PrisonersDilemmaEnv(EnvBase):
    """Environment for Prisoner's Dilemma game"""

    def __init__(self, payoff_cc: int = 3, payoff_cd: int = 0,
                 payoff_dc: int = 5, payoff_dd: int = 1):
        super().__init__()
        self.payoff_cc = payoff_cc
        self.payoff_cd = payoff_cd
        self.payoff_dc = payoff_dc
        self.payoff_dd = payoff_dd
        self.round_number = 0
        self._pending_actions = {}

    @tool(readonly=False)
    async def submit_action(self, agent_name: str, action: str) -> SubmitActionResponse:
        """Submit action decision for an agent."""
        self._pending_actions[agent_name] = action
        return SubmitActionResponse(
            agent_name=agent_name, action=action, status="submitted"
        )

    @tool(readonly=True, kind="statistics")
    async def get_payoff_matrix(self):
        """Get payoff matrix."""
        return {"payoff_cc": self.payoff_cc, "payoff_cd": self.payoff_cd}

    async def step(self, tick: int, t: datetime):
        """Run forward one step."""
        # Game logic implementation
        pass
```

The framework automatically:
- Collects tool metadata via metaclass
- Generates function schemas for LLM consumption
- Validates tool parameters using Pydantic models
- Supports three tool types: regular, `observe` (single agent parameter), and `statistics` (no parameters)

### 3. CodeGenRouter: Generative Code Execution

The core innovation - **CodeGenRouter** - automatically generates and executes Python code to call environment module functions:

- LLM generates Python code based on instructions and available tools
- AST-based security validation ensures safe execution
- Single-round execution reduces token consumption and latency compared to multi-turn ReAct patterns
- Automatic retry with error feedback for robustness

### 4. Built-in Environment Modules

Over **15 environment modules** from diverse research domains are available in `simframework/contrib/env/`:

| Domain | Modules |
|---------|----------|
| **Mobility** | `MobilitySpace` - Spatial movement with AOI (Area of Interest) support |
| **Social** | `SimpleSocialSpace` - Direct messaging, `SocialMediaSpace` - Social media with recommendation algorithms |
| **Economic** | `EconomySpace`, `CommonsTragedy`, `PublicGoods` |
| **Game Theory** | `PrisonersDilemma`, `TrustGame`, `ReputationGame`, `VolunteerDilemma` |
| **Events** | `EventSpace` - Event scheduling and management |
| **Information** | `GlobalInformation` - Shared state management |

All modules are composable - mix and match them to build complex multi-domain scenarios.

### 5. Multiple Router Strategies

Support for different agent-environment interaction patterns:

- **`CodeGenRouter`** - Generative code execution (recommended for efficiency)
- **`ReActRouter`** - ReAct pattern (reasoning + acting)
- **`PlanExecuteRouter`** - Plan-then-execute pattern
- **`TwoTierReActRouter`** - Two-level ReAct
- **`TwoTierPlanExecuteRouter`** - Two-level plan-execute

## Installation

```bash
# Install from source
git clone <repository-url>
cd simframework
pip install -e .

# Or using uv (recommended)
uv pip install -e .
```

### Requirements

- Python 3.11+
- Dependencies listed in `pyproject.toml`

## Quick Start

### Complete Simulation Example

Here's a complete example from `main.py` showing how to run a DailyMobility simulation:

```python
import asyncio
import os
import json
from datetime import datetime
from simframework.contrib.env.mobility_space import MobilitySpace
from simframework.contrib.env.event_space import EventSpace
from simframework.agent import PersonAgent
from simframework.env import CodeGenRouter
from simframework.society import SimFramework  # Alias for AgentSimulation

async def main():
    # 1. Load agent profiles
    profiles_path = "profiles.json"
    with open(profiles_path, "r") as f:
        profiles = json.load(f)

    # 2. Create environment modules
    home_dir = os.path.expanduser("~/SimFramework_data")
    map_path = os.path.join(home_dir, "beijing.pb")
    os.makedirs(home_dir, exist_ok=True)

    mobility_env = MobilitySpace(map_path, home_dir, persons=[...])
    event_space = EventSpace()

    # 3. Create CodeGenRouter with environment modules
    env_router = CodeGenRouter(
        env_modules=[mobility_env, event_space],
        log_path=f"logs/instruction_log_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl",
    )

    # 4. Generate world description from tools
    world_description = await env_router.generate_world_description_from_tools()

    # 5. Create agents
    agents = []
    for profile in profiles:
        agent = PersonAgent(
            id=profile["id"],
            profile=profile,
            memory_config={...},
            world_description=world_description,
        )
        agents.append(agent)

    # 6. Create and run society
    START_TIME = datetime.now().replace(hour=0, minute=0, second=0)
    TIME_STEP_SECONDS = 900  # 15 minutes
    TOTAL_STEPS = 97

    society = SimFramework(  # Uses AgentSimulation class
        agents=agents,
        env_router=env_router,
        start_t=START_TIME,
    )
    await society.init()
    await society.run(num_steps=TOTAL_STEPS, tick=TIME_STEP_SECONDS)

asyncio.run(main())
```

### Environment Module Example: Prisoner's Dilemma

A complete example showing all features of the environment module system:

```python
"""
Prisoner's Dilemma Game Environment
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from simframework.env import EnvBase, tool

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
    """Environment for Prisoner's Dilemma game"""

    def __init__(
        self,
        payoff_cc: int = 3,
        payoff_cd: int = 0,
        payoff_dc: int = 5,
        payoff_dd: int = 1,
    ):
        """Initialize environment with custom payoff matrix"""
        super().__init__()
        self.payoff_cc = payoff_cc
        self.payoff_cd = payoff_cd
        self.payoff_dc = payoff_dc
        self.payoff_dd = payoff_dd
        self.round_number = 0
        self.round_history: List[dict] = []
        self._pending_actions: Dict[str, str] = {}

    @property
    def description(self):
        """Description for LLM - explains the game rules and available operations"""
        return f"""You are a Prisoner's Dilemma game environment module.

**Game Rules:**
- Two players: Agent-A and Agent-B
- Each round: Choose "Yes" (cooperate) or "No" (defect)
- Payoffs determined by payoff matrix:
  * Both cooperate (Yes, Yes): {self.payoff_cc} points each
  * You cooperate, opponent defects (Yes, No): {self.payoff_cd} for you, {self.payoff_dc} for opponent
  * You defect, opponent cooperates (No, Yes): {self.payoff_dc} for you, {self.payoff_cd} for opponent
  * Both defect (No, No): {self.payoff_dd} points each

**Available Operations:**
1. submit_action(agent_name, action): Submit your decision
2. get_payoff_matrix(): View payoff structure
3. get_round_history(round_num=None): View past results
"""

    @tool(readonly=False)
    async def submit_action(self, agent_name: str, action: str) -> SubmitActionResponse:
        """Submit action decision for an agent."""
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
        """Get payoff matrix - shows all possible outcomes."""
        return GetPayoffMatrixResponse(
            payoff_cc=self.payoff_cc,
            payoff_cd=self.payoff_cd,
            payoff_dc=self.payoff_dc,
            payoff_dd=self.payoff_dd,
        )

    @tool(readonly=True, kind="observe")
    async def get_round_history(self, round_num: Optional[int] = None) -> List[dict]:
        """Get round history. Returns all rounds or specific round data."""
        if round_num is not None:
            return [r for r in self.round_history if r.get("round") == round_num]
        return self.round_history.copy()

    async def step(self, tick: int, t: datetime):
        """Execute a round when both agents have submitted."""
        self.t = t
        if len(self._pending_actions) >= 1:
            # Execute round with payoff calculation
            self.round_number += 1
            # ... round execution logic ...
            self._pending_actions.clear()

    async def init(self, start_datetime: datetime):
        """Initialize environment"""
        await super().init(start_datetime)

    def _dump_state(self) -> dict:
        """Serialize state for saving"""
        return {
            "payoff_cc": self.payoff_cc,
            "round_number": self.round_number,
            "round_history": self.round_history,
        }

    def _load_state(self, state: dict):
        """Restore state from saved data"""
        self.payoff_cc = state.get("payoff_cc", 3)
        self.round_number = state.get("round_number", 0)
        self.round_history = state.get("round_history", [])
```

### Social Space Example: Direct Messaging

```python
"""
Social Space Environment - provides social communication
"""
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from simframework.env import EnvBase, tool

class Message(BaseModel):
    """Message model for social space"""
    sender_id: int
    content: str
    timestamp: datetime
    message_id: int

class SendMessageResponse(BaseModel):
    """Response model for send_message() function"""
    sender_id: int = Field(..., description="The ID of sender agent")
    receiver_id: int = Field(..., description="The ID of receiver agent")
    content: str = Field(..., description="The content of message")

class SimpleSocialSpace(EnvBase):
    def __init__(self, agent_id_name_pairs: List[Tuple[int, str]]):
        """Initialize with agent list"""
        super().__init__()
        self._messages: List[Message] = []
        self._agent_names = {agent_id: name for agent_id, name in agent_id_name_pairs}
        self._message_id_counter = 0

    @tool(readonly=False)
    async def send_message(self, sender_id: int, receiver_id: int, content: str) -> SendMessageResponse:
        """Send a message from one agent to another."""
        self._message_id_counter += 1
        message = Message(
            sender_id=sender_id,
            content=content,
            timestamp=self.t,
            message_id=self._message_id_counter,
        )
        self._messages.append(message)
        return SendMessageResponse(
            sender_id=sender_id, receiver_id=receiver_id, content=content
        )

    @tool(readonly=True, kind="observe")
    async def receive_messages(self, agent_id: int) -> List[dict]:
        """Get all messages for a specific agent."""
        agent_messages = [m for m in self._messages if m.receiver_id == agent_id]
        return [
            {
                "sender_id": m.sender_id,
                "sender_name": self._agent_names.get(m.sender_id, f"Agent_{m.sender_id}"),
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in agent_messages
        ]

    async def init(self, start_datetime: datetime):
        """Initialize environment"""
        await super().init(start_datetime)

    async def step(self, tick: int, t: datetime):
        """Run forward one step"""
        self.t = t
```

### Running Benchmarks

The framework includes comprehensive benchmarks in `env_benchmark.py`:

```bash
# Run router evaluation benchmark
python env_benchmark.py
```

**Evaluation Metrics:**
- **Tool Selection IOU** - Measures correct tool selection (Intersection over Union)
- **Sequence LCS Score** - Measures correct call ordering (Longest Common Subsequence)
- **Parameter Accuracy** - Measures correct parameter passing
- **Successful Call** - Overall success rate combining all metrics

### Running Example Simulations

```bash
# Daily mobility benchmark with 40 agents
python main.py --num-agents 40

# Run with different profiles starting from index 100
python main.py --num-agents 100 --profile-start-idx 100
```

## Project Structure

```
simframework/
├── agent/                  # Agent implementations
│   ├── base.py             # AgentBase abstract class with profile support
│   └── person.py           # PersonAgent with memory & LLM integration
├── env/                   # Core environment framework
│   ├── base.py             # EnvBase class and @tool decorator
│   ├── router_codegen.py    # CodeGenRouter - generative code execution
│   ├── router_react.py      # ReActRouter - ReAct pattern
│   ├── router_plan_execute.py    # Plan-then-execute
│   └── ...                # Other router implementations
├── contrib/env/            # Built-in environment modules
│   ├── mobility_space/      # Spatial movement simulation
│   ├── social_media/        # Social media with algorithms
│   ├── simple_social_space.py
│   ├── prisoners_dilemma.py
│   ├── public_goods.py
│   ├── economy_space.py
│   ├── commons_tragedy.py
│   ├── trust_game.py
│   ├── reputation_game.py
│   ├── volunteer_dilemma.py
│   └── event_space.py
├── society/                # Society-level simulation management
│   ├── society.py          # AgentSimulation - main simulation orchestrator
│   └── helper.py          # Helper utilities
├── config/                 # Configuration management
├── storage/                # Replay data storage
└── logger/                 # Logging utilities
```

## Tool Types and Decorator Parameters

The `@tool` decorator supports several parameters:

```python
@tool(
    readonly=False,           # Can this tool modify environment state?
    name="custom_name",     # Override default function name
    description="Custom desc", # Override docstring
    kind=None                # Tool type: None (regular), "observe", "statistics"
)
async def my_tool(self, agent_id: int, param: str):
    pass
```

- **Regular tools** (`kind=None`): Can be read-write or read-only
- **Observe tools** (`kind="observe"`): Must be `readonly=True`, accept single `agent_id` parameter
- **Statistics tools** (`kind="statistics"`): Must be `readonly=True`, accept no parameters

## Key Design Principles

1. **Composability** - Mix and match environment modules freely across domains
2. **Efficiency** - Single-round code generation reduces token overhead by 60-80%
3. **Safety** - AST validation and sandboxed execution with whitelist-based imports
4. **Extensibility** - Easy to add new modules via decorators and Pydantic models
5. **Research-Ready** - Built-in benchmarking, replay system, and evaluation tools

## Advanced Features

### Memory System
Agents have access to persistent memory with vector store support:
```python
agent = PersonAgent(
    id=1,
    profile=profile,
    memory_config={
        "vector_store": {
            "provider": "chroma",  # or "qdrant"
            "config": {"collection_name": "agent_memories"}
        }
    }
)
```

### Replay System
Full simulation replay with SQLite storage:
```python
from simframework.storage import ReplayWriter

replay_writer = ReplayWriter("sqlite.db")
society = SimFramework(
    agents=agents,
    env_router=router,
    start_t=start_time,
    replay_writer=replay_writer,
)
# All agent actions, tool calls, and state changes are recorded
```

## Acknowledgments

This framework builds upon insights from the LLM agent simulation community, including:
- SimFramework and AgentKernel for systematic simulation design
- Model Context Protocol (MCP) for tool interface standardization
- ReAct and Plan-Execute patterns for agent reasoning
