# EAPRouter

EAPRouter grounds LLM agent intents in multi-module environments by translating an agent's textual intent and structured context into a restricted Python **executable action program** (EAP), validating it, executing it over environment modules, and reusing it when later requests have similar intent and input structure.

The code accompanies the EMNLP submission narrative: **Grounding LLM Agent Intent in Multi-Module Environments with Executable Action Programs**.

## Core Ideas

### Executable Action Programs

An executable action program is a restricted Python snippet used as an intermediate representation between an agent intent and concrete environment-module action calls. EAPs explicitly represent:

- action selection across modules
- action ordering and cross-module dependencies
- argument grounding from structured context
- intermediate-result reuse
- structured result construction
- textual feedback through `print()`

At runtime, EAPs receive:

- `instruction`: the agent's textual intent
- `context`: structured data, such as agent IDs and template variables
- `modules`: mounted environment modules

EAPs write structured output to `result` and may emit textual feedback with `print()`. The runtime keeps `ctx` and `results` as backward-compatible aliases for older scripts, but new EAPs should use `context` and `result`.

### EAPRouter

`EAPRouter` implements the EAP lifecycle:

- aggregates callable action interfaces from environment modules
- pre-generates EAPs for frequent requests such as `<observe>` and `<statistic>`
- generates runtime EAPs with an LLM when no reusable program exists
- validates EAPs with AST-based safety checks and restricted imports/builtins
- executes EAPs in a protected interpreter with timeout control
- caches reusable EAPs using instruction-template similarity and structured-context checks

`CodeGenRouter` remains available as a backward-compatible alias for older experiment scripts.

## Environment Module Integration

Environment modules inherit from `EnvBase` and expose callable actions with `@tool()`. Type hints, Pydantic models, and docstrings describe each action to EAPRouter.

```python
from pydantic import BaseModel, Field
from eaprouter.env import EnvBase, tool


class SubmitActionResponse(BaseModel):
    agent_name: str = Field(..., description="Agent name")
    action: str = Field(..., description="Action value")
    status: str = Field(..., description="Execution status")


class PrisonersDilemmaEnv(EnvBase):
    @tool(readonly=False)
    async def submit_action(
        self, agent_name: str, action: str
    ) -> SubmitActionResponse:
        """Submit one agent's decision for the current round."""
        return SubmitActionResponse(
            agent_name=agent_name,
            action=action,
            status="submitted",
        )

    @tool(readonly=True, kind="observe")
    async def get_round_history(self) -> list[dict]:
        """Return visible round history for observation requests."""
        return []

    @tool(readonly=True, kind="statistic")
    async def get_payoff_matrix(self) -> dict:
        """Return the payoff matrix for aggregate statistic requests."""
        return {"cc": 3, "cd": 0, "dc": 5, "dd": 1}
```

`kind="observe"` actions are grouped for `<observe>` requests. `kind="statistic"` actions are grouped for `<statistic>` requests. The older spelling `kind="statistics"` is still accepted and normalized for compatibility.

## Quick Start

```python
import asyncio
from datetime import datetime

from eaprouter.contrib.env.event_space import EventSpace
from eaprouter.contrib.env.mobility_space import MobilitySpace
from eaprouter.env import EAPRouter


async def main():
    mobility = MobilitySpace("beijing.pb", "~/eaprouter_data", persons=[])
    events = EventSpace()

    router = EAPRouter(env_modules=[mobility, events])
    await router.init(datetime.now())

    context = {"id": 3, "variables": {"poi_id": 700002365}}
    instruction = (
        "Go to Blue Maple Bistro (POI ID: {poi_id}) "
        "and start a 30-minute eating out event."
    )

    result, answer = await router.ask(
        context,
        instruction,
        template_mode=True,
    )
    print(answer)
    print(result)


asyncio.run(main())
```

## Built-In Modules

The repository includes more than 15 environment modules in `eaprouter/contrib/env/`, covering mobility, events, direct social communication, social media, economic games, public goods, commons tragedy, trust games, reputation games, volunteer dilemma, global information, and cognitive/social-effect tasks.

Modules can be mixed to create multi-domain simulation scenarios. EAPRouter exposes their action interfaces together so that one EAP can coordinate multiple modules in a single grounding procedure.

## Router Baselines

The benchmark code compares EAPRouter against function-calling baselines:

- `ReActRouter`
- `PlanExecuteRouter`
- `SearchToolRouter`
- `TwoTierReActRouter`
- `TwoTierPlanExecuteRouter`

Run the grounding benchmark with:

```bash
python env_benchmark.py
```

Run the safety benchmark with:

```bash
python security_benchmark.py --direct-only
```

The benchmark metrics follow the paper:

- **IoU**: action-selection overlap
- **NLCS**: normalized longest common subsequence for action ordering
- **Parameter Accuracy**: grounded argument correctness
- **Successful Call Ratio**: strict match of action sequence and parameters

## Project Structure

```text
eaprouter/
  agent/                  # Agent implementations
  env/
    base.py               # EnvBase and @tool integration protocol
    router_eap.py         # EAPRouter implementation; CodeGenRouter alias
    router_react.py       # ReAct function-calling baseline
    router_plan_execute.py
    router_search_tool.py
  contrib/env/            # Built-in environment modules
  society/                # Simulation orchestration
  storage/                # Replay storage
```

## Installation

```bash
uv pip install -e .
```

or:

```bash
pip install -e .
```

Python 3.11+ is required.

## Review Notice

Due to double-blind review requirements, this repository does not include the private data files required to run the full social-simulation experiments end to end. The code is provided to support inspection of the method, implementation, and benchmark logic.
