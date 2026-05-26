import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

from eaprouter.env import RouterBase
from eaprouter.agent import AgentBase
from eaprouter.society.helper import EAPRouterHelper
from eaprouter.storage import ReplayWriter

__all__ = ["EAPRouterSimulation"]


class EAPRouterSimulation:
    def __init__(
        self,
        agents: Sequence[AgentBase],
        env_router: RouterBase,
        start_t: datetime,
        run_dir: Optional[Path] = None,
        enable_replay: bool = True,
        replay_writer: Optional[ReplayWriter] = None,
    ):
        """
        初始化 EAPRouter 实例

        Args:
            agents (list[AgentBase]): 智能体列表
            env_router (RouterBase): 环境路由器
            start_t (datetime): 仿真开始时间
            run_dir (Path, optional): 运行目录，用于存储回放数据。默认为 None。
            enable_replay (bool): 是否启用回放数据记录。默认为 True。
            replay_writer (ReplayWriter, optional): 已初始化的回放写入器。若提供，则不再在 init() 中创建；
                调用方需已将其传入 env_router 与 agents 的 __init__ 或 set_replay_writer。
        """
        self._env_router = env_router
        self._agents = agents
        self._t = start_t
        self._should_terminate: bool = False
        self._step_count: int = 0

        self._run_dir = run_dir
        self._enable_replay = enable_replay
        self._replay_writer: Optional[ReplayWriter] = replay_writer

        self._helper = EAPRouterHelper(
            env_router=self._env_router,
            agents=self._agents,
        )

    @property
    def current_time(self) -> datetime:
        """
        Get the current simulation time.

        Returns:
            The current datetime of the simulation.
        """
        return self._t

    @property
    def step_count(self) -> int:
        """
        Get the number of simulation steps that have been executed.

        Returns:
            The total number of steps executed.
        """
        return self._step_count

    async def init(self):
        await self._env_router.init(self._t)
        for agent in self._agents:
            await agent.init(env=self._env_router)

        # Replay writer: use provided one or create and inject when enabled
        if self._replay_writer is None and self._enable_replay and self._run_dir is not None:
            db_path = self._run_dir / "sqlite.db"
            self._replay_writer = ReplayWriter(db_path)
            await self._replay_writer.init()
            for agent in self._agents:
                agent.set_replay_writer(self._replay_writer)
                self._setup_agent_position_callback(agent)
            self._env_router.set_replay_writer(self._replay_writer)

        if self._replay_writer is not None:
            profiles = [
                (agent.id, agent.name, agent.get_profile())
                for agent in self._agents
            ]
            await self._replay_writer.write_agent_profiles_batch(profiles)
            for agent in self._agents:
                self._setup_agent_position_callback(agent)

    def _setup_agent_position_callback(self, agent: AgentBase) -> None:
        """Set up position callback for an agent to query from environment modules.

        Args:
            agent: The agent to set up the callback for.
        """
        async def get_position_from_env():
            # Try to get position from environment router
            return await self._env_router.get_agent_position(agent.id)

        agent._get_position_callback = get_position_from_env

    async def close(self):
        for agent in self._agents:
            await agent.close()
        await self._env_router.close()

        # Close replay writer
        if self._replay_writer is not None:
            await self._replay_writer.close()
            self._replay_writer = None

    # context manager
    async def __aenter__(self):
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def step(self, tick: int):
        """
        Run forward one step for all agents and the environment.
        The agents will step first, then the environment.

        Args:
            tick: The number of ticks of this simulation step.
        """
        self._t += timedelta(seconds=tick)
        tasks = []
        for agent in self._agents:
            tasks.append(agent.step(tick, self._t))
        await asyncio.gather(*tasks)
        await self._env_router.step(tick, self._t)
        self._step_count += 1

    async def run(self, num_steps: int, tick: int):
        """
        Run the simulation for a specified number of steps.

        Args:
            num_steps: The number of simulation steps to run.
            tick: The duration (in seconds) of each step.
        """
        for _ in range(num_steps):
            if self._should_terminate:
                break
            await self.step(tick)

    async def run_to(self, end_t: datetime, tick: int):
        """
        Run the simulation until the end time.

        Args:
            end_t: The end time of the simulation.
            tick: The duration (in seconds) of each step.
        """
        while self._t < end_t and not self._should_terminate:
            await self.step(tick)

    async def ask(self, question: str) -> str:
        """
        Ask the society a question.
        In the society, the question is answered by the agents and the environment.

        Args:
            question: The question to ask the society.

        Returns:
            The answer from the society.
        """
        return await self._helper.ask(question)

    async def intervene(self, instruction: str) -> str:
        """
        Intervene in the society.
        In the society, the intervention is executed by the agents and the environment.

        Args:
            instruction: The instruction to intervene in the society.

        Returns:
            The answer from the society.
        """
        return await self._helper.intervene(instruction)

    # ---- Dump & Load ----
    async def dump(self) -> dict:
        """
        Dump agents, environment router, and society variables into a dict.
        Excludes MCP environments in the router dump.
        """
        agents_dump: list[dict] = []
        for a in self._agents:
            try:
                agents_dump.append(
                    {
                        "class": a.__class__.__name__,
                        "id": a.id,
                        "dump": await a.dump(),
                    }
                )
            except Exception:
                continue

        return {
            "society": {
                "t": self._t.isoformat(),
            },
            "env_router": await self._env_router.dump(),
            "agents": agents_dump,
        }

    async def load(self, dump_data: dict):
        """
        Load society variables, agents dump by matching existing instances via id & class, and env router.
        """
        try:
            soc = dump_data.get("society") or {}
            t_str = soc.get("t")
            if isinstance(t_str, str) and len(t_str) > 0:
                from datetime import datetime as _dt

                self._t = _dt.fromisoformat(t_str)
        except Exception:
            pass

        # env router
        env_dump = dump_data.get("env_router") or {}
        if isinstance(env_dump, dict):
            try:
                await self._env_router.load(env_dump)
            except Exception:
                pass

        # agents: map by (class, id)
        by_key = {}
        for a in self._agents:
            by_key[(a.__class__.__name__, a.id)] = a
        for item in dump_data.get("agents", []) or []:
            try:
                key = (item.get("class"), item.get("id"))
                a = by_key.get(key)
                if a is None:
                    continue
                d = item.get("dump") or {}
                await a.load(d)
            except Exception:
                continue
