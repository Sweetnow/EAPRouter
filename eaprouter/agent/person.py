"""
PersonAgent - 基于SocietyAgent逻辑重新实现的Agent
采用eaprouter接口，移除Block抽象层，直接通过LLM调用和过程执行完成所有流程
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional
from enum import Enum
import inspect
import os
import asyncio
import json

from pydantic import BaseModel, Field
from litellm import AllMessageValues

from mem0 import AsyncMemory

from eaprouter.agent import AgentBase, DIALOG_TYPE_REFLECTION
from eaprouter.env import RouterBase
from eaprouter.config import Config
import logging

if TYPE_CHECKING:
    from eaprouter.storage import ReplayWriter


def _get_debug_info(description: str = "") -> str:
    """获取当前文件、行号和阶段描述的调试信息"""
    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_frame = frame.f_back
        filename = os.path.basename(caller_frame.f_code.co_filename)
        lineno = caller_frame.f_lineno
        return f"[{filename}:{lineno}] {description}"
    return description


# ==================== Pydantic Models ====================


class EmotionType(str, Enum):
    """情感类型枚举。"""

    JOY = "Joy"
    DISTRESS = "Distress"
    RESENTMENT = "Resentment"
    PITY = "Pity"
    HOPE = "Hope"
    FEAR = "Fear"
    SATISFACTION = "Satisfaction"
    RELIEF = "Relief"
    DISAPPOINTMENT = "Disappointment"
    PRIDE = "Pride"
    ADMIRATION = "Admiration"
    SHAME = "Shame"
    REPROACH = "Reproach"
    LIKING = "Liking"
    DISLIKING = "Disliking"
    GRATITUDE = "Gratitude"
    ANGER = "Anger"
    GRATIFICATION = "Gratification"
    REMORSE = "Remorse"
    LOVE = "Love"
    HATE = "Hate"


class Emotion(BaseModel):
    """情感强度模型（0-10）。"""

    sadness: int = Field(default=5, ge=0, le=10, description="悲伤强度")
    joy: int = Field(default=5, ge=0, le=10, description="快乐强度")
    fear: int = Field(default=5, ge=0, le=10, description="恐惧强度")
    disgust: int = Field(default=5, ge=0, le=10, description="厌恶强度")
    anger: int = Field(default=5, ge=0, le=10, description="愤怒强度")
    surprise: int = Field(default=5, ge=0, le=10, description="惊讶强度")


class Satisfactions(BaseModel):
    """需求满意度模型（0-1）。"""

    satiety: float = Field(default=0.7, ge=0.0, le=1.0, description="饱足感满意度")
    energy: float = Field(default=0.3, ge=0.0, le=1.0, description="能量满意度")
    safety: float = Field(default=0.9, ge=0.0, le=1.0, description="安全满意度")
    social: float = Field(default=0.8, ge=0.0, le=1.0, description="社交满意度")


NEED_DESCRIPTION = """## Understanding Human Needs

As a person, you have various **needs** that drive your behavior and decisions. Each need has an associated **satisfaction level** (ranging from 0.0 to 1.0), where lower values indicate less satisfaction and higher urgency. When a satisfaction level drops below a certain **threshold**, the corresponding need becomes urgent and should be addressed.

### Types of Needs

Your needs are organized by priority, with lower priority numbers indicating higher urgency:

#### 1. **Satiety** (Priority 1 - Highest)
- **Meaning**: The need to eat food to satisfy your hunger
- **When it becomes urgent**: When `satiety` drops below the threshold (typically at meal times)
- **Can interrupt other plans**: Yes - This is a basic survival need that can interrupt any ongoing activity

#### 2. **Energy** (Priority 2)
- **Meaning**: The need to rest, sleep or do some leisure or relaxing activities to recover your energy
- **When it becomes urgent**: When `energy` drops below the threshold (typically at night or after prolonged activity)
- **Can interrupt other plans**: Yes - Fatigue can make it difficult to continue other activities effectively

#### 3. **Safety** (Priority 3)
- **Meaning**: The need to maintain or improve your safety level, such as by working, moving to a safe place, or maintaining financial security
- **When it becomes urgent**: When `safety` drops below the threshold (often related to income, currency, or physical safety)
- **Can interrupt other plans**: No - Safety needs are important but typically don't require immediate interruption of ongoing activities

#### 4. **Social** (Priority 4)
- **Meaning**: The need to satisfy your social needs, such as going out with friends, chatting with friends, or maintaining social relationships
- **When it becomes urgent**: When `social` drops below the threshold (often when you have few friends or haven't socialized recently)
- **Can interrupt other plans**: No - Social needs are important for well-being but can usually be planned for

#### 5. **Whatever** (Priority 5 - Lowest)
- **Meaning**: You have no specific urgent needs right now and can do whatever you want
- **When it applies**: When all other needs are satisfied above their thresholds
- **Can interrupt other plans**: No - This is a passive state
"""

NeedType = Literal["satiety", "energy", "safety", "social", "whatever"]


class Need(BaseModel):
    """需求模型。"""

    need_type: NeedType = Field(description="需求类型")
    description: str = Field(description="需求描述")
    reasoning: str = Field(default="", description="选择理由")


class PlanStepStatus(str, Enum):
    """计划步骤状态枚举。"""

    PENDING = "pending"  # 待执行
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


class PlanStepEvaluation(BaseModel):
    """计划步骤评估结果。"""

    success: bool = Field(description="步骤是否成功")
    evaluation: str = Field(description="评估结果描述")
    consumed_time: int = Field(default=10, ge=0, description="消耗时间（分钟）")


class PlanStep(BaseModel):
    """计划步骤模型。"""

    intention: str = Field(description="步骤意图")
    status: PlanStepStatus = Field(
        default=PlanStepStatus.PENDING, description="步骤状态"
    )
    start_time: Optional[datetime] = Field(
        default=None, description="开始时间（datetime）"
    )
    evaluation: Optional[PlanStepEvaluation] = Field(
        default=None, description="评估结果"
    )


class Plan(BaseModel):
    """计划模型。"""

    target: str = Field(description="计划目标")
    reasoning: str = Field(default="", description="计划推理说明")
    steps: list[PlanStep] = Field(description="计划步骤列表")
    index: int = Field(default=0, ge=0, description="当前步骤索引")
    completed: bool = Field(default=False, description="是否完成")
    failed: bool = Field(default=False, description="是否失败")
    start_time: Optional[datetime] = Field(
        default=None, description="开始时间（datetime）"
    )
    end_time: Optional[datetime] = Field(
        default=None, description="结束时间（datetime）"
    )

    def to_adjust_needs_prompt(self) -> str:
        execution_results = []
        for step in self.steps:
            eval_result = step.evaluation
            if eval_result:
                execution_results.append(
                    f"Step: {step.intention}, Result: {eval_result.evaluation}"
                )

        evaluation_results = (
            "\n".join(execution_results)
            if execution_results
            else "No execution results"
        )
        return f"""Goal: {self.target}
Execution situation:
{evaluation_results}
"""


class CognitionUpdateResult(BaseModel):
    """认知更新结果模型。"""

    thought: str = Field(description="更新的思考")
    emotion: Emotion = Field(description="更新的情感强度")
    emotion_types: EmotionType = Field(description="情感类型")


class EmotionUpdateResult(BaseModel):
    """情感更新结果模型。"""

    emotion: Emotion = Field(description="更新的情感强度")
    emotion_types: EmotionType = Field(description="情感类型")
    conclusion: Optional[str] = Field(default=None, description="结论")


class NeedAdjustment(BaseModel):
    """需求调整项模型。"""

    need_type: NeedType = Field(description="需求类型")
    adjustment_type: Literal["increase", "decrease", "maintain"] = Field(
        description="调整类型：增加、减少、维持"
    )
    new_value: float = Field(ge=0.0, le=1.0, description="调整后的新值")
    reasoning: str = Field(default="", description="调整理由")


class NeedAdjustmentResult(BaseModel):
    """需求调整结果模型。"""

    adjustments: list[NeedAdjustment] = Field(description="调整列表")
    reasoning: str = Field(default="", description="整体调整理由")


class Intention(BaseModel):
    """意图模型。"""

    intention: str = Field(description="意图描述")
    priority: int = Field(ge=1, description="优先级（数字越小优先级越高）")
    attitude: float = Field(ge=0.0, le=1.0, description="个人偏好和评价")
    subjective_norm: float = Field(ge=0.0, le=1.0, description="社会环境和他人的看法")
    perceived_control: float = Field(ge=0.0, le=1.0, description="执行难度和可控性")
    reasoning: str = Field(default="", description="意图存在的理由")


class IntentionUpdate(BaseModel):
    """意图更新结果模型。"""

    intentions: list[Intention] = Field(description="候选意图列表（将选择优先级最高的一个）")
    reasoning: str = Field(default="", description="更新理由")


class CognitionIntentionUpdateResult(BaseModel):
    """合并的认知与意图更新结果模型。"""

    need_adjustment: NeedAdjustmentResult = Field(description="需求调整结果")
    current_need: Need = Field(description="当前需求结果")
    cognition_update: CognitionUpdateResult = Field(description="情感与思考更新结果")
    intention_update: IntentionUpdate = Field(description="意图更新结果")


class ReActInstructionResponse(BaseModel):
    """ReAct Act阶段的指令响应模型（非template模式）。"""

    reasoning: str = Field(
        default="",
        description="1.Why you are giving this action instruction. 2.check if the action instruction is available based on the available operations."
    )
    instruction: str = Field(
        description="A single, clear sentence or short paragraph that tells the router what action to take"
    )
    status: Optional[str] = Field(
        default=None,
        description="Optional status to skip environment call. If provided and not 'in_progress' or 'unknown', the environment router will be skipped. Valid values: 'success', 'fail', 'error', 'in_progress', 'unknown'"
    )


class ReActInstructionResponseWithTemplate(ReActInstructionResponse):
    """ReAct Act阶段的指令响应模型（template模式，含variables）。"""

    instruction: str = Field(
        description="A single, clear sentence or short paragraph that tells the router what action to take. This instruction MUST contain variable placeholders like {variable_name} if there are ANY variables in the instruction."
    )
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to substitute in the instruction template. This is a dictionary where keys are variable names (without curly braces) and values are their corresponding values. For example, if instruction is 'Move to {aoi_id}', variables should be {'aoi_id': 500001234}."
    )


class PersonAgent(AgentBase):
    """
    PersonAgent - 基于SocietyAgent逻辑重新实现的Agent

    执行流程（step()函数）：
    1. 使用<observe>对环境进行观察，作为最新的记忆
    2. 根据历史记忆，对需求（Need）进行调整，按调整的类型和列表返回，并且需要给出reasoning
    3. 根据历史记忆、当前Need满足度，更新Emotion和thought
    4. 根据历史记忆、当前Need需求满足度以及最紧迫的need、当前Emotion和thought，着重关注当前最紧迫的need，使用计划行为理论更新意图（从候选意图中选择优先级最高的一个）。
    5. 根据选中的意图以及与该意图相关的记忆，生成详细的执行计划 Plan，然后开始与环境交互以便完成这个Plan的每个Step。与环境交互的部分采用ReAct范式，对于一个Step最多与环境进行3次交互（参数，可以设置）。交互的过程将存入记忆中。环境交互的成功与否将在下一个循环时影响agent的需求和情绪
    """

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP agent module candidate list.
        Includes parameter descriptions.
        """
        description = f"""{cls.__name__}: A sophisticated agent with memory, needs, emotions, and planning capabilities.

**Description:** {cls.__doc__}

**Behavior:** On each step, the agent:
1. Observes the environment using <observe> and stores it as memory
2. Adjusts needs (satiety, energy, safety, social) based on historical memories
3. Updates emotions and thoughts based on memories and current need satisfaction
4. Updates intentions using Theory of Planned Behavior (TPB), focusing on the most urgent need
5. Generates detailed execution plans and interacts with the environment using ReAct paradigm to complete plan steps

**Initialization Parameters:**
- id (int): The unique identifier for the agent.
- profile (dict | Any): The profile of the agent. Can be a dictionary with agent attributes or any other type. Common profile fields include: name, gender, age, education, occupation, marriage_status, persona, background_story, profile_text.
- T_H (float, optional): Hunger threshold. Default: 0.2.
- T_D (float, optional): Energy threshold. Default: 0.2.
- T_P (float, optional): Safety threshold. Default: 0.2.
- T_C (float, optional): Social threshold. Default: 0.3.
- max_plan_steps (int, optional): Maximum number of plan steps. Default: 6.
- short_memory_window_size (int, optional): Short-term memory window size for storing recent N message records. Default: 10.
- max_intentions (int, optional): Maximum number of candidate intentions. Default: 5.
- max_react_interactions_per_step (int, optional): Maximum number of environment interactions per plan step using ReAct paradigm. Default: 3.
- template_mode_enabled (bool, optional): Enable template mode for environment interactions. When True, instructions sent to the environment router are treated as template instructions where variables from ctx['variables'] are substituted using {{variable_name}} syntax. The instruction MUST contain variable placeholders like {{variable_name}} if there are ANY variables in the instruction. Default: False.

**Example initialization config:**
```json
{{
  "id": 1,
  "profile": {{
    "name": "Alice",
    "gender": "female",
    "age": 30,
    "education": "University",
    "occupation": "Engineer",
    "marriage_status": "single",
    "persona": "helpful",
    "background_story": "A software engineer who loves coding."
  }},
  "T_H": 0.2,
  "T_D": 0.2,
  "T_P": 0.2,
  "T_C": 0.3,
  "max_plan_steps": 6,
  "short_memory_window_size": 10,
  "max_intentions": 5,
  "max_react_interactions_per_step": 3,
  "template_mode_enabled": false
}}
```

**Note:** The agent uses mem0 for memory management. Memory configuration is automatically retrieved from Config.get_mem0_config(str(id)). The agent maintains short-term memory, long-term memory, and cognitive memory for different purposes.
"""
        return description

    def __init__(
        self,
        id: int,
        profile: Any,
        name: Optional[str] = None,  # 可选显示名
        replay_writer: Optional["ReplayWriter"] = None,  # 可选回放写入器
        T_H: float = 0.2,  # 饥饿阈值
        T_D: float = 0.2,  # 能量阈值
        T_P: float = 0.2,  # 安全阈值
        T_C: float = 0.3,  # 社交阈值
        max_plan_steps: int = 6,
        short_memory_window_size: int = 10,  # 短期记忆窗口大小
        max_intentions: int = 5,  # 最大意图数量
        max_react_interactions_per_step: int = 3,  # 每个Step最多与环境交互次数
        template_mode_enabled: bool = False,  # 是否启用template模式
    ):
        """
        初始化PersonAgent。

        Args:
            id: Agent ID
            profile: Agent的profile（可以是包含profile_text的字典，或直接是字符串）
            memory_config: 兼容保留（不再使用，memory_config 统一从 Config 读取）
            world_description: 兼容保留（不再使用，world_description 统一从 env_router 获取）
            name: 可选显示名（默认从profile或id推断）
            replay_writer: 可选回放写入器
            T_H: 饥饿阈值
            T_D: 能量阈值
            T_P: 安全阈值
            T_C: 社交阈值
            max_plan_steps: 最大计划步骤数
            short_memory_window_size: 短期记忆窗口大小，用于存储最近的N条消息记录
            max_intentions: 最大意图数量
            max_react_interactions_per_step: 每个Step最多与环境交互次数
            template_mode_enabled: 是否启用template模式，启用后指令会被视为模板指令，支持{variable_name}语法
        """
        super().__init__(id=id, profile=profile, name=name, replay_writer=replay_writer)
        self._memory_user_id = f"agent-{id}"
        self._world_description = ""

        # memory_config 统一从 Config 获取（兼容入参但不使用）
        self._memory = AsyncMemory(config=Config.get_mem0_config(str(id)))

        # 阈值
        self.T_H = T_H
        self.T_D = T_D
        self.T_P = T_P
        self.T_C = T_C
        self.max_plan_steps = max_plan_steps
        self.short_memory_window_size = short_memory_window_size

        # Template模式开关
        self.template_mode_enabled = template_mode_enabled

        # 短期记忆队列（存储最近的N条消息记录，每个元素是包含记忆文本和时间戳的字典）
        self._short_memory: list[dict[str, str]] = []

        # 状态跟踪
        self._step_count = 0
        self._tick: Optional[int] = None
        self._t: Optional[datetime] = None
        self._observation_ctx: Optional[dict] = None
        self._observation: Optional[str] = None
        self._current_step_acts: list[dict] = []  # 当前步骤的acts记录

        # 需求满意度（0-1）
        self._satisfactions: Satisfactions = Satisfactions()

        # 当前需求
        self._need = None

        # 情感状态
        self._emotion: Emotion = Emotion()
        self._emotion_types: EmotionType = EmotionType.RELIEF

        # 思考
        self._thought: str = "Currently nothing good or bad is happening"

        # 合并认知/意图更新的结构化输出
        self._last_cognition_intention_update: Optional[
            CognitionIntentionUpdateResult
        ] = None

        self._plan: Optional[Plan] = None
        """
        Current plan to be executed.
        """

        # 意图（LLM返回后只保留优先级最高的一个）
        self._intention: Intention | None = None
        self.max_intentions = max_intentions
        self.max_react_interactions_per_step = max_react_interactions_per_step

        # 用于保护记忆操作的异步锁
        self._memory_lock = asyncio.Lock()

        # 意图历史记录（每两步记录一次）
        self._intention_history: list[dict] = []

        # 时间步详细记录（每个时间步的intention、plan、act和需求状态）
        self._step_records: list[dict] = []

        # 步骤记录文件路径（在第一次记录时初始化）
        self._step_records_file: Optional[str] = None

        # 当前step的认知记忆列表（内在状态更新，不立即加入mem0）
        self._cognition_memory: list[dict[str, Any]] = []

        self._logger.info(f"✓ PersonAgent (ID: {id}) 已创建")

    # ==================== 重试辅助方法 ====================

    async def _retry_async_operation(
        self,
        operation_name: str,
        async_func,
        *args,
        max_retries: int = 3,
        **kwargs
    ):
        """通用异步操作重试机制。
        
        Args:
            operation_name: 操作名称（用于日志）
            async_func: 要执行的异步函数
            *args: 函数的位置参数
            max_retries: 最大重试次数，默认3次
            **kwargs: 函数的关键字参数
            
        Returns:
            函数的返回值，或在所有重试失败后返回None
        """
        for attempt in range(max_retries):
            try:
                return await async_func(*args, **kwargs)
            except Exception as e:
                self._logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    self._logger.warning(
                        f"Skipping {operation_name} after {max_retries} retries"
                    )
                    return None

    def _retry_sync_operation(
        self,
        operation_name: str,
        sync_func,
        *args,
        max_retries: int = 3,
        **kwargs
    ):
        """通用同步操作重试机制。
        
        Args:
            operation_name: 操作名称（用于日志）
            sync_func: 要执行的同步函数
            *args: 函数的位置参数
            max_retries: 最大重试次数，默认3次
            **kwargs: 函数的关键字参数
            
        Returns:
            函数的返回值，或在所有重试失败后返回None
        """
        for attempt in range(max_retries):
            try:
                return sync_func(*args, **kwargs)
            except Exception as e:
                self._logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    self._logger.warning(
                        f"Skipping {operation_name} after {max_retries} retries"
                    )
                    return None

    @property
    def profile_text(self) -> str:
        """获取 profile 文本。"""
        if isinstance(self._profile, str):
            return self._profile
        elif isinstance(self._profile, dict):
            return self._profile.get("profile_text", str(self._profile))
        else:
            return str(self._profile)

    @property
    def name(self) -> str:
        """获取 Agent 的名称（由基类 __init__ 的 name 或 profile 推导）。"""
        return self._name

    def get_profile(self) -> dict[str, Any]:
        """获取 Agent 的 profile 字典。

        Returns:
            包含 Agent 所有属性的字典。
        """
        if isinstance(self._profile, dict):
            return self._profile.copy()
        elif hasattr(self._profile, "model_dump"):
            return self._profile.model_dump()
        else:
            return {"profile_text": str(self._profile)}

    @property
    def memory(self) -> AsyncMemory:
        """获取 memory 实例。"""
        return self._memory

    # ==================== 记忆管理方法 ====================

    async def _add_memory_with_timestamp(
        self,
        memory_text: str,
        metadata: Optional[dict] = None,
        t: Optional[datetime] = None,
    ) -> None:
        """添加带时间戳的记忆（带3次重试机制）。

        Args:
            memory_text: 记忆文本
            metadata: 记忆元数据，如果为None，则使用默认值
            t: 时间，如果为None，则使用当前时间
        """
        async with self._memory_lock:
            if metadata is None:
                metadata = {}

            if t is None:
                assert self._t is not None, "t is not set"
                t = self._t

            current_time = t.isoformat()
            metadata["timestamp"] = current_time

            # 3次重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self.memory.add(
                        memory_text,
                        user_id=self._memory_user_id,
                        metadata=metadata,
                        infer=False,
                    )
                    # 成功后添加到短期记忆
                    self._short_memory.insert(
                        0, {"content": memory_text, "timestamp": current_time}
                    )
                    if len(self._short_memory) > self.short_memory_window_size:
                        self._short_memory.pop()
                    return
                except Exception as e:
                    self._logger.warning(
                        f"Failed to add memory (attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    if attempt == max_retries - 1:
                        self._logger.warning(
                            f"Skipping memory after {max_retries} retries: {memory_text[:100]}..."
                        )
                        return

    async def _get_recent_memories(self, limit: int = 5) -> list[dict[str, str]]:
        """获取最近的记忆。

        Args:
            limit: 返回的记忆数量限制

        Returns:
            最近的记忆列表，每个元素是包含 'content' 和 'timestamp' 的字典
        """
        async with self._memory_lock:
            return self._short_memory[:limit]

    def _add_cognition_memory(
        self,
        memory_text: str,
        memory_type: str = "cognition",
        metadata: Optional[dict] = None,
    ) -> None:
        """添加认知记忆到当前step的cognition_memory列表（不立即加入mem0）。
        
        此方法内置重试机制，最多3次重试。如果都失败则跳过。

        Args:
            memory_text: 记忆文本
            memory_type: 记忆类型（如 "cognition", "emotion", "need", "intention", "plan"）
            metadata: 额外的元数据
        """
        if metadata is None:
            metadata = {}
        
        # 3次重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._cognition_memory.append({
                    "content": memory_text,
                    "type": memory_type,
                    "metadata": metadata,
                })
                return
            except Exception as e:
                self._logger.warning(
                    f"Failed to add cognition memory (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    self._logger.warning(
                        f"Skipping cognition memory after {max_retries} retries: {memory_text[:100]}..."
                    )

    async def _flush_cognition_memory_to_memory(self) -> None:
        """将当前step的cognition_memory统一格式化为结构化文本并一次性添加到memory。
        
        这个方法应该在step结束时调用，将所有认知记忆合并为一个结构化的记忆条目。
        包含3次重试机制，如果添加失败3次则跳过。
        """
        if not self._cognition_memory:
            return

        assert self._t is not None, "t is not set"
        t = self._t

        # 按类型分组组织记忆
        memory_by_type: dict[str, list[str]] = {}
        for mem in self._cognition_memory:
            mem_type = mem.get("type", "cognition")
            content = mem.get("content", "")
            if mem_type not in memory_by_type:
                memory_by_type[mem_type] = []
            memory_by_type[mem_type].append(content)

        # 构建结构化的记忆文本
        structured_parts = []
        for mem_type, contents in memory_by_type.items():
            if contents:
                structured_parts.append(f"## {mem_type.upper()}\n" + "\n".join(f"- {c}" for c in contents))

        if structured_parts:
            structured_memory_text = "\n\n".join(structured_parts)
            
            # 一次性添加到memory（已包含3次重试机制）
            await self._add_memory_with_timestamp(
                structured_memory_text,
                metadata={"type": "cognition"},
                t=t,
            )

        # 清空当前step的认知记忆
        self._cognition_memory.clear()

    # ==================== 需求管理方法 ====================

    async def _adjust_needs_from_memory(self) -> NeedAdjustmentResult:
        """根据历史记忆调整需求满意度。

        Returns:
            NeedAdjustmentResult: 包含调整列表和整体理由的结果
        """
        assert self._tick is not None and self._t is not None, "tick and t are not set"
        t = self._t

        # 获取agent状态
        state_text = await self.get_state()

        # 获取当前observation
        current_observation_text = (
            self._observation if self._observation else "No current observation."
        )

        prompt = f"""{state_text}

<task>
Based on your historical memories, current observation, and current situation, adjust your need satisfaction levels.
</task>

<current_observation>
Here is the current observation from the environment in this time step. You should use this information to help you adjust your needs.
{current_observation_text}
</current_observation>

<explanation>
{NEED_DESCRIPTION}
</explanation>

<instructions>
Based on your recent memories, current observation, and experiences, determine if any need satisfaction levels should be adjusted:
1. Consider what happened in your recent memories (e.g., did you eat? did you rest? did you socialize?)
2. Consider what you observed in the current environment (e.g., available food, current location, time of day, etc.)
3. For each need type, decide if it should be increased, decreased, or maintained
4. Provide specific reasoning for each adjustment based on your memories and observation
5. Only adjust needs that have been affected by recent events or current observation
6. DO NOT change the the word of the need type in the adjustments.

Adjustment types:
- "increase": The need satisfaction should increase (e.g., after eating, satiety increases)
- "decrease": The need satisfaction should decrease (e.g., after a tiring activity, energy decreases)
- "maintain": The need satisfaction should stay the same (e.g., no relevant events occurred)
</instructions>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "adjustments": [
        {{
            "need_type": "satiety" | "energy" | "safety" | "social",
            "adjustment_type": "increase" | "decrease" | "maintain",
            "new_value": 0.0-1.0,
            "reasoning": "Why this adjustment is needed based on memories"
        }},
        ...
    ],
    "reasoning": "Overall reasoning for all adjustments"
}}
```
</format>

Your response is:
```json
"""

        # self._logger.debug(
        #     f"{_get_debug_info('准备调用LLM调整需求满意度')} - prompt length: {len(prompt)}"
        # )
        # self._logger.debug(
        #     f"{_get_debug_info('发送给LLM的完整prompt（调整需求）')}:\n{prompt}"
        # )

        result = await self.acompletion_with_pydantic_validation(
            model_type=NeedAdjustmentResult,
            messages=[{"role": "user", "content": prompt}],
            tick=self._tick,
            t=t,
        )

        self._logger.debug(
            f"{_get_debug_info('收到LLM响应（调整需求）')} - adjustments: {result.adjustments}, reasoning: {result.reasoning[:100]}..."
        )

        # 应用调整
        for adjustment in result.adjustments:
            if adjustment.need_type == "satiety":
                self._satisfactions.satiety = adjustment.new_value
            elif adjustment.need_type == "energy":
                self._satisfactions.energy = adjustment.new_value
            elif adjustment.need_type == "safety":
                self._satisfactions.safety = adjustment.new_value
            elif adjustment.need_type == "social":
                self._satisfactions.social = adjustment.new_value

        # 记录调整记忆到cognition_memory（不立即加入mem0）
        self._add_cognition_memory(
            f"Adjusted needs based on memories: {result.reasoning}",
            memory_type="need",
        )

        return result

    async def _determine_current_need(self):
        """确定当前需求。"""
        assert self._t is not None and self._tick is not None, "t and tick are not set"
        t = self._t

        # 获取当前满意度值
        satiety = self._satisfactions.satiety
        energy = self._satisfactions.energy
        safety = self._satisfactions.safety
        social = self._satisfactions.social

        # 计算是否低于阈值
        satiety_below_threshold = satiety <= self.T_H
        energy_below_threshold = energy <= self.T_D
        safety_below_threshold = safety <= self.T_P
        social_below_threshold = social <= self.T_C

        # 构建当前满意度状态信息
        current_satisfaction_info = f"""Current Need Satisfaction Status:
- satiety: {satiety:.2f} (Threshold: {self.T_H}, Below Threshold: {satiety_below_threshold})
- energy: {energy:.2f} (Threshold: {self.T_D}, Below Threshold: {energy_below_threshold})
- safety: {safety:.2f} (Threshold: {self.T_P}, Below Threshold: {safety_below_threshold})
- social: {social:.2f} (Threshold: {self.T_C}, Below Threshold: {social_below_threshold})"""

        # 获取当前observation
        current_observation_text = (
            self._observation if self._observation else "No current observation."
        )

        prompt = f"""<task>
Determine your current need based on your profile, needs satisfaction levels, current observation, and current situation.
</task>

<profile>
{self.profile_text}
</profile>

<current_observation>
Here is the current observation from the environment in this time step. You should use this information to help you determine your current need.
{current_observation_text}
</current_observation>

<explanation>
{NEED_DESCRIPTION}
</explanation>

<decision_rules>
When determining your current need, follow these principles:
1. **Priority matters**: Lower priority numbers mean higher urgency. Always consider needs in priority order (1 → 2 → 3 → 4 → 5).
2. **Urgency threshold**: A need is considered urgent when its satisfaction value is **below or equal to** its threshold. Only urgent needs should be prioritized. 
3. **Default state**: If no needs are urgent (all satisfaction levels are above their thresholds), return "whatever" to indicate you have no specific urgent needs.

Remember: Your needs reflect your current state of well-being. Pay attention to satisfaction levels and thresholds to make appropriate decisions about what you need most right now.
</decision_rules>

<current_satisfaction_status>
{current_satisfaction_info}
</current_satisfaction_status>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "reasoning": "Brief explanation of your decision",
    "need_type": "satiety" | "energy" | "safety" | "social" | "whatever",
    "description": "A brief description of why you chose this need (e.g., 'I feel hungry' or 'I have no specific needs right now')"
}}
```
</format>

Your response is:
```json
"""

        # self._logger.debug(
        #     f"{_get_debug_info('准备调用LLM确定当前需求')} - prompt length: {len(prompt)}"
        # )
        # self._logger.debug(
        #     f"{_get_debug_info('发送给LLM的完整prompt（确定需求）')}:\n{prompt}"
        # )

        response = await self.acompletion_with_pydantic_validation(
            model_type=Need,
            messages=[{"role": "user", "content": prompt}],
            tick=self._tick,
            t=t,
        )

        self._logger.debug(
            f"{_get_debug_info('收到LLM响应（确定需求）')} - need_type: {response.need_type}, description: {response.description[:100]}..."
        )

        self._need = response.need_type

    # ==================== 计划生成方法 ====================

    async def _generate_plan_from_intention(self, intention: Intention) -> None:
        """根据选中的意图生成详细的执行计划。

        Args:
            intention: 选中的意图
        """
        assert self._tick is not None and self._t is not None, "tick and t are not set"
        assert self._observation is not None, "observation is not set"
        t = self._t

        if self._plan and not self._plan.completed and not self._plan.failed:
            return None

        # 获取与该意图相关的记忆（带3次重试机制）
        related_memories = None
        async with self._memory_lock:
            related_memories = await self._retry_async_operation(
                "search memories for intention",
                self.memory.search,
                intention.intention,
                user_id=self._memory_user_id,
                limit=10
            )

        related_memories_text = ""
        if related_memories and "results" in related_memories:
            memory_items = []
            for result in related_memories["results"]:
                memory_items.append(f"<memory>\n{result['memory']}\n</memory>")
            related_memories_text = "\n".join(memory_items)
        else:
            related_memories_text = "No related memories found."

        # 获取agent状态
        state_text = await self.get_state()

        # 在 plan 阶段，将 observation 单独提取出来
        current_observation_text = (
            self._observation if self._observation else "No current observation."
        )

        # 生成详细计划
        plan_prompt = f"""{state_text}

<task>
Generate specific execution steps based on the selected intention.
</task>

<intention>
Intention: {intention.intention}
Priority: {intention.priority}
TPB Evaluation:
- Attitude: {intention.attitude:.2f}
- Subjective Norm: {intention.subjective_norm:.2f}
- Perceived Control: {intention.perceived_control:.2f}
Reasoning: {intention.reasoning}
</intention>

<current_observation>
Here are some basic information observed from the environment in this time step. You should use this information to help you generate plans.
If the observation provided related information, you do not need to plan query steps to get the information.
{current_observation_text}
</current_observation>

<related_memories>
{related_memories_text}
</related_memories>

<explanation>
1. Each execution step should have `intention` field, which is a clear and concise description of what to do
2. `steps` list should only include steps necessary to fulfill the intention (limited to {self.max_plan_steps} steps)
3. Consider related memories when planning steps
4. Steps should be actionable and realistic
5. No need to plan too many query steps because an overall observation will be provided first for each step.
</explanation>

<IMPORTANT>
You are operating in the simulated world described above. Please do not attempt any actions that are infeasible according to the AVAILABLE ACTIONS in the environment.
You're actions are limited by the AVAILABLE ACTIONS in the environment, it can only be a single or a combination of the AVAILABLE ACTIONS.
Ensure that the level of detail in your actions corresponds precisely to the allowed action space.
</IMPORTANT>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "target": "{intention.intention}",
    "reasoning": "Consider: 1.Why you are giving this plan. 2.The plan MUST be supported by the available operations in the environment,check it carefully.",
    "steps": [
        {{
            "intention": "Step 1 description"
        }},
        {{
            "intention": "Step 2 description"
        }},
        ...
    ]
}}
```
</format>

Your response is:
```json
"""

        # self._logger.debug(
        #     f"{_get_debug_info('准备调用LLM生成计划')} - intention: {intention.intention}, prompt length: {len(plan_prompt)}"
        # )
        # self._logger.debug(
        #     f"{_get_debug_info('发送给LLM的完整prompt（生成计划）')}:\n{plan_prompt}"
        # )

        plan = await self.acompletion_with_pydantic_validation(
            model_type=Plan,
            messages=[{"role": "user", "content": plan_prompt}],
            tick=self._tick,
            t=t,
        )

        self._logger.debug(
            f"{_get_debug_info('收到LLM响应（生成计划）')} - target: {plan.target}, reasoning: {plan.reasoning}, steps: {plan.steps}"
        )

        # 使用响应中的计划数据，设置运行时字段
        plan.index = 0
        plan.completed = False
        plan.failed = False
        plan.start_time = t
        # 确保所有步骤的状态初始化为PENDING
        for step in plan.steps:
            step.status = PlanStepStatus.PENDING
        self._plan = plan

        # 记录计划生成记忆到cognition_memory（不立即加入mem0）
        plan_steps_text = "\n".join([f"{i+1}. {step.intention}" for i, step in enumerate(plan.steps)])
        self._add_cognition_memory(
            f"Generated plan for intention: {intention.intention}\nPlan steps:\n{plan_steps_text}\nReasoning: {plan.reasoning}",
            memory_type="plan",
        )

    # ==================== 计划相关方法 ====================

    async def _check_step_completion(self, step: PlanStep) -> PlanStepStatus:
        """根据observation和历史记忆，确定当前step的完成情况。

        Args:
            step: 要检查的计划步骤

        Returns:
            PlanStepStatus: 步骤的状态
        """
        assert self._tick is not None and self._t is not None, "tick and t are not set"
        assert self._observation is not None, "observation is not set"

        # 获取与该步骤相关的记忆（带3次重试机制）
        related_memories = None
        async with self._memory_lock:
            related_memories = await self._retry_async_operation(
                "search memories for step completion",
                self.memory.search,
                step.intention,
                user_id=self._memory_user_id,
                limit=5
            )

        related_memories_text = ""
        if related_memories and "results" in related_memories:
            memory_items = []
            for result in related_memories["results"]:
                memory_items.append(f"<memory>\n{result['memory']}\n</memory>")
            related_memories_text = "\n".join(memory_items)
        else:
            related_memories_text = "No related memories found."

        # 获取agent状态
        state_text = await self.get_state()

        prompt = f"""{state_text}

<task>
Determine the completion status of a plan step based on the current observation and historical memories.
</task>

<step_info>
Intention: {step.intention}
Status: {step.status.value}
Start time: {step.start_time.isoformat() if step.start_time else 'Not started'}
</step_info>

<related_memories>
{related_memories_text}
</related_memories>

<instructions>
Based on the observation and memories, determine if this step is:
1. **completed**: The step has been successfully completed (the intention has been fulfilled)
2. **in_progress**: The step is currently being executed but not yet complete
    - If the environment indicates that the step cannot be supported, return "failed" instead of "in_progress"
3. **failed**: The step has failed and cannot be completed (especially if you observe repeated unsuccessful attempts in the memories)
4. **pending**: The step has not been started yet (only if start_time is None)

Consider:
- What the step intention is trying to achieve
- What has been observed in the environment
- What actions have been taken according to memories
- Whether the goal of the step has been achieved
- If the memories show multiple failed attempts at the same goal with no progress, strongly consider returning "failed" to avoid infinite loops

Return only one of: "completed", "in_progress", "failed", "pending"
</instructions>

Your response (one word only):"""

        # self._logger.debug(
        #     f"{_get_debug_info('准备调用LLM检查步骤完成情况')} - step intention: {step.intention}, prompt length: {len(prompt)}"
        # )
        # self._logger.debug(
        #     f"{_get_debug_info('发送给LLM的完整prompt（检查步骤完成）')}:\n{prompt}"
        # )

        response = await self.acompletion(
            [{"role": "user", "content": prompt}],
            stream=False,
        )
        content = response.choices[0].message.content  # type: ignore
        status_text = str(content).strip().lower() if content else "pending"

        self._logger.debug(
            f"{_get_debug_info('收到LLM响应（检查步骤完成）')} - raw response: {content}, parsed status: {status_text}"
        )

        # 映射到枚举值
        status_map = {
            "completed": PlanStepStatus.COMPLETED,
            "in_progress": PlanStepStatus.IN_PROGRESS,
            "failed": PlanStepStatus.FAILED,
            "pending": PlanStepStatus.PENDING,
        }

        return status_map.get(status_text, PlanStepStatus.PENDING)

    async def _should_interrupt_plan(self) -> bool:
        """结合最新的意图，判定是否要中断计划。

        Returns:
            bool: True表示应该中断当前计划，False表示继续执行
        """
        assert self._tick is not None and self._t is not None, "tick and t are not set"
        assert self._need is not None, "need is not set"

        if not self._plan or self._plan.completed or self._plan.failed:
            return False

        if not self._intention:
            return False

        # 获取agent状态
        state_text = await self.get_state()

        prompt = f"""{state_text}

<task>
Determine if the current plan should be interrupted based on the latest intention.
</task>

<instructions>
Consider:
1. Is the latest intention significantly different from the current plan target?
2. Is the latest intention more urgent or important than completing the current plan?
3. Should the current plan be interrupted to pursue the new intention?

Return "yes" if the plan should be interrupted, "no" if it should continue.
</instructions>

Your response (yes/no only):"""

        # self._logger.debug(
        #     f"{_get_debug_info('准备调用LLM判断是否中断计划')} - prompt length: {len(prompt)}"
        # )
        # self._logger.debug(
        #     f"{_get_debug_info('发送给LLM的完整prompt（判断中断计划）')}:\n{prompt}"
        # )

        response = await self.acompletion(
            [{"role": "user", "content": prompt}],
            stream=False,
        )
        content = response.choices[0].message.content  # type: ignore
        answer = str(content).strip().lower() if content else "no"

        self._logger.debug(
            f"{_get_debug_info('收到LLM响应（判断中断计划）')} - raw response: {content}, parsed answer: {answer}, should_interrupt: {answer.startswith('yes')}"
        )

        return answer.startswith("yes")

    # ==================== 步骤执行方法 ====================

    def _format_interaction_history(self, interaction_history: list[str]) -> str:
        """将交互历史格式化为对话形式。

        Args:
            interaction_history: 包含 "Action: ..." 和 "Observation: ..." 的列表

        Returns:
            格式化后的对话字符串，如果没有历史则返回"No previous interactions."
        """
        if not interaction_history:
            return "No previous interactions."

        dialog_lines = []
        for i in range(0, len(interaction_history), 2):
            # 每两条为一组：Action 和 Observation
            if i < len(interaction_history):
                action = interaction_history[i]
                if action.startswith("Action: "):
                    agent_action = action.replace("Action: ", "", 1)
                    dialog_lines.append(f"Agent: {agent_action}")

            if i + 1 < len(interaction_history):
                observation = interaction_history[i + 1]
                if observation.startswith("Observation: "):
                    env_response = observation.replace("Observation: ", "", 1)
                    dialog_lines.append(f"Environment: {env_response}")

        return "\n".join(dialog_lines)

    async def _step_execution(self) -> tuple[PlanStepStatus, list[dict]]:
        """执行当前步骤，采用ReAct范式，使用LLM原生多轮对话。

        ReAct范式：对于每个Step，最多与环境进行max_react_interactions_per_step次交互。
        使用多轮对话方式，observation作为第一轮对话的响应，后续交互作为对话历史。

        Returns:
            tuple[PlanStepStatus, list[dict]]: 步骤执行后的状态（已完成、进行中、失败）和acts记录列表
        """
        assert self._tick is not None and self._t is not None, "tick and t are not set"
        # 当前步骤的acts记录
        step_acts = []

        current_plan = self._plan
        if not current_plan:
            return PlanStepStatus.PENDING, []

        steps = current_plan.steps
        step_index = current_plan.index

        if step_index >= len(steps):
            return PlanStepStatus.PENDING, []

        current_step = steps[step_index]
        intention = current_step.intention

        if not intention:
            return PlanStepStatus.PENDING, []

        # 如果步骤状态是进行中，先检查完成情况
        if current_step.status == PlanStepStatus.IN_PROGRESS:
            status = await self._check_step_completion(current_step)
            current_step.status = status

            if status == PlanStepStatus.COMPLETED:
                # 步骤已完成，移动到下一步
                if step_index + 1 < len(steps):
                    current_plan.index = step_index + 1
                else:
                    current_plan.completed = True
                    current_plan.end_time = self._t
                    await self._emotion_update_for_plan(current_plan, completed=True)
                self._plan = current_plan
                return PlanStepStatus.COMPLETED, []
            elif status == PlanStepStatus.FAILED:
                current_step.status = PlanStepStatus.FAILED
                current_plan.failed = True
                current_plan.end_time = self._t
                await self._emotion_update_for_plan(current_plan, completed=False)
                self._plan = current_plan
                return PlanStepStatus.FAILED, []
            else:
                # 仍在进行中，返回进行中状态
                self._plan = current_plan
                return PlanStepStatus.IN_PROGRESS, []

        # 记录步骤开始时间
        if current_step.start_time is None:
            current_step.start_time = self._t
            current_step.status = PlanStepStatus.IN_PROGRESS

        # 获取与当前步骤相关的记忆（带3次重试机制）
        related_memories = None
        async with self._memory_lock:
            related_memories = await self._retry_async_operation(
                "search memories for step execution",
                self.memory.search,
                intention,
                user_id=self._memory_user_id,
                limit=5
            )

        related_memories_text = ""
        if related_memories and "results" in related_memories:
            memory_items = []
            for result in related_memories["results"]:
                memory_items.append(f"<memory>\n{result['memory']}\n</memory>")
            related_memories_text = "\n".join(memory_items)
        else:
            related_memories_text = "No related memories found."

        # 获取agent状态（在循环外获取一次，避免重复调用）
        state_text = await self.get_state()

        # 初始化ctx，包含id、observation、returns三个字段
        ctx = {
            "id": self._id,
            "observation": self._observation_ctx if self._observation_ctx else "",
            "returns": [],  # 存储之前交互的返回context
        }

        # 构建初始user消息（包含任务描述、状态、相关记忆等）
        current_observation_text = (
            self._observation if self._observation else "No current observation."
        )

        initial_user_message = f"""{state_text}

<task>
You need to complete the following plan step by interacting with the environment.

Plan step intention: "{intention}"

<related_memories>
{related_memories_text}
</related_memories>

<CRITICAL: AVOID REDUNDANT QUERIES>
1. **Check observation first**: Before generating any instruction, carefully review the observation provided. If it already contains the information you need, DO NOT query the environment again for the same information.

2. **Check previous interactions**: Review the conversation history. DO NOT repeat the same query or action that was already attempted. If a previous interaction already provided the needed information, use that information instead of querying again.

3. **Energy conservation**: Interacting with the environment consumes energy (like a real person). Only query the environment when absolutely necessary. If you already know the answer from observation or previous interactions, proceed with the action directly.

4. **Avoid duplicate actions**: If you have already attempted an action in a previous interaction, do not repeat it unless there is a clear reason (e.g., the previous attempt failed and you need to retry with modifications).

5. **Use available information**: Always prioritize using information from:
   - Current observation (highest priority)
   - Previous interactions in this conversation
   - Related memories
   Only query the environment as a last resort when the information is truly needed and not available elsewhere.
</CRITICAL: AVOID REDUNDANT QUERIES>

<IMPORTANT>
You are operating in the simulated world described above. Please do NOT attempt any actions that are infeasible according to the AVAILABLE ACTIONS in the environment.
Your actions are limited by the AVAILABLE ACTIONS in the environment, it can only be a single or a combination of the AVAILABLE ACTIONS.
Ensure that the level of detail in your actions corresponds precisely to the allowed action space.
</IMPORTANT>

Now, I will provide you with the initial observation from the environment. Based on this observation and the task above, you should generate clear and actionable instructions to complete the step. The instruction should be a single, clear sentence or short paragraph that tells the environment router what action to take. Do not tear the core action into over-detailed steps.
</task>"""

        # 初始化多轮对话消息列表
        messages: list[AllMessageValues] = [
            {"role": "user", "content": initial_user_message}
        ]

        # 添加assistant消息，表示调用observe工具获取初始observation
        messages.append({
            "role": "assistant",
            "content": "I need to observe the environment first.",
            "tool_calls": [{
                "id": "observe_initial",
                "type": "function",
                "function": {
                    "name": "env_router",
                    "arguments": json.dumps({"instruction": "<observe>"})
                }
            }]
        })

        # 将observation作为第一轮tool响应（环境响应）
        observation_message = current_observation_text if current_observation_text else "No observation available."
        messages.append({
            "role": "tool",
            "content": observation_message,
            "tool_call_id": "observe_initial",
        })

        final_answer = ""
        final_status = "unknown"
        interaction_count = 0  # 记录实际交互次数

        # ReAct循环：最多进行max_react_interactions_per_step次交互
        for interaction_num in range(self.max_react_interactions_per_step):
            # 添加user消息，请求生成instruction
            # template_mode_enabled=false时，prompt和返回值类型不包含template相关描述和字段
            template_mode_section = ""
            if self.template_mode_enabled:
                template_mode_section = """
<template_mode enabled>
**MANDATORY**: When you have variables to pass (e.g., location, item, amount), you MUST use placeholder form {{variable_name}} in the instruction. Plain text with concrete values instead of placeholders will break template caching and is NOT allowed.

Good cases (placeholders in instruction, variables dict with values):
- instruction: "Move to {{location}}", variables: {{"location": "home"}}
- instruction: "Buy {{item}} for {{price}} dollars", variables: {{"item": "apple", "price": 5}}
- instruction: "Send {{amount}} to {{target}}", variables: {{"amount": "100", "target": "Alice"}}

Bad cases (DO NOT do this):
- instruction: "Move to home", variables: {{"location": "home"}}  # BAD: use {{location}} instead of "home"
- instruction: "Buy apple for 5 dollars", variables: {{"item": "apple", "price": 5}}  # BAD: use placeholders {{item}}, {{price}}
- instruction: "Send 100 to Alice", variables: {{"amount": "100", "target": "Alice"}}  # BAD: instruction must use {{amount}}, {{target}}

Only extract variables that are actually used in the instruction (must appear as {{variable_name}} in the instruction).
Common variables: location, target, amount, item, reason, etc.
</template_mode>

"""
            if self.template_mode_enabled:
                json_format = """```json
{{
    "reasoning": "1.Why you are giving this action instruction. 2.check if the action instruction is available based on the available operations.",
    "instruction": "A single, clear sentence or short paragraph that tells the router what action to take. Can contain {{variable_name}} placeholders.",
    "variables": {{"variable_name": "value", ...}} (optional, include variables if instruction contains {{variable_name}} placeholders),
    "status": "success" | "fail" | "error" | "in_progress" | "unknown" | null (optional, only provide if you can determine the step status without calling the environment)
}}
```"""
            else:
                json_format = """```json
{{
    "reasoning": "1.Why you are giving this action instruction. 2.check if the action instruction is available based on the available operations.",
    "instruction": "A single, clear sentence or short paragraph that tells the router what action to take",
    "status": "success" | "fail" | "error" | "in_progress" | "unknown" | null (optional, only provide if you can determine the step status without calling the environment)
}}
```"""
            user_message_content = f"""Based on the conversation history above, generate a clear and actionable instruction for the environment router to complete the step.

Interaction {interaction_num + 1}/{self.max_react_interactions_per_step}

Generate a clear instruction that:
1. Clearly states what you want to do to complete the step
2. Is specific and actionable according to the available operations
3. Can be understood by a code generation router that will call environment module tools
4. References the step intention: "{intention}"
5. You can use continue to do something if you think the step is not complete.
6. **DO NOT query for information that is already available in the observation or previous interactions**
{template_mode_section}<optional_status>
If you can determine the step status directly from the conversation history (e.g., the step is already completed, or it's impossible to complete), you can provide a "status" field to skip the environment call:
- "success": The step has been completed successfully (skip environment call)
- "fail": The step has failed and cannot be completed (skip environment call)
- "error": An error occurred (skip environment call)
- "in_progress" or omit: Continue with environment call
- "unknown" or omit: Continue with environment call
</optional_status>

<format>
You should return the result in JSON format with the following structure:
{json_format}
</format>
Your instruction:"""

            messages.append({"role": "user", "content": user_message_content})

            self._logger.debug(
                f"{_get_debug_info('ReAct Act阶段')} - interaction {interaction_num + 1}/{self.max_react_interactions_per_step}, messages count: {len(messages)}"
            )

            try:
                response_model = ReActInstructionResponseWithTemplate if self.template_mode_enabled else ReActInstructionResponse
                response = await self.acompletion_with_pydantic_validation(
                    model_type=response_model,
                    messages=messages,
                    tick=self._tick,
                    t=self._t,
                )
                instruction = response.instruction.strip()
                reasoning = response.reasoning.strip()
                llm_status = response.status
                variables = getattr(response, "variables", {})  # 仅template模式有variables字段
            except Exception as e:
                self._logger.warning(
                f"{_get_debug_info('ReAct Act阶段Pydantic验证失败')} - {str(e)}, 使用默认指令"
            )
                instruction = ""
                reasoning = ""
                llm_status = None
                variables = {}  # 异常时使用空字典

            if not instruction:
                instruction = (
                    f"I want to: {intention}. Please help me execute this step."
                )

            # 生成tool_call_id（在添加assistant消息之前生成，以便在tool_calls中使用）
            tool_call_id = f"call_{interaction_num}_{interaction_count}"

            # 添加assistant消息（instruction），包含tool_calls表示调用env_router
            messages.append({
                "role": "assistant",
                "content": f"Reasoning: {reasoning}\nInstruction: {instruction}",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "env_router",
                        "arguments": json.dumps({"instruction": instruction})
                    }
                }]
            })

            self._logger.debug(
                f"{_get_debug_info('收到LLM响应（ReAct Act）')} - reasoning: {reasoning}, instruction: {instruction}, status: {llm_status}, variables: {variables}"
            )

            # 检查LLM是否直接返回了status，如果是且不是in_progress/unknown，则跳过环境调用
            if llm_status and llm_status not in ["in_progress", "unknown", None]:
                self._logger.info(
                    f"{_get_debug_info('LLM直接返回status，跳过环境调用')} - status: {llm_status}"
                )
                final_status = llm_status
                final_answer = f"Step status determined directly: {llm_status}. Reasoning: {reasoning}"
                
                # 添加tool消息（模拟环境响应，但使用LLM返回的status）
                messages.append({
                    "role": "tool",
                    "content": final_answer,
                    "tool_call_id": tool_call_id,
                })
                
                # 记录act信息（跳过环境调用的情况）
                step_acts.append(
                    {
                        "plan_step_index": step_index,
                        "interaction_num": interaction_num + 1,
                        "instruction": instruction,
                        "reasoning": reasoning,
                        "answer": final_answer,
                        "status": final_status,
                        "skipped_env": True,  # 标记跳过了环境调用
                    }
                )
                
                # 保存记忆到cognition_memory
                self._add_cognition_memory(
                    f"ReAct interaction {interaction_num + 1} for step '{intention}': {instruction} -> Status determined directly: {llm_status}",
                    memory_type="react",
                    metadata={
                        "step_intention": intention,
                        "interaction_num": interaction_num + 1,
                        "skipped_env": True,
                    },
                )
                
                interaction_count += 1
                
                # 如果status不是in_progress，退出循环
                if final_status not in ["in_progress", "unknown"]:
                    break
            else:
                # 正常调用环境
                self._logger.debug(
                    f"{_get_debug_info('ReAct Observe阶段')} - 调用环境router执行指令: {instruction}, variables: {variables}"
                )

                # 如果启用了template模式且有variables，使用template模式调用
                if self.template_mode_enabled:
                    # 构建包含variables的ctx
                    template_ctx = ctx.copy()
                    template_ctx["variables"] = variables
                    updated_ctx, answer = await self.ask_env(template_ctx, instruction, readonly=False, template_mode=True)
                else:
                    updated_ctx, answer = await self.ask_env(ctx, instruction, readonly=False, template_mode=False)

                # self._logger.debug(
                #     f"{_get_debug_info('收到环境router响应（ReAct Observe）')} - answer length: {len(answer)}"
                # )
                self._logger.debug(
                    f"{_get_debug_info('环境router返回的完整answer')}:\n{answer}"
                )
                # 将返回的context添加到returns列表中
                if updated_ctx:
                    ctx["returns"].append(updated_ctx)

                final_status = updated_ctx.get("status", "unknown")
                final_answer = answer

                # 添加tool消息（环境响应），使用之前生成的tool_call_id
                messages.append({
                    "role": "tool",
                    "content": answer,
                    "tool_call_id": tool_call_id,
                })

                # 每次调用ask_env完成后，重新执行<observe>来获取环境的最新状态
                # 这保证了agent的"观察-思考-执行"循环
                self._logger.debug(
                    f"{_get_debug_info('重新执行observe获取最新环境状态')} - instruction: <observe>"
                )
                
                # 生成observe的tool_call_id
                observe_tool_call_id = f"observe_{interaction_num}_{interaction_count}"
                
                # 添加assistant消息，表示调用observe工具
                messages.append({
                    "role": "assistant",
                    "content": "I need to observe the environment to get the latest state.",
                    "tool_calls": [{
                        "id": observe_tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "env_router",
                            "arguments": json.dumps({"instruction": "<observe>"})
                        }
                    }]
                })
                
                observe_ctx, observation = await self.ask_env(
                    {"id": self._id}, "<observe>", readonly=True
                )
                
                # 检查observe的status状态
                observe_status = (
                    observe_ctx.get("status", "unknown") if observe_ctx else "unknown"
                )
                
                # 更新observation
                self._observation_ctx = observe_ctx
                self._observation = observation
                
                # 将observe结果添加到对话历史中
                observation_message = observation if observation else "No observation available."
                messages.append({
                    "role": "tool",
                    "content": observation_message,
                    "tool_call_id": observe_tool_call_id,
                })
                
                self._logger.debug(
                    f"{_get_debug_info('observe操作完成，已更新环境状态并加入对话历史')} - status: {observe_status}, observation length: {len(observation) if observation else 0}"
                )

                # 记录act信息（正常环境调用的情况）
                step_acts.append(
                    {
                        "plan_step_index": step_index,  # 该act属于计划的第几个步骤
                        "interaction_num": interaction_num + 1,  # 该步骤内的第几次交互
                        "instruction": instruction,
                        "reasoning": reasoning,
                        "answer": answer,
                        "status": final_status,
                    }
                )

                # 保存每次交互的记忆到cognition_memory（不立即加入mem0）
                self._add_cognition_memory(
                    f"ReAct interaction {interaction_num + 1} for step '{intention}': {instruction} -> {answer[:200]}",
                    memory_type="react",
                    metadata={
                        "step_intention": intention,
                        "interaction_num": interaction_num + 1,
                    },
                )

                interaction_count += 1

                # 如果status不是in_progress，退出循环（任务完成、失败或错误）
                if final_status not in ["in_progress", "unknown"]:
                    break

        # 根据执行结果确定步骤状态
        # final_status 来自代码生成的results['status']，包括：success, in_progress, fail, error
        self._logger.info(f"ReAct循环完成，最终status: {final_status}, 交互次数: {interaction_count}")

        if final_status == "success":
            # 代码执行成功，步骤完成
            self._logger.info(f"环境返回成功状态，步骤完成: {current_step.intention}")
            step_status = PlanStepStatus.COMPLETED
            current_step.status = step_status

            # 步骤完成，移动到下一步或标记计划完成
            if step_index + 1 < len(steps):
                current_plan.index = step_index + 1
            else:
                current_plan.completed = True
                current_plan.end_time = self._t
                await self._emotion_update_for_plan(current_plan, completed=True)
        elif final_status == "in_progress":
            # 代码执行中，步骤仍在进行中
            self._logger.info(
                f"环境返回进行中状态，步骤继续进行: {current_step.intention}"
            )
            step_status = PlanStepStatus.IN_PROGRESS
            current_step.status = step_status
        elif final_status == "fail":
            # 任务失败，步骤标记为失败
            self._logger.info(
                f"环境返回失败状态，标记步骤为失败: {current_step.intention}"
            )
            step_status = PlanStepStatus.FAILED
            current_step.status = step_status
            current_plan.failed = True
            current_plan.end_time = self._t
            await self._emotion_update_for_plan(current_plan, completed=False)
        elif final_status == "error":
            # 执行错误，步骤标记为失败
            self._logger.info(
                f"环境返回错误状态，标记步骤为失败: {current_step.intention}"
            )
            step_status = PlanStepStatus.FAILED
            current_step.status = step_status
            current_plan.failed = True
            current_plan.end_time = self._t
            await self._emotion_update_for_plan(current_plan, completed=False)
        else:
            # 未知状态，默认失败
            self._logger.warning(f"未知的status状态: {final_status}，标记步骤为失败")
            step_status = PlanStepStatus.FAILED
            current_step.status = step_status
            current_plan.failed = True
            current_plan.end_time = self._t
            await self._emotion_update_for_plan(current_plan, completed=False)

        # 记录执行结果
        evaluation = PlanStepEvaluation(
            success=(step_status == PlanStepStatus.COMPLETED),
            evaluation=f"ReAct interactions: {interaction_count}. Final result: {final_answer}. Status: {step_status.value}",
            consumed_time=10,  # 默认10分钟
        )
        current_step.evaluation = evaluation

        # 保存步骤执行记忆到cognition_memory（不立即加入mem0）
        # 注意：ReAct交互的记忆已经在循环中单独处理，这里只记录步骤执行总结
        memory_text = f"Executed step: {intention}. Status: {step_status.value}. Interactions: {interaction_count}. Final result: {final_answer[:200]}"
        self._add_cognition_memory(memory_text, memory_type="plan_execution")

        self._plan = current_plan
        return step_status, step_acts

    async def _emotion_update_for_plan(self, plan: Plan, completed: bool) -> None:
        """为计划完成/失败更新情感。"""
        assert self._tick is not None and self._t is not None, "tick and t are not set"
        plan_target = plan.target
        status = "successfully completed" if completed else "failed to complete"

        incident = f"You have {status} the plan: {plan_target}"

        # 获取agent状态
        state_text = await self.get_state()

        prompt = f"""{state_text}

<task>
Update your emotional state based on a significant incident: the completion or failure of a plan you were executing.
</task>

<incident>
{incident}
</incident>

<instructions>
Based on the incident above, consider:
1. How does this outcome align with your expectations and goals?
2. What impact does this have on your current emotional state?
3. How does your personality and past experiences influence your emotional response?

Update your emotion intensities to reflect your genuine emotional response to this incident. The values should be integers between 0-10, where:
- 0-2: Very low intensity
- 3-4: Low intensity
- 5-6: Moderate intensity
- 7-8: High intensity
- 9-10: Very high intensity

Select the most appropriate emotion type that best captures your overall emotional state from the available options.
</instructions>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "emotion": {{
        "sadness": int (0-10),
        "joy": int (0-10),
        "fear": int (0-10),
        "disgust": int (0-10),
        "anger": int (0-10),
        "surprise": int (0-10)
    }},
    "emotion_types": "One word from: Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate",
    "conclusion": "A brief, natural-language conclusion about how this incident affected your emotional state (e.g., 'I feel relieved that I successfully completed my plan' or 'I'm disappointed that my plan failed')"
}}
```
</format>

Your response is:
```json
"""

        # self._logger.debug(
        #     f"{_get_debug_info('准备调用LLM更新计划相关情感')} - plan target: {plan_target}, completed: {completed}, prompt length: {len(prompt)}"
        # )
        # self._logger.debug(
        #     f"{_get_debug_info('发送给LLM的完整prompt（更新计划情感）')}:\n{prompt}"
        # )

        response = await self.acompletion_with_pydantic_validation(
            model_type=EmotionUpdateResult,
            messages=[{"role": "user", "content": prompt}],
            tick=self._tick,
            t=self._t,
        )

        self._logger.debug(
            f"{_get_debug_info('收到LLM响应（更新计划情感）')} - emotion_type: {response.emotion_types.value}, conclusion: {response.conclusion[:100] if response.conclusion else 'None'}..."
        )

        self._emotion = response.emotion
        self._emotion_types = response.emotion_types
        if response.conclusion:
            # 记录情感更新记忆到cognition_memory（不立即加入mem0）
            self._add_cognition_memory(
                response.conclusion,
                memory_type="emotion",
            )

    # ==================== 认知更新方法 ====================

    async def _update_cognition_and_intention(self) -> CognitionIntentionUpdateResult:
        """合并执行需求调整、确定当前需求、更新情感思考、更新意图。"""
        assert self._tick is not None and self._t is not None, "tick and t are not set"
        t = self._t

        # 获取agent状态
        state_text = await self.get_state()

        # 获取当前observation
        current_observation_text = (
            self._observation if self._observation else "No current observation."
        )

        # 当前满意度信息（供模型参考）
        satiety = self._satisfactions.satiety
        energy = self._satisfactions.energy
        safety = self._satisfactions.safety
        social = self._satisfactions.social
        current_satisfaction_info = f"""Current Need Satisfaction Status:
- satiety: {satiety:.2f} (Threshold: {self.T_H}, Below Threshold: {satiety <= self.T_H})
- energy: {energy:.2f} (Threshold: {self.T_D}, Below Threshold: {energy <= self.T_D})
- safety: {safety:.2f} (Threshold: {self.T_P}, Below Threshold: {safety <= self.T_P})
- social: {social:.2f} (Threshold: {self.T_C}, Below Threshold: {social <= self.T_C})"""

        prompt = f"""{state_text}

<task>
Complete the following steps in order. Later steps MUST use outputs from earlier steps:
1) Adjust need satisfaction levels based on memories and current observation.
2) Determine current need using the adjusted satisfaction values from step 1.
3) Update thoughts and emotions using current need and observation.
4) Update intentions using TPB, focusing on the most urgent need.
</task>

<current_observation>
Here is the current observation from the environment in this time step. You should use this information across all steps.
{current_observation_text}
</current_observation>

<need_adjustment>
<explanation>
{NEED_DESCRIPTION}
</explanation>

<instructions>
Based on your recent memories, current observation, and experiences, determine if any need satisfaction levels should be adjusted:
1. Consider what happened in your recent memories (e.g., did you eat? did you rest? did you socialize?)
2. Consider what you observed in the current environment (e.g., available food, current location, time of day, etc.)
3. For each need type, decide if it should be increased, decreased, or maintained
4. Provide specific reasoning for each adjustment based on your memories and observation
5. Only adjust needs that have been affected by recent events or current observation
6. DO NOT change the the word of the need type in the adjustments.

Adjustment types:
- "increase": The need satisfaction should increase (e.g., after eating, satiety increases)
- "decrease": The need satisfaction should decrease (e.g., after a tiring activity, energy decreases)
- "maintain": The need satisfaction should stay the same (e.g., no relevant events occurred)
</instructions>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "adjustments": [
        {{
            "reasoning": "Why this adjustment is needed based on memories"
            "need_type": "satiety" | "energy" | "safety" | "social",
            "adjustment_type": "increase" | "decrease" | "maintain",
            "new_value": 0.0-1.0,
        }},
        ...
    ],
    "reasoning": "Overall reasoning for all adjustments"
}}
```
</format>
</need_adjustment>

<current_need>
<profile>
{self.profile_text}
</profile>

<explanation>
{NEED_DESCRIPTION}
</explanation>

<decision_rules>
When determining your current need, follow these principles:
1. **Priority matters**: Lower priority numbers mean higher urgency. Always consider needs in priority order (1 → 2 → 3 → 4 → 5).
2. **Urgency threshold**: A need is considered urgent when its satisfaction value is **below or equal to** its threshold. Only urgent needs should be prioritized.
3. **Default state**: If no needs are urgent (all satisfaction levels are above their thresholds), return "whatever" to indicate you have no specific urgent needs.

Remember: Your needs reflect your current state of well-being. Pay attention to satisfaction levels and thresholds to make appropriate decisions about what you need most right now.
Use the adjusted satisfaction values from the need adjustment step above.
</decision_rules>

<current_satisfaction_status>
{current_satisfaction_info}
</current_satisfaction_status>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "reasoning": "Brief explanation of your decision",
    "need_type": "satiety" | "energy" | "safety" | "social" | "whatever",
    "description": "A brief description of why you chose this need (e.g., 'I feel hungry' or 'I have no specific needs right now')"
}}
```
</format>
</current_need>

<cognition_update>
<instructions>
Review your recent memories, current observation, current need satisfaction levels, and current state, then:

1. **Thought Update**:
   - Reflect on what has happened recently and how it relates to your goals, values, and personality
   - Consider how your current need satisfaction levels affect your thoughts
   - Formulate a natural, coherent thought that captures your current mental state
   - The thought should be a complete sentence or short paragraph that reflects your genuine reflection

2. **Emotion Intensity Update**:
   - Consider how recent events AND your current need satisfaction levels have affected your emotional state
   - Low need satisfaction (especially for urgent needs) may cause negative emotions
   - High need satisfaction may cause positive emotions
   - Update emotion intensities to accurately reflect your current feelings
   - Values should be integers between 0-10, where:
     * 0-2: Very low intensity
     * 3-4: Low intensity
     * 5-6: Moderate intensity
     * 7-8: High intensity
     * 9-10: Very high intensity
   - Only update if there has been a meaningful change; otherwise, keep values similar to current state

3. **Emotion Type Selection**:
   - Choose the single emotion type that best describes your overall current emotional state
   - Consider the dominant emotion you're experiencing right now, influenced by both recent memories and need satisfaction
   - Select from: Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate
   - The value of "emotion_types" MUST be exactly one of the above words, case-sensitive, in English only. Do not translate or add any extra text.

Your response should reflect genuine introspection and emotional awareness based on your personality, recent experiences, and current need satisfaction levels.
</instructions>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "thought": "Your updated thoughts - a natural, reflective statement about your current mental state and recent experiences",
    "emotion": {{
        "sadness": int (0-10),
        "joy": int (0-10),
        "fear": int (0-10),
        "disgust": int (0-10),
        "anger": int (0-10),
        "surprise": int (0-10)
    }},
    "emotion_types": "One word from: Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate"
}}
```
</format>
</cognition_update>

<intention_update>
<explanation>
Theory of Planned Behavior (TPB) evaluates intentions based on three factors:
1. **Attitude**: Personal preference and evaluation of the behavior (0.0-1.0)
2. **Subjective Norm**: Social environment and others' views on this behavior (0.0-1.0)
3. **Perceived Control**: Difficulty and controllability of executing this behavior (0.0-1.0)

An intention with higher scores in these three dimensions is more likely to be executed.
</explanation>

<instructions>
Based on your current situation, current observation, especially focusing on the most urgent need, generate candidate intentions:

1. **Generate candidate intentions**:
   - Consider what intentions would help satisfy your current urgent need
   - Generate intentions that are relevant to your current situation
   - Evaluate each intention using TPB (attitude, subjective_norm, perceived_control)
   - Assign appropriate priority (lower number = higher priority)

2. **Total control**:
   - Keep the total number of candidate intentions within {self.max_intentions} (you can have fewer)
   - Prioritize intentions that address your most urgent need
   - Ensure intentions are realistic and actionable based on the AVAILABLE ACTIONS in the environment
   - You can include the current intention if it is still relevant, or create entirely new ones
   - Your intention should not include too many details, it's a intention, not a plan or movement

3. **Reasoning**: Provide overall reasoning for the intention update.
</instructions>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "reasoning": "Overall reasoning for the intention update",
    "intentions": [
        {{
            "intention": "Description of the intention",
            "priority": int (lower = higher priority),
            "attitude": 0.0-1.0,
            "subjective_norm": 0.0-1.0,
            "perceived_control": 0.0-1.0,
            "reasoning": "Why this intention is included"
        }},
        ...
    ]
}}
```
</format>
</intention_update>

<output_format>
Return a single JSON object with keys in this exact order:
1. "need_adjustment"
2. "current_need"
3. "cognition_update"
4. "intention_update"
Each value must follow its own format above. Do not add or rename fields.
</output_format>

Your response is:
```json
"""

        response = await self.acompletion_with_pydantic_validation(
            model_type=CognitionIntentionUpdateResult,
            messages=[{"role": "user", "content": prompt}],
            tick=self._tick,
            t=t,
        )

        self._last_cognition_intention_update = response

        self._logger.debug(
            f"{_get_debug_info('收到LLM响应（合并认知与意图）')} - "
            f"need_adjustments: {response.need_adjustment.adjustments}, "
            f"current_need: {response.current_need.need_type}, "
            f"emotion_type: {response.cognition_update.emotion_types.value}, "
            f"intentions: {response.intention_update.intentions}"
        )

        # 应用需求调整
        for adjustment in response.need_adjustment.adjustments:
            if adjustment.need_type == "satiety":
                self._satisfactions.satiety = adjustment.new_value
            elif adjustment.need_type == "energy":
                self._satisfactions.energy = adjustment.new_value
            elif adjustment.need_type == "safety":
                self._satisfactions.safety = adjustment.new_value
            elif adjustment.need_type == "social":
                self._satisfactions.social = adjustment.new_value

        # 记录调整记忆到cognition_memory（不立即加入mem0）
        self._add_cognition_memory(
            f"Adjusted needs based on memories: {response.need_adjustment.reasoning}",
            memory_type="need",
        )

        # 更新当前需求
        self._need = response.current_need.need_type

        # 更新情感和思考
        self._thought = response.cognition_update.thought
        self._emotion = response.cognition_update.emotion
        self._emotion_types = response.cognition_update.emotion_types

        # 记录情感和思考更新记忆到cognition_memory（不立即加入mem0）
        self._add_cognition_memory(
            f"Updated thought: {response.cognition_update.thought}\n"
            f"Updated emotion: {response.cognition_update.emotion_types.value}",
            memory_type="cognition",
        )

        # 更新意图
        response.intention_update.intentions.sort(key=lambda x: x.priority)
        self._intention = (
            response.intention_update.intentions[0]
            if response.intention_update.intentions
            else None
        )

        # 记录更新记忆到cognition_memory（不立即加入mem0）
        intention_text = ""
        if self._intention:
            intention_text = (
                f"Selected intention: {self._intention.intention} "
                f"(Priority: {self._intention.priority})"
            )
        self._add_cognition_memory(
            f"Updated intention: {response.intention_update.reasoning}\n{intention_text}",
            memory_type="intention",
        )

        return response

    async def _update_emotion_and_thought(self) -> None:
        """根据历史记忆和当前Need满足度更新Emotion和thought。"""
        assert self._tick is not None and self._t is not None, "tick and t are not set"
        assert self._need is not None, "need is not set"

        # 获取agent状态
        state_text = await self.get_state()

        # 获取当前observation
        current_observation_text = (
            self._observation if self._observation else "No current observation."
        )

        prompt = f"""{state_text}

<task>
Reflect on your recent experiences, current observation, and current need satisfaction levels, then update your thoughts and emotional state accordingly.
</task>

<current_observation>
Here is the current observation from the environment in this time step. You should use this information to help you update your thoughts and emotions.
{current_observation_text}
</current_observation>

<instructions>
Review your recent memories, current observation, current need satisfaction levels, and current state, then:

1. **Thought Update**: 
   - Reflect on what has happened recently and how it relates to your goals, values, and personality
   - Consider how your current need satisfaction levels affect your thoughts
   - Formulate a natural, coherent thought that captures your current mental state
   - The thought should be a complete sentence or short paragraph that reflects your genuine reflection

2. **Emotion Intensity Update**:
   - Consider how recent events AND your current need satisfaction levels have affected your emotional state
   - Low need satisfaction (especially for urgent needs) may cause negative emotions
   - High need satisfaction may cause positive emotions
   - Update emotion intensities to accurately reflect your current feelings
   - Values should be integers between 0-10, where:
     * 0-2: Very low intensity
     * 3-4: Low intensity  
     * 5-6: Moderate intensity
     * 7-8: High intensity
     * 9-10: Very high intensity
   - Only update if there has been a meaningful change; otherwise, keep values similar to current state

3. **Emotion Type Selection**:
   - Choose the single emotion type that best describes your overall current emotional state
   - Consider the dominant emotion you're experiencing right now, influenced by both recent memories and need satisfaction
   - Select from: Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate

Your response should reflect genuine introspection and emotional awareness based on your personality, recent experiences, and current need satisfaction levels.
</instructions>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "thought": "Your updated thoughts - a natural, reflective statement about your current mental state and recent experiences",
    "emotion": {{
        "sadness": int (0-10),
        "joy": int (0-10),
        "fear": int (0-10),
        "disgust": int (0-10),
        "anger": int (0-10),
        "surprise": int (0-10)
    }},
    "emotion_types": "One word from: Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate"
}}
```
</format>

Your response is:
```json
"""

        # self._logger.debug(
        #     f"{_get_debug_info('准备调用LLM更新情感和思考')} - prompt length: {len(prompt)}"
        # )
        # self._logger.debug(
        #     f"{_get_debug_info('发送给LLM的完整prompt（更新情感和思考）')}:\n{prompt}"
        # )

        response = await self.acompletion_with_pydantic_validation(
            model_type=CognitionUpdateResult,
            messages=[{"role": "user", "content": prompt}],
            tick=self._tick,
            t=self._t,
        )

        self._logger.debug(
            f"{_get_debug_info('收到LLM响应（更新情感和思考）')} - emotion_type: {response.emotion_types.value}, emotion: {response.emotion}, thought: {response.thought[:200]}..."
        )

        self._thought = response.thought
        self._emotion = response.emotion
        self._emotion_types = response.emotion_types

        # 记录情感和思考更新记忆到cognition_memory（不立即加入mem0）
        self._add_cognition_memory(
            f"Updated thought: {response.thought}\nUpdated emotion: {response.emotion_types.value}",
            memory_type="cognition",
        )

    async def _update_intention(self) -> IntentionUpdate:
        """根据历史记忆、当前Need需求满足度、最紧迫的need、当前Emotion和thought，使用计划行为理论更新意图。

        Returns:
            IntentionUpdate: 包含候选意图列表和更新理由（将选择优先级最高的一个）
        """
        assert self._tick is not None and self._t is not None, "tick and t are not set"
        assert self._need is not None, "need is not set"
        t = self._t

        # 获取agent状态
        state_text = await self.get_state()

        # 获取当前observation
        current_observation_text = (
            self._observation if self._observation else "No current observation."
        )

        prompt = f"""{state_text}

<task>
Update your intention based on your current situation, current observation, and focusing on the most urgent need. Use the Theory of Planned Behavior (TPB) to evaluate candidate intentions.
</task>

<current_observation>
Here is the current observation from the environment in this time step. You should use this information to help you generate and evaluate candidate intentions.
{current_observation_text}
</current_observation>

<explanation>
Theory of Planned Behavior (TPB) evaluates intentions based on three factors:
1. **Attitude**: Personal preference and evaluation of the behavior (0.0-1.0)
2. **Subjective Norm**: Social environment and others' views on this behavior (0.0-1.0)
3. **Perceived Control**: Difficulty and controllability of executing this behavior (0.0-1.0)

An intention with higher scores in these three dimensions is more likely to be executed.
</explanation>

<instructions>
Based on your current situation, current observation, especially focusing on the most urgent need, generate candidate intentions:

1. **Generate candidate intentions**: 
   - Consider what intentions would help satisfy your current urgent need
   - Generate intentions that are relevant to your current situation
   - Evaluate each intention using TPB (attitude, subjective_norm, perceived_control)
   - Assign appropriate priority (lower number = higher priority)

2. **Total control**: 
   - Keep the total number of candidate intentions within {self.max_intentions} (you can have fewer)
   - Prioritize intentions that address your most urgent need
   - Ensure intentions are realistic and actionable based on the AVAILABLE ACTIONS in the environment
   - You can include the current intention if it is still relevant, or create entirely new ones
   - Your intention should not include too many details, it's a intention, not a plan or movement

3. **Reasoning**: Provide overall reasoning for the intention update.
</instructions>

<format>
You should return the result in JSON format with the following structure:
```json
{{
    "intentions": [
        {{
            "intention": "Description of the intention",
            "priority": int (lower = higher priority),
            "attitude": 0.0-1.0,
            "subjective_norm": 0.0-1.0,
            "perceived_control": 0.0-1.0,
            "reasoning": "Why this intention is included"
        }},
        ...
    ],
    "reasoning": "Overall reasoning for the intention update"
}}
```
</format>

Your response is:
```json
"""

        # self._logger.debug(
        #     f"{_get_debug_info('准备调用LLM更新意图')} - prompt length: {len(prompt)}"
        # )
        # self._logger.debug(
        #     f"{_get_debug_info('发送给LLM的完整prompt（更新意图）')}:\n{prompt}"
        # )

        result = await self.acompletion_with_pydantic_validation(
            model_type=IntentionUpdate,
            messages=[{"role": "user", "content": prompt}],
            tick=self._tick,
            t=t,
        )

        self._logger.debug(
            f"{_get_debug_info('收到LLM响应（更新意图）')} - intentions: {result.intentions}, reasoning: {result.reasoning[:100]}..."
        )

        # 按优先级排序并只保留 top1
        result.intentions.sort(key=lambda x: x.priority)
        # 只保留优先级最高的 1 个 intention
        self._intention = result.intentions[0] if result.intentions else None

        # 记录更新记忆到cognition_memory（不立即加入mem0）
        intention_text = ""
        if self._intention:
            intention_text = f"Selected intention: {self._intention.intention} (Priority: {self._intention.priority})"
        self._add_cognition_memory(
            f"Updated intention: {result.reasoning}\n{intention_text}",
            memory_type="intention",
        )

        return result

    # ==================== 主执行方法 ====================

    async def init(
        self,
        env: RouterBase,
    ):
        """
        Initialize the agent.

        Args:
            llm: The LLM of the agent.
            env: The environment router of the agent.
        """
        await super().init(env=env)
        self._world_description = await env.get_world_description()

    async def step(self, tick: int, t: datetime) -> str:
        """
        执行一个完整的agent step。

        执行顺序：
        1. 使用<observe>对环境进行观察，作为最新的记忆
        2. 根据历史记忆，对需求（Need）进行调整，按调整的类型和列表返回，并且需要给出reasoning
        3. 根据历史记忆、当前Need满足度，更新Emotion和thought
        4. 根据历史记忆、当前Need需求满足度以及最紧迫的need、当前Emotion和thought，着重关注当前最紧迫的need，使用计划行为理论更新意图（从候选意图中选择优先级最高的一个）。
        5. 根据选中的意图以及与该意图相关的记忆，生成详细的执行计划 Plan，然后开始与环境交互以便完成这个Plan的每个Step。与环境交互的部分采用ReAct范式，对于一个Step最多与环境进行3次交互（参数，可以设置）。交互的过程将存入记忆中。环境交互的成功与否将在下一个循环时影响agent的需求和情绪

        Args:
            tick: 当前tick数
            t: 当前时间

        Returns:
            本步执行的描述文本
        """
        # 保存tick和t到实例变量
        self._tick = tick
        self._t = t

        self._step_count += 1
        step_log = []

        # 初始化当前时间步的所有acts记录
        self._current_step_acts = []
        # 清空当前step的认知记忆列表
        self._cognition_memory.clear()
        self._logger.debug(
            f"{_get_debug_info('开始执行agent step')} - step_count: {self._step_count}, tick: {tick}, t: {t.isoformat()}"
        )

        self._observation_ctx = None
        self._observation = None

        self._logger.debug(
            f"{_get_debug_info('准备调用环境router进行观察')} - instruction: <observe>"
        )

        observe_ctx, observation = await self.ask_env(
            {"id": self._id}, "<observe>", readonly=True
        )

        # self._logger.debug(
        #     f"{_get_debug_info('收到环境router观察响应')} - observation length: {len(observation) if observation else 0}"
        # )
        self._logger.debug(
            f"{_get_debug_info('环境router返回的完整observation')}:\n{observation if observation else 'None'}"
        )

        # 检查observe的status状态
        observe_status = (
            observe_ctx.get("status", "unknown") if observe_ctx else "unknown"
        )
        self._logger.debug(
            f"{_get_debug_info('observe操作返回的status')} - status: {observe_status}"
        )

        # 如果status不是in_progress，则认为observe操作完成
        if observe_status not in ["in_progress"]:
            self._observation_ctx = observe_ctx
            self._observation = observation
            # 将观察结果存入记忆
            await self._add_memory_with_timestamp(
                f"Observed environment: {observation}",
                metadata={"type": "observation"},
                t=t,
            )
            step_log.append(f"Observe: OK (status={observe_status})")
        else:
            step_log.append(f"Observe: InProgress (status={observe_status})")
            return "Skipped (observe in progress)"

        # 2-4. 合并执行需求调整、确定当前需求、更新情感思考、更新意图
        cognition_result = await self._update_cognition_and_intention()
        need_adjustment_result = cognition_result.need_adjustment
        step_log.append(
            f"NeedAdjust: {len(need_adjustment_result.adjustments)} adjustments"
        )
        step_log.append(f"Need: {self._need}")
        step_log.append(f"Emotion: {self._emotion_types.value}")
        step_log.append(
            f"Intention: {self._intention.intention if self._intention else 'None'}"
        )

        # 5. Plan相关的步骤处理
        if not self._intention:
            step_log.append("No intention available")
            return " | ".join(step_log)

        # 使用当前意图（LLM已返回优先级最高的一个）
        step_log.append(f"SelectedIntention: {self._intention.intention}")

        # 5.1 如果有活跃计划，根据observation和历史记忆，确定当前step的完成情况
        if self._plan and not self._plan.completed and not self._plan.failed:
            current_step_index = self._plan.index
            if current_step_index < len(self._plan.steps):
                current_step = self._plan.steps[current_step_index]
                # 如果步骤状态是进行中，检查完成情况
                if current_step.status == PlanStepStatus.IN_PROGRESS:
                    status = await self._check_step_completion(current_step)
                    current_step.status = status

                    if status == PlanStepStatus.COMPLETED:
                        # 步骤已完成，移动到下一步
                        if current_step_index + 1 < len(self._plan.steps):
                            self._plan.index = current_step_index + 1
                        else:
                            self._plan.completed = True
                            self._plan.end_time = self._t
                            await self._emotion_update_for_plan(
                                self._plan, completed=True
                            )
                        step_log.append(f"Step {current_step_index} completed")
                    elif status == PlanStepStatus.FAILED:
                        current_step.status = PlanStepStatus.FAILED
                        self._plan.failed = True
                        self._plan.end_time = self._t
                        await self._emotion_update_for_plan(self._plan, completed=False)
                        step_log.append(f"Step {current_step_index} failed")
                    else:
                        step_log.append(
                            f"Step {current_step_index} in progress, waiting"
                        )

        # 5.2 结合最新的意图，判定是否要中断计划
        should_interrupt = False
        if self._plan and not self._plan.completed and not self._plan.failed:
            should_interrupt = await self._should_interrupt_plan()
            if should_interrupt:
                step_log.append("Plan interrupted")
                self._plan = None

        # 5.3 如果没有活跃计划，根据最新意图生成计划
        if self._plan is None or self._plan.completed or self._plan.failed:
            await self._generate_plan_from_intention(self._intention)
            step_log.append(f"PlanGen: {self._plan.target if self._plan else 'N/A'}")

        # 5.4 有计划，执行计划的下一个step并根据与环境的交互结果判定这个step的状态
        if self._plan and not self._plan.completed and not self._plan.failed:
            current_step_index = self._plan.index
            if current_step_index < len(self._plan.steps):
                current_step = self._plan.steps[current_step_index]

                # 如果当前步骤是进行中，不再执行新步骤
                if current_step.status == PlanStepStatus.IN_PROGRESS:
                    step_log.append(f"Step {current_step_index} in progress, waiting")
                else:
                    # 执行步骤
                    step_status, step_acts = await self._step_execution()
                    self._current_step_acts.extend(step_acts)  # 收集acts
                    step_log.append(f"StepExec: {step_status.value}")

                    # 如果步骤完成或失败，继续处理后续步骤（如果可能）
                    # 如果步骤是进行中，停止处理
                    if step_status == PlanStepStatus.IN_PROGRESS:
                        step_log.append("Stopped: step in progress")
                    elif step_status == PlanStepStatus.COMPLETED:
                        # 继续执行后续步骤（如果还有）
                        while (
                            self._plan
                            and not self._plan.completed
                            and not self._plan.failed
                            and self._plan.index < len(self._plan.steps)
                        ):
                            next_step_status, next_step_acts = (
                                await self._step_execution()
                            )
                            self._current_step_acts.extend(next_step_acts)  # 收集acts
                            step_log.append(f"StepExec: {next_step_status.value}")
                            if next_step_status == PlanStepStatus.IN_PROGRESS:
                                step_log.append("Stopped: step in progress")
                                break
                            elif next_step_status == PlanStepStatus.FAILED:
                                break

        summary = " | ".join(step_log)

        # 记录当前时间步的详细信息
        await self._record_step_details()

        # 将当前step的认知记忆统一刷新到memory
        await self._flush_cognition_memory_to_memory()

        # 每两步查询一次agent的意图（第0、2、4步等）
        if self._step_count % 2 == 0:
            await self._query_current_intention()

        # 写入回放数据快照
        await self._write_replay_snapshot(step=self._step_count, t=t)

        return summary

    async def get_state(self) -> str:
        """返回agent的当前状态的字符串表示，用于放在prompt的最上方以提高缓存利用率。

        Returns:
            str: 格式化的状态字符串
        """
        # 获取最近的记忆
        recent_memories = await self._get_recent_memories(limit=10)
        memories_text = ""
        if recent_memories:
            memory_lines = []
            for mem in recent_memories:
                timestamp = mem.get("timestamp", "Unknown")
                content = mem.get("content", "")
                memory_lines.append(f'<memory t="{timestamp}">\n{content}\n</memory>')
            memories_text = "\n".join(memory_lines)
        else:
            memories_text = "No recent memories."

        # 构建计划信息
        plan_text = "No active plan."
        if self._plan:
            plan_steps_text = []
            for i, step in enumerate(self._plan.steps):
                step_text = (
                    f"  Step {i+1}: {step.intention} (Status: {step.status.value})"
                )
                if step.start_time:
                    step_text += f" [Started: {step.start_time.isoformat()}]"
                if step.evaluation:
                    step_text += f" [Result: {step.evaluation.evaluation}]"
                plan_steps_text.append(step_text)

            plan_status = (
                "completed"
                if self._plan.completed
                else "failed" if self._plan.failed else "active"
            )
            plan_text = f"""Plan Target: {self._plan.target}
Plan Status: {plan_status}
Current Step Index: {self._plan.index}
Steps:
{chr(10).join(plan_steps_text)}"""

        # 构建意图信息
        intention_text = "No intention."
        if self._intention:
            intention_text = (
                f"  {self._intention.intention} (Priority: {self._intention.priority})"
            )

        state_text = f"""<agent_state>
<world_description>
{self._world_description if self._world_description else 'No world description provided.'}
</world_description>

<profile>
{self.profile_text}
</profile>

<recent_memories>
{memories_text}
</recent_memories>

<need>
Need description: {NEED_DESCRIPTION}
Current need: {self._need if self._need else 'Not determined'}
Satisfactions:
- satiety: {self._satisfactions.satiety:.2f}
- energy: {self._satisfactions.energy:.2f}
- safety: {self._satisfactions.safety:.2f}
- social: {self._satisfactions.social:.2f}
</need>

<emotion>
Type: {self._emotion_types.value}
Intensities (0-10):
- sadness: {self._emotion.sadness}
- joy: {self._emotion.joy}
- fear: {self._emotion.fear}
- disgust: {self._emotion.disgust}
- anger: {self._emotion.anger}
- surprise: {self._emotion.surprise}
</emotion>

<thought>
{self._thought}
</thought>

<intention>
{intention_text}
</intention>

<plan>
{plan_text}
</plan>

</agent_state>"""

        return state_text

    # ==================== 回放数据写入方法 ====================

    async def _write_replay_snapshot(self, step: int, t: datetime) -> None:
        """写入回放状态快照。

        收集当前 Agent 的状态信息并写入回放数据库。

        Args:
            step: 当前步数
            t: 当前模拟时间
        """
        if self._replay_writer is None:
            return

        # 确定当前动作描述
        action = None
        if self._plan and self._plan.index < len(self._plan.steps):
            current_step = self._plan.steps[self._plan.index]
            action = current_step.intention

        # 构建状态 JSON
        status: dict[str, Any] = {
            "satisfactions": self._satisfactions.model_dump() if self._satisfactions else None,
            "emotion": self._emotion.model_dump() if self._emotion else None,
            "emotion_type": self._emotion_types.value if self._emotion_types else None,
            "thought": self._thought,
            "need": self._need,  # _need is a string (NeedType), not a Pydantic model
            "intention": self._intention.model_dump() if self._intention else None,
            "plan": {
                "target": self._plan.target,
                "index": self._plan.index,
                "completed": self._plan.completed,
                "failed": self._plan.failed,
                "steps": [
                    {
                        "intention": s.intention,
                        "status": s.status.value,
                    }
                    for s in self._plan.steps
                ],
            } if self._plan else None,
        }

        await self._write_status_snapshot(
            step=step,
            t=t,
            action=action,
            status=status,
        )

        # 记录思考作为对话（如果有）
        if self._thought:
            await self._write_dialog(
                step=step,
                t=t,
                dialog_type=DIALOG_TYPE_REFLECTION,
                speaker=self.name,
                content=self._thought,
            )

    async def dump(self) -> dict:
        # TODO: implement
        ...

    async def load(self, dump_data: dict):
        # TODO: implement
        ...

    async def ask(self, message: str, readonly: bool = True) -> str:
        """询问agent问题。

        Args:
            message: 要询问的问题
            readonly: 是否只读
        """
        # 获取agent状态
        state_text = await self.get_state()

        # 从memory中搜索相关信息（带3次重试机制）
        results = None
        async with self._memory_lock:
            results = await self._retry_async_operation(
                "search memories for ask",
                self.memory.search,
                message,
                user_id=self._memory_user_id,
                limit=20
            )

        memory_text = ""
        if results:
            # 提取记忆和时间信息，准备排序
            memory_list = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                memory_content = result.get("memory", "")
                # 从metadata获取时间戳
                timestamp = None
                metadata = result.get("metadata", {})
                timestamp_str = metadata.get("timestamp") if metadata else None
                if timestamp_str:
                    try:
                        timestamp_str_parsed = timestamp_str.replace("Z", "+00:00")
                        timestamp = datetime.fromisoformat(timestamp_str_parsed)
                    except (ValueError, TypeError):
                        pass

                memory_list.append(
                    {
                        "content": memory_content,
                        "timestamp": timestamp,
                        "timestamp_str": timestamp_str or "Unknown",
                    }
                )

            # 按时间从晚到早排序（timestamp为None的排在最后）
            memory_list.sort(
                key=lambda x: x["timestamp"] if x["timestamp"] else datetime.min,
                reverse=True,
            )

            # 使用XML标签包裹
            for mem in memory_list:
                memory_text += f"<memory t=\"{mem['timestamp_str']}\">\n{mem['content']}\n</memory>\n"

        # 构造prompt
        if memory_text:
            prompt = f"""{state_text}

<task>
Answer the question based on your current state and related memories.
</task>

<related_memories>
{memory_text}
</related_memories>

<question>
{message}
</question>

Your answer:"""
        else:
            prompt = f"""{state_text}

<task>
Answer the question based on your current state.
</task>

<question>
{message}
</question>

Your answer:"""

        self._logger.debug(
            f"{_get_debug_info('准备调用LLM回答用户问题')} - question: {message[:100]}..., prompt length: {len(prompt)}"
        )
        # self._logger.debug(
        #     f"{_get_debug_info('发送给LLM的完整prompt（回答用户问题）')}:\n{prompt}"
        # )

        # 使用LLM生成回答（不使用system_prompt）
        response = await self.acompletion(
            [{"role": "user", "content": prompt}],
            stream=False,
        )
        content = response.choices[0].message.content  # type: ignore

        # self._logger.debug(
        #     f"{_get_debug_info('收到LLM响应（回答用户问题）')} - answer length: {len(str(content)) if content else 0}"
        # )
        self._logger.debug(
            f"{_get_debug_info('LLM返回的完整answer')}:\n{content if content else 'None'}"
        )

        return str(content) if content else ""

    async def _record_step_details(self) -> None:
        """记录当前时间步的详细信息，包括intention、plan、act和需求状态。"""
        assert self._tick is not None and self._t is not None, "tick and t are not set"

        # 收集意图信息
        intentions_info = []
        if self._intention:
            intentions_info.append(
                {
                    "intention": self._intention.intention,
                    "priority": self._intention.priority,
                    "attitude": self._intention.attitude,
                    "subjective_norm": self._intention.subjective_norm,
                    "perceived_control": self._intention.perceived_control,
                    "reasoning": self._intention.reasoning,
                }
            )

        # 收集计划信息
        plan_info = None
        if self._plan:
            plan_steps_info = []
            for i, step in enumerate(self._plan.steps):
                step_info = {
                    "intention": step.intention,
                    "status": step.status.value,
                    "start_time": (
                        step.start_time.isoformat() if step.start_time else None
                    ),
                }
                if step.evaluation:
                    step_info["evaluation"] = {
                        "success": step.evaluation.success,
                        "evaluation": step.evaluation.evaluation,
                        "consumed_time": step.evaluation.consumed_time,
                    }
                plan_steps_info.append(step_info)

            plan_info = {
                "target": self._plan.target,
                "reasoning": self._plan.reasoning,
                "index": self._plan.index,
                "completed": self._plan.completed,
                "failed": self._plan.failed,
                "start_time": (
                    self._plan.start_time.isoformat() if self._plan.start_time else None
                ),
                "end_time": (
                    self._plan.end_time.isoformat() if self._plan.end_time else None
                ),
                "steps": plan_steps_info,
            }

        # 收集当前步骤的acts（从实例变量中获取）
        current_acts = self._current_step_acts.copy() if self._current_step_acts else []

        # 收集 observation 信息
        observation_info = None
        if self._observation:
            observation_info = {
                "observation": self._observation,
                "observation_status": (
                    self._observation_ctx.get("status")
                    if self._observation_ctx
                    else None
                ),
            }

        cognition_intention_update = None
        if self._last_cognition_intention_update:
            cognition_intention_update = (
                self._last_cognition_intention_update.model_dump(mode="json")
            )

        # 构建记录
        step_record = {
            "timestamp": self._t.isoformat(),
            "step_count": self._step_count,
            "observation": observation_info,
            "needs": {
                "satiety": self._satisfactions.satiety,
                "energy": self._satisfactions.energy,
                "safety": self._satisfactions.safety,
                "social": self._satisfactions.social,
                "current_need": self._need,
            },
            "cognition_intention_update": cognition_intention_update,
            "intentions": intentions_info,
            "plan": plan_info,
            "acts": current_acts,
        }

        self._step_records.append(step_record)

        # 立即保存记录到文件
        self._save_step_record_immediately(step_record)

        self._logger.debug(
            f"{_get_debug_info('记录时间步详情')} - tick: {self._tick}, step_count: {self._step_count}, "
            f"intentions: {len(intentions_info)}, acts: {len(current_acts)}"
        )

    def _save_step_record_immediately(self, step_record: dict) -> None:
        """
        立即保存单条时间步记录到文件（使用格式化的JSON，带缩进，提高可读性）。

        Args:
            step_record: 要保存的时间步记录
        """
        import os
        import json
        from datetime import datetime

        # 如果是第一次记录，初始化文件路径
        if self._step_records_file is None:
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self._step_records_file = (
                f"logs/agent_{self._id}_step_records-{timestamp}.json"
            )
            self._logger.info(
                f"Agent {self._id} step records file initialized: {self._step_records_file}"
            )
            # 第一次写入时，写入数组开始标记
            with open(self._step_records_file, "w", encoding="utf-8") as f:
                f.write("[\n")

        # 追加写入文件（格式化的JSON，带缩进）
        try:
            with open(self._step_records_file, "a", encoding="utf-8") as f:
                # 如果不是第一条记录，添加逗号和换行
                if len(self._step_records) > 1:
                    f.write(",\n")
                # 写入格式化的JSON对象（带2空格缩进）
                json_str = json.dumps(step_record, ensure_ascii=False, indent=2)
                # 为每行添加额外的2空格缩进（因为是在数组中）
                indented_json = "\n".join(
                    "  " + line if line.strip() else line
                    for line in json_str.split("\n")
                )
                f.write(indented_json)
        except Exception as e:
            self._logger.error(f"Failed to save step record: {e}")

    # TODO: 临时代码，后续修改为从外部ask agent
    async def _query_current_intention(self) -> None:
        """
        Query the agent's current intention every two steps.
        Valid intentions: sleep, home activity, other, work, shopping, eating out, leisure and entertainment.
        """
        assert self._t is not None, "current time is not set"

        # Define valid intentions
        valid_intentions = {
            "sleep",
            "home activity",
            "other",
            "work",
            "shopping",
            "eating out",
            "leisure and entertainment",
        }

        # Get agent state
        state_text = await self.get_state()

        # Build query prompt
        prompt = f"""{state_text}

<task>
Based on your recent memories, current state, and current activity, choose ONE intention that best describes what you are doing or planning to do right now.
</task>

<instructions>
The choices are:
1. sleep - You are sleeping or about to sleep
2. home activity - You are doing activities at home or about to do activities at home.
3. other - Other activities not listed here.
4. work - You are working or doing work-related activities or about to work.
5. shopping - You are shopping or buying things or about to shop.
6. eating out - You are eating at a restaurant or food establishment or about to eat at a restaurant or food establishment.
7. leisure and entertainment - You are doing leisure activities or entertainment or about to do leisure activities or entertainment.

Respond with ONLY the intention name from the list above. Do not include any explanation, just the name.
</instructions>

Your response:"""

        # self._logger.debug(
        #     f"{_get_debug_info('准备查询当前意图')} - step: {self._step_count}, t: {self._t.isoformat()}"
        # )

        # Query using LLM
        response = await self.acompletion(
            [{"role": "user", "content": prompt}],
            stream=False,
        )
        intention_text = (
            response.choices[0].message.content.strip().lower()  # type: ignore
            if response.choices[0].message.content  # type: ignore
            else "other"
        )

        # Parse intention
        intention = None
        for valid in valid_intentions:
            if valid.lower() in intention_text.lower():
                intention = valid
                break

        if intention is None:
            intention = "other"

        # Record intention
        intention_record = {
            "timestamp": self._t.isoformat(),
            "step": self._step_count,
            "intention": intention,
        }
        self._intention_history.append(intention_record)

        self._logger.info(
            f"Agent {self._id} - Step {self._step_count}: Intention = {intention}"
        )

    async def close(self):
        """关闭agent，清理资源。"""
        # Save intention history
        self._save_intention_history()

        # 重置memory（带3次重试机制）
        await self._retry_async_operation(
            "reset memory",
            self._memory.reset
        )
        self._logger.info(f"PersonAgent '{self.id}' (ID: {self.id}) 已关闭")
        return None

    def _save_intention_history(self) -> None:
        """
        Save the agent's intention history to a log file (with timestamp).
        """
        import os
        import json
        from datetime import datetime

        os.makedirs("logs", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        log_file = f"logs/agent_{self._id}_intention_history-{timestamp}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self._intention_history, f, indent=2, ensure_ascii=False)

        self._logger.info(
            f"Agent {self._id} intention history saved to {log_file} ({len(self._intention_history)} records)"
        )
