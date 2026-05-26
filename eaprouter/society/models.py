"""Pydantic models for eaprouter experiment configuration validation"""

from datetime import datetime
from typing import Dict, Any, List, Union, Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "EnvModuleConfig",
    "AgentConfig",
    "InitConfig",
    "RunStep",
    "RunToStep",
    "AskStep",
    "InterveneStep",
    "StepUnion",
    "StepsConfig",
]


class EnvModuleConfig(BaseModel):
    """环境模块配置模型"""
    
    module_type: str = Field(..., description="环境模块类型")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="环境模块初始化参数")


class AgentConfig(BaseModel):
    """Agent配置模型，匹配init_config.json中的格式"""
    
    agent_id: int = Field(..., description="Agent的唯一ID")
    agent_type: str = Field(..., description="Agent类型")
    kwargs: Dict[str, Any] = Field(..., description="Agent初始化参数，包含id、profile等所有参数")
    
    @field_validator("kwargs")
    @classmethod
    def validate_kwargs(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """验证kwargs中必须包含id字段"""
        if "id" not in v:
            raise ValueError("kwargs must contain 'id' field")
        return v


class InitConfig(BaseModel):
    """初始化配置文件模型"""
    
    env_modules: List[EnvModuleConfig] = Field(..., min_length=1, description="环境模块列表")
    agents: List[AgentConfig] = Field(..., min_length=1, description="Agent列表")


class RunStep(BaseModel):
    """运行指定步数的步骤"""
    
    type: Literal["run"] = Field("run", description="步骤类型")
    num_steps: int = Field(..., gt=0, description="运行的步数")
    tick: int = Field(1, gt=0, description="每步的时间间隔（秒）")


class RunToStep(BaseModel):
    """运行到指定时间的步骤"""
    
    type: Literal["run_to"] = Field("run_to", description="步骤类型")
    end_t: str = Field(..., description="结束时间（ISO格式）")
    tick: int = Field(1, gt=0, description="每步的时间间隔（秒）")
    
    @field_validator("end_t")
    @classmethod
    def validate_end_t(cls, v: str) -> str:
        """验证时间格式"""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid ISO datetime format: {v}")
        return v


class AskStep(BaseModel):
    """提问步骤"""
    
    type: Literal["ask"] = Field("ask", description="步骤类型")
    question: str = Field(..., min_length=1, description="要提问的问题")


class InterveneStep(BaseModel):
    """干预步骤"""
    
    type: Literal["intervene"] = Field("intervene", description="步骤类型")
    instruction: str = Field(..., min_length=1, description="干预指令")


StepUnion = Union[RunStep, RunToStep, AskStep, InterveneStep]


class StepsConfig(BaseModel):
    """Steps.yaml配置文件模型"""
    
    start_t: str = Field(..., description="仿真开始时间（ISO格式）")
    steps: List[StepUnion] = Field(..., min_length=1, description="步骤列表")
    
    @field_validator("start_t")
    @classmethod
    def validate_start_t(cls, v: str) -> str:
        """验证开始时间格式"""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid ISO datetime format: {v}")
        return v
