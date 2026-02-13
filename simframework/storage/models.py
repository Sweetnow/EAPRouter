"""SQLModel definitions for replay data storage."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class AgentProfile(SQLModel, table=True):
    """Agent profile information."""

    __tablename__ = "agent_profile"

    id: int = Field(primary_key=True)
    name: str
    profile: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now)


class AgentStatus(SQLModel, table=True):
    """Agent status snapshot at a specific step."""

    __tablename__ = "agent_status"

    id: int = Field(primary_key=True)
    step: int = Field(primary_key=True, index=True)
    t: datetime = Field(index=True)
    action: Optional[str] = None
    status: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now)


class AgentDialog(SQLModel, table=True):
    """Agent dialog record."""

    __tablename__ = "agent_dialog"

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: int = Field(index=True)
    step: int = Field(index=True)
    t: datetime
    type: int = Field(index=True)  # 0=反思 (thought/reflection); V2 only uses 0
    speaker: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
