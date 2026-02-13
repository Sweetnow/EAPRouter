"""Storage module for SimFramework replay data.

- models: SQLModel definitions for framework tables (agent_profile, agent_status,
  agent_dialog). ReplayWriter creates these in init() and provides typed write_*
  methods.
- table_schema: ColumnDef and TableSchema for dynamic table registration. Env
  modules use these with ReplayWriter.register_table() and write().
- replay_writer: ReplayWriter writes to SQLite; framework tables via ORM, dynamic
  tables via register_table + write. See replay_writer module docstring for
  architecture and extensibility.
"""

from .replay_writer import ReplayWriter
from .models import (
    AgentProfile,
    AgentStatus,
    AgentDialog,
)
from .table_schema import ColumnDef, TableSchema

__all__ = [
    "ReplayWriter",
    "AgentProfile",
    "AgentStatus",
    "AgentDialog",
    "ColumnDef",
    "TableSchema",
]
