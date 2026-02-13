"""Replay table definitions for MobilitySpace.

- AGENT_POSITION_SCHEMA: TableSchema for ReplayWriter.register_table() (create table).
- agent_position_table: SQLAlchemy Table for replay API query (read-only).
MobilitySpace creates and writes agent_position; the replay router uses this module
to query it, consistent with social_media/replay_tables.py.
"""

from simframework.storage import ColumnDef, TableSchema
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, Table

# ----- TableSchema for table creation (used by MobilitySpace.init) -----
AGENT_POSITION_SCHEMA = TableSchema(
    name="agent_position",
    columns=[
        ColumnDef("id", "INTEGER", nullable=False),
        ColumnDef("step", "INTEGER", nullable=False),
        ColumnDef("t", "TIMESTAMP", nullable=False),
        ColumnDef("lng", "REAL"),
        ColumnDef("lat", "REAL"),
        ColumnDef("created_at", "TIMESTAMP", default="CURRENT_TIMESTAMP"),
    ],
    primary_key=["id", "step"],
    indexes=[["step"], ["t"]],
)

# ----- SQLAlchemy Table for replay API (query-only) -----
_meta = MetaData()
agent_position_table = Table(
    "agent_position",
    _meta,
    Column("id", Integer, primary_key=True),
    Column("step", Integer, primary_key=True),
    Column("t", DateTime),
    Column("lng", Float),
    Column("lat", Float),
    Column("created_at", DateTime),
)
