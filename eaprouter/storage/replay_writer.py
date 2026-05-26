"""Replay data writer for storing simulation state to SQLite.

Architecture:
- Framework tables (agent_profile, agent_status, agent_dialog): defined in models.py
  as SQLModel classes; created in init() via SQLModel.metadata.create_all; written via
  write_agent_profile, write_agent_status, write_agent_dialog (and batch variants).
- Dynamic tables (e.g. agent_position, social_*): each environment module owns its
  schema and data. The module calls register_table(TableSchema) during init() to
  create the table, and write(table_name, row) to persist rows. Query-only
  SQLAlchemy Table definitions for the replay API live in each module's
  replay_tables.py (e.g. contrib/env/mobility_space/replay_tables.py).
Extensibility: new env modules can add replay tables by defining a TableSchema,
calling register_table() and write(), and exposing a Table in replay_tables.py for
the replay router to use.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sqlalchemy import JSON, Column, MetaData, Table, insert, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from .models import AgentDialog, AgentProfile, AgentStatus
from .table_schema import TableSchema


class ReplayWriter:
    """Thread-safe async SQLite writer for replay data.

    This class handles writing simulation state data to SQLite database
    for later replay and analysis. It uses SQLAlchemy/SQLModel for ORM support.
    """

    def __init__(self, db_path: Path):
        """Initialize the replay writer.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = db_path
        self._engine: Optional[AsyncEngine] = None
        self._session_maker: Optional[sessionmaker] = None
        self._lock = asyncio.Lock()
        self._registered_tables: Set[str] = set()

    async def init(self) -> None:
        """Initialize database connection and create tables."""
        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create async engine
        connection_string = f"sqlite+aiosqlite:///{self._db_path}"
        self._engine = create_async_engine(connection_string, echo=False)

        # Create session maker
        self._session_maker = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

        # Create tables defined in SQLModel (framework models only; agent_position is created by MobilitySpace)
        async with self._engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        for name in SQLModel.metadata.tables:
            self._registered_tables.add(name)


    async def close(self) -> None:
        """Flush pending writes and close database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None

    # ==================== Generic Table Operations ====================

    async def register_table(self, schema: TableSchema) -> None:
        """Register and create a new table dynamically.

        Args:
            schema: TableSchema defining the table structure.
        """
        if schema.name in self._registered_tables:
            return

        # Check if table already exists in SQLModel metadata (it shouldn't for dynamic ones)
        if schema.name in SQLModel.metadata.tables:
            self._registered_tables.add(schema.name)
            return

        async with self._lock:
            # Create table using SQLAlchemy Core
            # We construct a Table object and create it
            async with self._engine.begin() as conn:
                 # Use raw SQL for dynamic creation as it's easier to map TableSchema to SQL than constructing SA Table dynamically
                create_sql = schema.to_create_sql()
                await conn.execute(text(create_sql))
                
                # Create indexes
                for index_sql in schema.to_index_sql():
                    await conn.execute(text(index_sql))

            self._registered_tables.add(schema.name)

    async def write(self, table_name: str, data: Dict[str, Any]) -> None:
        """Write a single row to a table.

        Args:
            table_name: Name of the table to write to.
            data: Dictionary of column names to values.
        """
        # For statically defined tables, allow using generic write but process specially?
        # Actually, for dynamic tables, we need to construct INSERT statement
        
        # Prepare data (handle datetime and JSON)
        processed_data = self._process_data_for_write(data)

        async with self._lock:
           async with self._session_maker() as session:
                # Use raw SQL for generic write to arbitrary tables (simplest for dynamic schemas)
                # Or reflect table... but reflection is slow.
                # Since we know the schema from register_table, we could cache SA Table objects.
                # But here we just use what we have.
                
                columns = list(processed_data.keys())
                placeholders = ", ".join([f":{col}" for col in columns])
                columns_str = ", ".join(columns)
                sql = text(f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})")
                
                await session.execute(sql, processed_data)
                await session.commit()

    async def write_batch(
        self, table_name: str, data_list: List[Dict[str, Any]]
    ) -> None:
        """Write multiple rows to a table in a single transaction."""
        if not data_list:
            return

        processed_list = [self._process_data_for_write(d) for d in data_list]
        columns = list(processed_list[0].keys())
        placeholders = ", ".join([f":{col}" for col in columns])
        columns_str = ", ".join(columns)
        sql = text(f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})")

        async with self._lock:
            async with self._session_maker() as session:
                await session.execute(sql, processed_list)
                await session.commit()
    
    def _process_data_for_write(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert objects into format suitable for SQLite/SQLAlchemy."""
        new_data = {}
        for k, v in data.items():
            if isinstance(v, datetime):
                new_data[k] = v # SQLAlchemy handles datetime
            elif isinstance(v, (dict, list)):
                # If we use JSON type in SA, we can pass dict directly. 
                # But for dynamic tables created via SQL, they might be TEXT columns.
                # We need to trust how they were defined.
                # However, our dynamic schema defines JSON as "JSON" type (which is TEXT affinity in SQLite)
                # It's safer to dump string for raw SQL execution unless we use SA Table bound parameter
                new_data[k] = json.dumps(v, ensure_ascii=False)
            else:
                new_data[k] = v
        return new_data

    # ==================== Typed Operations (Using Models) ====================

    async def write_agent_profile(
        self, agent_id: int, name: str, profile: Dict[str, Any]
    ) -> None:
        """Write agent profile using ORM."""
        obj = AgentProfile(id=agent_id, name=name, profile=profile)
        async with self._lock:
            async with self._session_maker() as session:
                await session.merge(obj)
                await session.commit()

    async def write_agent_profiles_batch(
        self, profiles: List[Tuple[int, str, Dict[str, Any]]]
    ) -> None:
        """Write batch profiles."""
        objs = [
            AgentProfile(id=pid, name=name, profile=prof)
            for pid, name, prof in profiles
        ]
        async with self._lock:
            async with self._session_maker() as session:
                for obj in objs:
                    await session.merge(obj)
                await session.commit()

    async def write_agent_status(
        self,
        agent_id: int,
        step: int,
        t: datetime,
        action: Optional[str] = None,
        status: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write agent status using ORM."""
        obj = AgentStatus(
            id=agent_id, step=step, t=t, action=action, status=status or {}
        )
        async with self._lock:
            async with self._session_maker() as session:
                await session.merge(obj)
                await session.commit()

    async def write_agent_statuses_batch(
        self,
        statuses: List[Tuple[int, int, datetime, Optional[str], Optional[Dict[str, Any]]]],
    ) -> None:
        """Write batch statuses."""
        objs = [
            AgentStatus(
                id=aid, step=step, t=t, action=action, status=status or {}
            )
            for aid, step, t, action, status in statuses
        ]
        async with self._lock:
            async with self._session_maker() as session:
                for obj in objs:
                    await session.merge(obj)
                await session.commit()

    async def write_agent_dialog(
        self,
        agent_id: int,
        step: int,
        t: datetime,
        dialog_type: int,
        speaker: str,
        content: str,
    ) -> None:
        """Write dialog."""
        obj = AgentDialog(
            agent_id=agent_id,
            step=step,
            t=t,
            type=dialog_type,
            speaker=speaker,
            content=content,
        )
        async with self._lock:
            async with self._session_maker() as session:
                session.add(obj)
                await session.commit()

    async def write_agent_dialogs_batch(
        self, dialogs: List[Tuple[int, int, datetime, int, str, str]]
    ) -> None:
        """Write batch dialogs."""
        objs = [
            AgentDialog(
                agent_id=aid,
                step=step,
                t=t,
                type=dtype,
                speaker=spk,
                content=cont,
            )
            for aid, step, t, dtype, spk, cont in dialogs
        ]
        async with self._lock:
            async with self._session_maker() as session:
                session.add_all(objs)
                await session.commit()

    # Query methods for testing and debugging
    async def get_step_count(self) -> int:
        async with self._session_maker() as session:
            result = await session.execute(text("SELECT COUNT(DISTINCT step) FROM agent_status"))
            return result.scalar() or 0

    async def get_agent_count(self) -> int:
        async with self._session_maker() as session:
             result = await session.execute(text("SELECT COUNT(*) FROM agent_profile"))
             return result.scalar() or 0

