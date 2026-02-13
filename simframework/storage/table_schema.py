"""Table schema definitions for dynamic replay table registration.

Environment modules (e.g. social_media, mobility_space) define their own replay tables
using TableSchema and ColumnDef. They call ReplayWriter.register_table(schema) to
create the table and ReplayWriter.write(table_name, row) to persist data. The replay
API reads these tables using query-only SQLAlchemy Table definitions provided by each
module in its replay_tables.py (e.g. contrib/env/social_media/replay_tables.py,
contrib/env/mobility_space/replay_tables.py). Framework tables (agent_profile,
agent_status, agent_dialog) are defined in models.py (SQLModel) and created by
ReplayWriter.init(); they are not registered via TableSchema.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

# SQLite column types
ColumnType = Literal["INTEGER", "REAL", "TEXT", "BLOB", "TIMESTAMP", "JSON"]


@dataclass
class ColumnDef:
    """Column definition for a table.
    
    Attributes:
        name: Column name
        type: SQLite column type
        nullable: Whether the column allows NULL values (default True)
        default: Default value expression (e.g., "CURRENT_TIMESTAMP")
    """
    name: str
    type: ColumnType
    nullable: bool = True
    default: Optional[str] = None
    
    def to_sql(self) -> str:
        """Generate SQL column definition."""
        parts = [self.name, self.type]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.default is not None:
            parts.append(f"DEFAULT {self.default}")
        return " ".join(parts)


@dataclass
class TableSchema:
    """Schema definition for a database table.
    
    This class allows environment modules to define their own tables
    that will be created dynamically by ReplayWriter.
    
    Attributes:
        name: Table name
        columns: List of column definitions
        primary_key: List of column names that form the primary key
        indexes: List of index definitions (column name or list of column names)
    
    Example:
        >>> schema = TableSchema(
        ...     name="agent_position",
        ...     columns=[
        ...         ColumnDef("id", "INTEGER", nullable=False),
        ...         ColumnDef("step", "INTEGER", nullable=False),
        ...         ColumnDef("t", "TIMESTAMP", nullable=False),
        ...         ColumnDef("lng", "REAL"),
        ...         ColumnDef("lat", "REAL"),
        ...     ],
        ...     primary_key=["id", "step"],
        ...     indexes=[["step"], ["t"]],
        ... )
    """
    name: str
    columns: List[ColumnDef]
    primary_key: List[str] = field(default_factory=list)
    indexes: List[List[str]] = field(default_factory=list)
    
    def to_create_sql(self) -> str:
        """Generate CREATE TABLE SQL statement."""
        column_defs = [col.to_sql() for col in self.columns]
        
        # Add primary key constraint
        if self.primary_key:
            pk_cols = ", ".join(self.primary_key)
            column_defs.append(f"PRIMARY KEY ({pk_cols})")
        
        columns_sql = ",\n    ".join(column_defs)
        return f"CREATE TABLE IF NOT EXISTS {self.name} (\n    {columns_sql}\n)"
    
    def to_index_sql(self) -> List[str]:
        """Generate CREATE INDEX SQL statements."""
        statements = []
        for idx_cols in self.indexes:
            idx_name = f"idx_{self.name}_{'_'.join(idx_cols)}"
            cols = ", ".join(idx_cols)
            statements.append(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {self.name}({cols})"
            )
        return statements
