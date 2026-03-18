"""
DuckDB database connection management.

Provides connection pooling, schema initialization, and common query utilities.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import duckdb
import structlog
from pydantic_settings import BaseSettings

logger = structlog.get_logger()


class DatabaseSettings(BaseSettings):
    """Database configuration from environment."""

    database_path: str = "data/cbb_lines.duckdb"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


_settings: Optional[DatabaseSettings] = None


def get_settings() -> DatabaseSettings:
    """Get database settings (cached)."""
    global _settings
    if _settings is None:
        _settings = DatabaseSettings()
    return _settings


def get_database_path() -> Path:
    """Get the absolute path to the database file."""
    settings = get_settings()
    db_path = Path(settings.database_path)
    if not db_path.is_absolute():
        # Relative to project root
        project_root = Path(__file__).parent.parent.parent
        db_path = project_root / db_path
    return db_path


@contextmanager
def get_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Context manager for DuckDB connections.

    Usage:
        with get_connection() as conn:
            result = conn.execute("SELECT * FROM teams").fetchall()
    """
    db_path = get_database_path()

    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def init_database() -> None:
    """
    Initialize database schema from base and tournament SQL files.

    Should be called once at application startup.
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    schema_paths = [
        data_dir / "schema.sql",
        data_dir / "tournament_schema.sql",
    ]

    existing_paths = [path for path in schema_paths if path.exists()]
    missing_paths = [path for path in schema_paths if not path.exists()]

    for path in missing_paths:
        logger.warning("Schema file not found", path=str(path))

    if not existing_paths:
        return

    with get_connection() as conn:
        for schema_path in existing_paths:
            logger.info("Initializing database schema", schema_path=str(schema_path))
            conn.execute(schema_path.read_text())

        _ensure_tournament_bracket_columns(conn)

    logger.info("Database schema initialized successfully")


def _ensure_tournament_bracket_columns(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure tournament tables have newer columns on older databases."""
    try:
        conn.execute("ALTER TABLE tournament_bracket ADD COLUMN victor_game_position VARCHAR(10)")
    except duckdb.CatalogException:
        return
    except duckdb.Error as exc:
        if "already exists" not in str(exc).lower():
            raise


def check_connection() -> bool:
    """
    Check if database connection is working.

    Returns True if connection is successful, False otherwise.
    """
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        return True
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False


def get_table_counts() -> dict[str, int]:
    """
    Get row counts for all tables.

    Useful for health checks and debugging.
    """
    tables = [
        "teams",
        "games",
        "box_scores",
        "team_strengths",
        "line_snapshots",
        "predictions",
        "backtest_predictions",
        "backtest_runs",
        "backtest_segments",
        "clv_reports",
    ]

    counts: dict[str, int] = {}

    with get_connection() as conn:
        for table in tables:
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = result[0] if result else 0
            except duckdb.CatalogException:
                # Table doesn't exist yet
                counts[table] = -1

    return counts


def execute_query(query: str, params: Optional[tuple] = None) -> list:
    """
    Execute a query and return results.

    Args:
        query: SQL query string
        params: Optional tuple of parameters for parameterized queries

    Returns:
        List of result rows
    """
    with get_connection() as conn:
        if params:
            result = conn.execute(query, params).fetchall()
        else:
            result = conn.execute(query).fetchall()
    return result


def execute_many(query: str, data: list[tuple]) -> int:
    """
    Execute a query for multiple rows (bulk insert).

    Args:
        query: SQL query string with placeholders
        data: List of tuples containing parameter values

    Returns:
        Number of rows affected
    """
    with get_connection() as conn:
        conn.executemany(query, data)
        return len(data)


def insert_dataframe(table_name: str, df) -> int:
    """
    Insert a Polars or Pandas DataFrame into a table.

    Args:
        table_name: Target table name
        df: Polars or Pandas DataFrame

    Returns:
        Number of rows inserted
    """
    with get_connection() as conn:
        # DuckDB can insert directly from DataFrames
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        return len(df)


def upsert_dataframe(table_name: str, df, key_columns: list[str]) -> int:
    """
    Upsert a DataFrame (insert or update on conflict).

    Args:
        table_name: Target table name
        df: Polars or Pandas DataFrame
        key_columns: List of columns that form the primary key

    Returns:
        Number of rows affected
    """
    key_clause = ", ".join(key_columns)

    with get_connection() as conn:
        # Create temp table
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS temp_upsert AS SELECT * FROM df")

        # Delete existing rows that match keys
        delete_conditions = " AND ".join(
            [f"{table_name}.{col} = temp_upsert.{col}" for col in key_columns]
        )
        conn.execute(
            f"DELETE FROM {table_name} WHERE EXISTS "
            f"(SELECT 1 FROM temp_upsert WHERE {delete_conditions})"
        )

        # Insert all rows from temp
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_upsert")

        # Clean up
        conn.execute("DROP TABLE temp_upsert")

        return len(df)
