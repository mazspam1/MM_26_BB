"""
Initialize tournament schema in database.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from packages.common.database import get_connection


def init_tournament_schema():
    schema_path = project_root / "data" / "tournament_schema.sql"

    if not schema_path.exists():
        print(f"Schema file not found: {schema_path}")
        return False

    print(f"Initializing tournament schema from {schema_path}")

    with get_connection() as conn:
        # Execute each CREATE statement separately to handle IF NOT EXISTS
        sql = schema_path.read_text()

        # Split by semicolons and execute each statement
        statements = [
            s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")
        ]

        for stmt in statements:
            try:
                conn.execute(stmt)
            except Exception as e:
                # Skip if already exists or minor issues
                if "already exists" not in str(e).lower():
                    print(f"Warning: {e}")

    print("Tournament schema initialized!")

    # Verify tables
    with get_connection() as conn:
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name LIKE 'tournament%'
        """).fetchall()

        print("\nTournament tables created:")
        for t in tables:
            print(f"  - {t[0]}")

    return True


if __name__ == "__main__":
    init_tournament_schema()
