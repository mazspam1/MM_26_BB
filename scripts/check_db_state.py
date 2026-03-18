"""
Quick check of database state for tournament predictions.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from packages.common.database import get_connection


def check_db():
    with get_connection() as conn:
        # Check teams
        teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
        print(f"Teams in database: {teams}")

        # Check team_strengths (ratings)
        try:
            ratings = conn.execute("SELECT COUNT(*) FROM team_strengths").fetchone()[0]
            print(f"Team ratings: {ratings}")

            # Show sample ratings
            sample = conn.execute("""
                SELECT t.name, ts.adj_offensive_efficiency, ts.adj_defensive_efficiency, ts.adj_em
                FROM team_strengths ts
                JOIN teams t ON ts.team_id = t.team_id
                WHERE ts.as_of_date = (SELECT MAX(as_of_date) FROM team_strengths)
                ORDER BY ts.adj_em DESC
                LIMIT 10
            """).fetchall()

            if sample:
                print("\nTop 10 teams by Efficiency Margin:")
                for name, off, def_, em in sample:
                    print(f"  {name:20} EM: {em:+6.1f}  Off: {off:5.1f}  Def: {def_:5.1f}")
        except Exception as e:
            print(f"No ratings table: {e}")

        # Check games
        try:
            games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            print(f"\nGames in database: {games}")
        except:
            print("\nNo games table")

        # Check box_scores
        try:
            boxes = conn.execute("SELECT COUNT(*) FROM box_scores").fetchone()[0]
            print(f"Box scores: {boxes}")
        except:
            print("No box_scores table")

        # Check tournament tables
        try:
            conn.execute("SELECT COUNT(*) FROM tournament_bracket").fetchone()
            print("\nTournament bracket table: EXISTS")
        except:
            print("\nTournament bracket table: NOT FOUND")

        try:
            conn.execute("SELECT COUNT(*) FROM tournament_predictions").fetchone()
            print("Tournament predictions table: EXISTS")
        except:
            print("Tournament predictions table: NOT FOUND")


if __name__ == "__main__":
    check_db()
