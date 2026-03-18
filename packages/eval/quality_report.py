"""
Daily data-quality reporting utilities.
"""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Optional

import structlog

from packages.common.config import get_settings
from packages.common.database import get_connection

logger = structlog.get_logger()


def _format_datetime(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def generate_quality_report(
    target_date: date,
    min_games_played: Optional[int] = None,
) -> dict:
    """Generate a data-quality report for a slate date."""
    settings = get_settings()
    threshold = min_games_played if min_games_played is not None else settings.min_games_played

    with get_connection() as conn:
        ratings_as_of = conn.execute(
            "SELECT MAX(as_of_date) FROM team_strengths"
        ).fetchone()[0]

        rows = conn.execute(
            """
            WITH latest_ratings AS (
                SELECT *
                FROM team_strengths
                WHERE as_of_date = (SELECT MAX(as_of_date) FROM team_strengths)
            ),
            line_games AS (
                SELECT DISTINCT game_id FROM line_snapshots
                UNION
                SELECT DISTINCT game_id FROM betting_splits
            )
            SELECT
                g.game_id,
                g.game_datetime,
                g.home_team_name,
                g.away_team_name,
                hr.games_played AS home_games_played,
                ar.games_played AS away_games_played,
                CASE WHEN hr.team_id IS NULL THEN 1 ELSE 0 END AS missing_home_rating,
                CASE WHEN ar.team_id IS NULL THEN 1 ELSE 0 END AS missing_away_rating,
                CASE WHEN lg.game_id IS NULL THEN 1 ELSE 0 END AS missing_lines
            FROM games g
            LEFT JOIN latest_ratings hr ON g.home_team_id = hr.team_id
            LEFT JOIN latest_ratings ar ON g.away_team_id = ar.team_id
            LEFT JOIN line_games lg ON g.game_id = lg.game_id
            WHERE g.game_date = ?
            ORDER BY g.game_datetime, g.game_id
            """,
            (target_date.isoformat(),),
        ).fetchall()

    total_games = len(rows)
    missing_ratings = []
    low_sample = []
    missing_lines = []

    for row in rows:
        game_id = row[0]
        game_datetime = _format_datetime(row[1])
        home_name = row[2] or "Home"
        away_name = row[3] or "Away"
        home_games = row[4]
        away_games = row[5]
        missing_home = bool(row[6])
        missing_away = bool(row[7])
        missing_line = bool(row[8])

        matchup = f"{away_name} @ {home_name}"
        base_entry = {
            "game_id": game_id,
            "game_datetime": game_datetime,
            "matchup": matchup,
            "home_games_played": home_games,
            "away_games_played": away_games,
        }

        if missing_home or missing_away:
            entry = dict(base_entry)
            entry["missing_home_rating"] = missing_home
            entry["missing_away_rating"] = missing_away
            missing_ratings.append(entry)

        if threshold > 0 and not (missing_home or missing_away):
            min_games = min(int(home_games or 0), int(away_games or 0))
            if min_games < threshold:
                entry = dict(base_entry)
                entry["min_games_played"] = min_games
                low_sample.append(entry)

        if missing_line:
            missing_lines.append(base_entry)

    games_with_ratings = total_games - len(missing_ratings)
    games_with_lines = total_games - len(missing_lines)

    summary = {
        "total_games": total_games,
        "games_with_ratings": games_with_ratings,
        "missing_ratings_games": len(missing_ratings),
        "low_sample_games": len(low_sample),
        "games_with_lines": games_with_lines,
        "missing_lines_games": len(missing_lines),
        "ratings_coverage_rate": (games_with_ratings / total_games) if total_games else 0.0,
        "line_availability_rate": (games_with_lines / total_games) if total_games else 0.0,
        "low_sample_rate": (len(low_sample) / total_games) if total_games else 0.0,
    }

    return {
        "date": target_date.isoformat(),
        "generated_at": datetime.utcnow().isoformat(),
        "ratings_as_of_date": ratings_as_of.isoformat() if hasattr(ratings_as_of, "isoformat") else str(ratings_as_of),
        "min_games_played": threshold,
        "summary": summary,
        "details": {
            "missing_ratings": missing_ratings,
            "low_sample": low_sample,
            "missing_lines": missing_lines,
        },
    }


def write_quality_report(report: dict, output_path: Optional[Path] = None) -> Path:
    """Persist a quality report to disk."""
    settings = get_settings()
    if output_path is None:
        output_dir = Path(settings.quality_report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        date_tag = report.get("date", "").replace("-", "")
        output_path = output_dir / f"quality_report_{date_tag}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Quality report saved", path=str(output_path))
    return output_path
