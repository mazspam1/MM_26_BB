"""
Prediction guardrails for data quality.
"""

from __future__ import annotations

from typing import Optional

from packages.common.config import get_settings
from packages.common.schemas import PredictionRow


def apply_min_games_guardrail(
    pred_row: PredictionRow,
    home_games_played: int,
    away_games_played: int,
    min_games_played: Optional[int] = None,
) -> PredictionRow:
    """
    Downgrade recommendations when sample size is too small.

    If a recommendation exists but either team has fewer than the minimum
    games played, force a no-bet with low confidence.
    """
    if pred_row.recommended_side is None:
        return pred_row

    settings = get_settings()
    threshold = min_games_played if min_games_played is not None else settings.min_games_played
    if threshold <= 0:
        return pred_row

    if min(home_games_played, away_games_played) >= threshold:
        return pred_row

    return pred_row.model_copy(
        update={
            "recommended_side": "no_bet",
            "recommended_units": 0.0,
            "confidence_rating": "low",
        }
    )
