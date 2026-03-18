"""
The Odds API connector for NCAA Basketball odds.

Free tier: 500 requests/month
Strategy: 5 snapshots/day (150/month) + 50 closing lines + 50 buffer = 250 requests

API docs: https://the-odds-api.com/liveapi/guides/v4/
"""

from datetime import date, datetime, timedelta
from typing import Literal, Optional

import asyncio
import re
import httpx
import structlog
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from packages.common.database import get_connection
from packages.common.schemas import LineSnapshot

logger = structlog.get_logger()


def _normalize_team_name(name: str) -> str:
    """Normalize team names for cross-source matching."""
    name = name.strip().lower()
    name = name.replace("&", "and")
    name = re.sub(r"\bst[.]?\b", "saint", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\b(state|university|college|tech|institute|of|the)\b", "", name, flags=re.I)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _team_name_keys(name: str) -> set[str]:
    """Generate matching keys for a team name (normalized, tokens, acronym)."""
    normalized = _normalize_team_name(name)
    tokens = [token for token in normalized.split() if token]
    keys = {normalized}
    keys.update(token for token in tokens if len(token) >= 3)
    acronym = "".join(token[0] for token in tokens if token)
    if len(acronym) >= 3:
        keys.add(acronym)
    return keys


def _keys_match(left: set[str], right: set[str]) -> bool:
    """Return True if any key overlaps or is a substring match."""
    if left & right:
        return True
    for left_key in left:
        for right_key in right:
            if left_key in right_key or right_key in left_key:
                return True
    return False


class OddsAPISettings(BaseSettings):
    """The Odds API configuration from environment."""

    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class OddsAPIQuota(BaseModel):
    """Track API quota usage."""

    requests_remaining: int
    requests_used: int


class OddsAPIConnector:
    """
    Connector for The Odds API.

    Handles:
    - Fetching NCAAB odds (spreads, totals, moneylines)
    - Quota tracking via response headers
    - Rate limiting protection
    """

    SPORT_KEY = "basketball_ncaab"
    DEFAULT_REGIONS = "us"
    DEFAULT_MARKETS = "h2h,spreads,totals"
    DEFAULT_ODDS_FORMAT = "american"

    def __init__(self, settings: Optional[OddsAPISettings] = None):
        self.settings = settings or OddsAPISettings()
        self._client: Optional[httpx.AsyncClient] = None
        self._quota: Optional[OddsAPIQuota] = None
        self._game_cache: dict = {}

        if not self.settings.odds_api_key:
            logger.warning("ODDS_API_KEY not configured - odds fetching will fail")

    @property
    def quota(self) -> Optional[OddsAPIQuota]:
        """Current API quota status."""
        return self._quota

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": "cbb-lines/0.1.0"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _update_quota(self, response: httpx.Response) -> None:
        """Update quota tracking from response headers."""
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")

        if remaining is not None:
            self._quota = OddsAPIQuota(
                requests_remaining=int(remaining),
                requests_used=int(used) if used else 0,
            )

            if self._quota.requests_remaining < 100:
                logger.warning(
                    "Odds API quota running low",
                    remaining=self._quota.requests_remaining,
                    used=self._quota.requests_used,
                )
            else:
                logger.debug(
                    "Odds API quota",
                    remaining=self._quota.requests_remaining,
                    used=self._quota.requests_used,
                )

    async def fetch_ncaab_odds(
        self,
        regions: str = DEFAULT_REGIONS,
        markets: str = DEFAULT_MARKETS,
        odds_format: str = DEFAULT_ODDS_FORMAT,
    ) -> list[dict]:
        """
        Fetch current NCAAB odds.

        Args:
            regions: Comma-separated region codes (e.g., "us", "us,uk")
            markets: Comma-separated market types (h2h, spreads, totals)
            odds_format: "american" or "decimal"

        Returns:
            Raw API response as list of game dictionaries

        Raises:
            httpx.HTTPStatusError: If API returns error status
            ValueError: If API key not configured
        """
        if not self.settings.odds_api_key:
            raise ValueError("ODDS_API_KEY not configured")

        client = await self._get_client()

        url = f"{self.settings.odds_api_base_url}/sports/{self.SPORT_KEY}/odds"
        params = {
            "apiKey": self.settings.odds_api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }

        logger.info(
            "Fetching NCAAB odds from API",
            sport=self.SPORT_KEY,
            regions=regions,
            markets=markets,
        )

        response = await client.get(url, params=params)

        # Update quota tracking
        self._update_quota(response)

        # Raise for HTTP errors
        response.raise_for_status()

        data = response.json()
        logger.info("Odds fetched successfully", games_count=len(data))

        return data

    def parse_odds_response(
        self,
        raw_data: list[dict],
        snapshot_type: Literal["open", "current", "close"] = "current",
    ) -> list[LineSnapshot]:
        """
        Parse raw API response into LineSnapshot objects.

        Args:
            raw_data: Raw API response from fetch_ncaab_odds
            snapshot_type: Type of snapshot (open, current, close)

        Returns:
            List of LineSnapshot objects
        """
        snapshots: list[LineSnapshot] = []
        now = datetime.utcnow()

        for game in raw_data:
            try:
                # Get game ID from ESPN (stored in game["id"])
                # Note: The Odds API uses its own IDs, we need to match by teams
                game_id_str = game.get("id", "")

                # Try to extract ESPN game ID if available
                # Otherwise we'll need to match by team names
                game_id = self._extract_game_id(game)

                if game_id == 0:
                    logger.debug(
                        "Could not extract game ID, skipping",
                        home=game.get("home_team"),
                        away=game.get("away_team"),
                    )
                    continue

                # Parse each bookmaker's odds
                for bookmaker in game.get("bookmakers", []):
                    bookmaker_name = bookmaker.get("key", "unknown")

                    snapshot = self._parse_bookmaker_odds(
                        game_id=game_id,
                        bookmaker=bookmaker_name,
                        bookmaker_data=bookmaker,
                        timestamp=now,
                        snapshot_type=snapshot_type,
                        home_team=game.get("home_team", ""),
                        away_team=game.get("away_team", ""),
                    )

                    if snapshot:
                        snapshots.append(snapshot)

            except Exception as e:
                logger.warning("Failed to parse game odds", error=str(e), game=game)
                continue

        return snapshots

    def _get_games_for_date(self, game_date: date) -> list[tuple[int, set[str], set[str]]]:
        """Load and cache games for a date with normalized keys."""
        if game_date in self._game_cache:
            return self._game_cache[game_date]

        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT game_id, home_team_name, away_team_name
                FROM games
                WHERE game_date = ?
                """,
                (game_date.isoformat(),),
            ).fetchall()

        games = []
        for game_id, home_name, away_name in rows:
            games.append((game_id, _team_name_keys(home_name), _team_name_keys(away_name)))

        self._game_cache[game_date] = games
        return games

    def _extract_game_id(self, game: dict) -> int:
        """
        Match The Odds API games to ESPN game IDs using team names and date.
        """
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence_time = game.get("commence_time", "")

        if not home_team or not away_team or not commence_time:
            return 0

        try:
            game_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).date()
        except ValueError:
            return 0

        home_keys = _team_name_keys(home_team)
        away_keys = _team_name_keys(away_team)

        candidate_dates = [
            game_date,
            game_date - timedelta(days=1),
            game_date + timedelta(days=1),
        ]

        for candidate_date in candidate_dates:
            for game_id, db_home_keys, db_away_keys in self._get_games_for_date(candidate_date):
                if _keys_match(home_keys, db_home_keys) and _keys_match(away_keys, db_away_keys):
                    return game_id
                if _keys_match(home_keys, db_away_keys) and _keys_match(away_keys, db_home_keys):
                    return game_id

        return 0

    def _parse_bookmaker_odds(
        self,
        game_id: int,
        bookmaker: str,
        bookmaker_data: dict,
        timestamp: datetime,
        snapshot_type: Literal["open", "current", "close"],
        home_team: str,
        away_team: str,
    ) -> Optional[LineSnapshot]:
        """Parse a single bookmaker's odds into a LineSnapshot."""
        spread_home: Optional[float] = None
        spread_home_price: Optional[int] = None
        spread_away: Optional[float] = None
        spread_away_price: Optional[int] = None
        total_line: Optional[float] = None
        over_price: Optional[int] = None
        under_price: Optional[int] = None
        home_ml: Optional[int] = None
        away_ml: Optional[int] = None

        home_key = _normalize_team_name(home_team)
        away_key = _normalize_team_name(away_team)

        for market in bookmaker_data.get("markets", []):
            market_key = market.get("key")
            outcomes = market.get("outcomes", [])

            if market_key == "spreads":
                for outcome in outcomes:
                    name = outcome.get("name", "")
                    point = outcome.get("point")
                    price = outcome.get("price")

                    name_key = _normalize_team_name(name)
                    name_lower = name.strip().lower()
                    is_home = None
                    if name_key and name_key == home_key:
                        is_home = True
                    elif name_key and name_key == away_key:
                        is_home = False
                    elif name_lower == "home":
                        is_home = True
                    elif name_lower == "away":
                        is_home = False
                    elif outcome.get("isHome") is True:
                        is_home = True
                    elif outcome.get("isHome") is False:
                        is_home = False

                    if is_home is True:
                        spread_home = float(point) if point is not None else None
                        spread_home_price = int(price) if price is not None else None
                    elif is_home is False:
                        spread_away = float(point) if point is not None else None
                        spread_away_price = int(price) if price is not None else None

            elif market_key == "totals":
                for outcome in outcomes:
                    name = outcome.get("name", "").lower()
                    point = outcome.get("point")
                    price = outcome.get("price")

                    if point is not None:
                        total_line = float(point)

                    if "over" in name:
                        over_price = int(price) if price is not None else None
                    elif "under" in name:
                        under_price = int(price) if price is not None else None

            elif market_key == "h2h":
                for outcome in outcomes:
                    name = outcome.get("name", "")
                    price = outcome.get("price")

                    name_key = _normalize_team_name(name)
                    name_lower = name.strip().lower()
                    is_home = None
                    if name_key and name_key == home_key:
                        is_home = True
                    elif name_key and name_key == away_key:
                        is_home = False
                    elif name_lower == "home":
                        is_home = True
                    elif name_lower == "away":
                        is_home = False
                    elif outcome.get("isHome") is True:
                        is_home = True
                    elif outcome.get("isHome") is False:
                        is_home = False

                    if is_home is True:
                        home_ml = int(price) if price is not None else None
                    elif is_home is False:
                        away_ml = int(price) if price is not None else None

        # Only create snapshot if we have at least some data
        if spread_home is None and total_line is None and home_ml is None:
            return None

        return LineSnapshot(
            game_id=game_id,
            bookmaker=bookmaker,
            snapshot_timestamp=timestamp,
            snapshot_type=snapshot_type,
            spread_home=spread_home,
            spread_home_price=spread_home_price,
            spread_away=spread_away,
            spread_away_price=spread_away_price,
            total_line=total_line,
            over_price=over_price,
            under_price=under_price,
            home_ml=home_ml,
            away_ml=away_ml,
        )


def save_line_snapshots_to_db(snapshots: list[LineSnapshot]) -> int:
    """
    Save line snapshots to database (append-only).

    Args:
        snapshots: List of LineSnapshot objects

    Returns:
        Number of snapshots saved
    """
    if not snapshots:
        return 0

    with get_connection() as conn:
        for snap in snapshots:
            next_id_row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM line_snapshots"
            ).fetchone()
            next_id = int(next_id_row[0]) if next_id_row is not None else 1
            conn.execute(
                """
                INSERT INTO line_snapshots (
                    id, game_id, bookmaker, snapshot_timestamp, snapshot_type,
                    spread_home, spread_home_price, spread_away, spread_away_price,
                    total_line, over_price, under_price, home_ml, away_ml
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    next_id,
                    snap.game_id,
                    snap.bookmaker,
                    snap.snapshot_timestamp.isoformat(),
                    snap.snapshot_type,
                    snap.spread_home,
                    snap.spread_home_price,
                    snap.spread_away,
                    snap.spread_away_price,
                    snap.total_line,
                    snap.over_price,
                    snap.under_price,
                    snap.home_ml,
                    snap.away_ml,
                ),
            )

    logger.info("Line snapshots saved to database", count=len(snapshots))
    return len(snapshots)


def fetch_ncaab_odds(
    snapshot_type: Literal["open", "current", "close"] = "current",
) -> list[LineSnapshot]:
    """Fetch and parse current NCAAB odds into LineSnapshot objects."""
    connector = OddsAPIConnector()

    async def _run():
        data = await connector.fetch_ncaab_odds()
        await connector.close()
        return data

    raw_data = asyncio.run(_run())
    return connector.parse_odds_response(raw_data, snapshot_type=snapshot_type)


def save_odds_snapshot(snapshots: list[LineSnapshot]) -> int:
    """Persist a parsed odds snapshot."""
    return save_line_snapshots_to_db(snapshots)


def get_consensus_line(game_id: int) -> Optional[dict]:
    """
    Get consensus (average) line across bookmakers for a game.

    Args:
        game_id: ESPN game ID

    Returns:
        Dictionary with consensus spread and total, or None if no lines
    """
    with get_connection() as conn:
        result = conn.execute(
            """
            SELECT
                AVG(spread_home) as consensus_spread,
                AVG(total_line) as consensus_total,
                STDDEV(spread_home) as spread_disagreement,
                STDDEV(total_line) as total_disagreement,
                COUNT(DISTINCT bookmaker) as book_count
            FROM line_snapshots
            WHERE game_id = ?
            AND snapshot_timestamp = (
                SELECT MAX(snapshot_timestamp)
                FROM line_snapshots
                WHERE game_id = ?
            )
            """,
            (game_id, game_id),
        ).fetchone()

    if result and result[0] is not None:
        return {
            "consensus_spread": result[0],
            "consensus_total": result[1],
            "spread_disagreement": result[2],
            "total_disagreement": result[3],
            "book_count": result[4],
        }

    return None
