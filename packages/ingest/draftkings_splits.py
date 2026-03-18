"""
DraftKings betting splits scraper.

Fetches public betting splits data (% Handle and % Bets) for spreads and totals
from DraftKings Network betting splits pages.

URLs:
- Spread splits: https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=92483&tb_edate=n7days&tb_emt=Spread
- Total splits: https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=92483&tb_edate=n7days&tb_emt=Total
"""

from datetime import date, datetime
from typing import Optional
import re

import httpx
import structlog
from bs4 import BeautifulSoup

from packages.common.database import get_connection

logger = structlog.get_logger()


def normalize_team_name(name: str) -> str:
    """Normalize team names for matching across sources."""
    name = name.strip().lower()
    name = name.replace("&", "and")
    name = re.sub(r"\bst[.]?\b", "saint", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\b(state|university|college|tech|institute|of|the)\b", "", name, flags=re.I)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def team_name_keys(name: str) -> set[str]:
    """Generate matching keys for a team name (normalized, tokens, acronym)."""
    normalized = normalize_team_name(name)
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


class DraftKingsSplitsScraper:
    """Scraper for DraftKings betting splits data."""

    BASE_URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits"
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(self):
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=30.0,
                headers=self.DEFAULT_HEADERS,
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def _get_max_page(self, html: str) -> int:
        """
        Extract the maximum page number from pagination links.

        Args:
            html: HTML content from DraftKings page

        Returns:
            Maximum page number (1 if no pagination found)
        """
        soup = BeautifulSoup(html, "html.parser")

        # Find all pagination links with tb_page parameter
        page_links = soup.find_all("a", href=re.compile(r"tb_page=(\d+)"))

        max_page = 1
        for link in page_links:
            href = link.get("href", "")
            match = re.search(r"tb_page=(\d+)", href)
            if match:
                page_num = int(match.group(1))
                max_page = max(max_page, page_num)

        return max_page

    def fetch_spread_splits(self, days: int = 7) -> list[dict]:
        """
        Fetch spread betting splits from DraftKings (all pages).

        Args:
            days: Number of days to fetch (n7days, n30days, etc.)

        Returns:
            List of game splits dictionaries
        """
        url = f"{self.BASE_URL}/"
        base_params = {
            "tb_eg": "92483",  # NCAAB event group ID
            "tb_edate": f"n{days}days",
            "tb_emt": "Spread",
        }

        client = self._get_client()
        all_splits = []

        # Fetch first page to determine total pages
        logger.info("Fetching spread betting splits page 1", url=url, params=base_params)
        response = client.get(url, params=base_params)
        response.raise_for_status()

        max_page = self._get_max_page(response.text)
        logger.info(f"Found {max_page} pages of spread splits")

        # Parse first page
        all_splits.extend(self._parse_splits_html(response.text, split_type="spread"))

        # Fetch remaining pages
        for page in range(2, max_page + 1):
            params = {**base_params, "tb_page": str(page)}
            logger.info(f"Fetching spread betting splits page {page}/{max_page}")
            response = client.get(url, params=params)
            response.raise_for_status()
            all_splits.extend(self._parse_splits_html(response.text, split_type="spread"))

        logger.info(f"Total spread splits fetched: {len(all_splits)}")
        return all_splits

    def fetch_total_splits(self, days: int = 7) -> list[dict]:
        """
        Fetch total (O/U) betting splits from DraftKings (all pages).

        Args:
            days: Number of days to fetch

        Returns:
            List of game splits dictionaries
        """
        url = f"{self.BASE_URL}/"
        base_params = {
            "tb_eg": "92483",  # NCAAB event group ID
            "tb_edate": f"n{days}days",
            "tb_emt": "Total",
        }

        client = self._get_client()
        all_splits = []

        # Fetch first page to determine total pages
        logger.info("Fetching total betting splits page 1", url=url, params=base_params)
        response = client.get(url, params=base_params)
        response.raise_for_status()

        max_page = self._get_max_page(response.text)
        logger.info(f"Found {max_page} pages of total splits")

        # Parse first page
        all_splits.extend(self._parse_splits_html(response.text, split_type="total"))

        # Fetch remaining pages
        for page in range(2, max_page + 1):
            params = {**base_params, "tb_page": str(page)}
            logger.info(f"Fetching total betting splits page {page}/{max_page}")
            response = client.get(url, params=params)
            response.raise_for_status()
            all_splits.extend(self._parse_splits_html(response.text, split_type="total"))

        logger.info(f"Total O/U splits fetched: {len(all_splits)}")
        return all_splits

    def _parse_splits_html(self, html: str, split_type: str) -> list[dict]:
        """
        Parse HTML to extract betting splits from DraftKings.

        Structure:
        - tb-se: game section
          - tb-se-title: contains h5 (game name "Away @ Home") and span (date/time)
          - tb-market-wrap: contains the betting data
            - tb-sodd: each line (favored/underdog or over/under)
              - tb-slipline: team + spread/total line
              - div: odds
              - div: handle %
              - div: bets %

        Args:
            html: HTML content from DraftKings page
            split_type: "spread" or "total"

        Returns:
            List of splits dictionaries
        """
        soup = BeautifulSoup(html, "html.parser")
        splits = []

        # Find all game sections (tb-se divs)
        game_sections = soup.find_all(class_="tb-se")
        logger.debug(f"Found {len(game_sections)} game sections")

        for section in game_sections:
            try:
                # Extract game info from tb-se-title
                title_div = section.find(class_="tb-se-title")
                if not title_div:
                    continue

                h5 = title_div.find("h5")
                span = title_div.find("span")

                if not h5:
                    continue

                game_text = h5.get_text(strip=True)
                date_time_text = span.get_text(strip=True) if span else ""

                # Parse game info
                game_info = self._extract_game_info_from_text(f"{game_text} {date_time_text}")
                if not game_info:
                    continue

                # Find market data in tb-market-wrap
                market_wrap = section.find(class_="tb-market-wrap")
                if not market_wrap:
                    continue

                # Find all tb-sodd divs (each contains one side of the bet)
                odds_divs = market_wrap.find_all(class_="tb-sodd")
                if len(odds_divs) < 2:
                    continue

                if split_type == "spread":
                    split_data = self._extract_spread_splits_from_odds_divs(odds_divs)
                else:
                    split_data = self._extract_total_splits_from_odds_divs(odds_divs)

                if split_data and game_info:
                    splits.append({**game_info, **split_data})

            except Exception as e:
                logger.warning("Failed to parse game section", error=str(e))
                continue

        logger.info(f"Parsed {len(splits)} {split_type} splits")
        return splits

    def _extract_game_info_from_text(self, text: str) -> Optional[dict]:
        """Extract game information from text string."""
        # Parse "Away Team @ Home Team" or "Away Team @ Home Team Date, Time"
        # Example: "South Carolina State @ South Carolina 12/22, 04:05PM"
        # Or: "South Carolina State @ South Carolina\n12/22, 04:05PM"
        text = " ".join(text.split())  # Normalize whitespace

        # First, try to extract date/time from the end
        date_str = None
        time_str = None
        date_time_match = re.search(r"\s+(\d{1,2}/\d{1,2}),?\s+(\d{1,2}:\d{2}(?:AM|PM))\s*$", text, re.I)
        if date_time_match:
            date_str = date_time_match.group(1)
            time_str = date_time_match.group(2)
            # Remove the date/time from text to parse team names cleanly
            text = text[: date_time_match.start()].strip()

        # Now parse "Away Team @ Home Team"
        if " @ " not in text:
            return None

        parts = text.split(" @ ", 1)
        if len(parts) != 2:
            return None

        away_team = parts[0].strip()
        home_team = parts[1].strip()

        if not away_team or not home_team:
            return None

        return {
            "away_team": away_team,
            "home_team": home_team,
            "date_str": date_str,
            "time_str": time_str,
        }

    def _extract_spread_splits_from_odds_divs(self, odds_divs) -> Optional[dict]:
        """
        Extract spread betting splits from tb-sodd divs.

        Structure of each tb-sodd div:
        - tb-slipline: team + spread (e.g., "South Carolina -31.5")
        - div: odds
        - div: handle % (e.g., "53%")
        - div: bets % (e.g., "55%")
        """
        splits_data = {}

        for odds_div in odds_divs:
            try:
                # Get the spread line
                slipline = odds_div.find(class_="tb-slipline")
                if not slipline:
                    continue

                slipline_text = slipline.get_text(strip=True)

                team_name = None
                team_match = re.match(r"^(.*?)([+-]?\d+\.?\d*)$", slipline_text)
                if team_match:
                    team_name = team_match.group(1).strip()

                # Extract spread value (look for +/- number at end)
                spread_match = re.search(r"([+-]?\d+\.?\d*)$", slipline_text.replace(" ", ""))
                if not spread_match:
                    # Try alternative pattern
                    spread_match = re.search(r"([+-]\d+\.?\d*)", slipline_text)
                if not spread_match:
                    continue

                spread_val = float(spread_match.group(1))
                is_favored = spread_val < 0  # Negative spread = favored

                # Store the spread line (use the favored team's spread, which is negative)
                if is_favored and "spread_line" not in splits_data:
                    splits_data["spread_line"] = spread_val
                if team_name:
                    if is_favored and "spread_favored_team" not in splits_data:
                        splits_data["spread_favored_team"] = team_name
                    elif not is_favored and "spread_underdog_team" not in splits_data:
                        splits_data["spread_underdog_team"] = team_name

                # Get all child divs after slipline
                child_divs = [d for d in odds_div.find_all("div", recursive=False)
                              if "tb-slipline" not in d.get("class", [])]

                # Extract percentages from the divs
                percentages = []
                for div in child_divs:
                    div_text = div.get_text(strip=True)
                    pct_match = re.search(r"(\d+)%", div_text)
                    if pct_match:
                        percentages.append(float(pct_match.group(1)))

                if len(percentages) >= 2:
                    handle_pct = percentages[0]
                    bets_pct = percentages[1]

                    if is_favored:
                        splits_data["spread_favored_handle_pct"] = handle_pct
                        splits_data["spread_favored_bets_pct"] = bets_pct
                    else:
                        splits_data["spread_underdog_handle_pct"] = handle_pct
                        splits_data["spread_underdog_bets_pct"] = bets_pct

            except Exception as e:
                logger.debug(f"Failed to parse odds div: {e}")
                continue

        return splits_data if splits_data else None

    def _extract_total_splits_from_odds_divs(self, odds_divs) -> Optional[dict]:
        """
        Extract total (O/U) betting splits from tb-sodd divs.

        Structure of each tb-sodd div:
        - tb-slipline: over/under + total (e.g., "Over 140.5" or "Under 140.5")
        - div: odds
        - div: handle %
        - div: bets %
        """
        splits_data = {}

        for odds_div in odds_divs:
            try:
                # Get the total line
                slipline = odds_div.find(class_="tb-slipline")
                if not slipline:
                    continue

                slipline_text = slipline.get_text(strip=True).lower()

                # Check if it's over or under
                is_over = "over" in slipline_text or slipline_text.startswith("o ")
                is_under = "under" in slipline_text or slipline_text.startswith("u ")

                if not (is_over or is_under):
                    continue

                # Extract the total line value (e.g., 149.5 from "Over 149.5")
                total_match = re.search(r"(\d+\.?\d*)", slipline_text)
                if total_match and "total_line" not in splits_data:
                    splits_data["total_line"] = float(total_match.group(1))

                # Get all child divs after slipline
                child_divs = [d for d in odds_div.find_all("div", recursive=False)
                              if "tb-slipline" not in d.get("class", [])]

                # Extract percentages from the divs
                percentages = []
                for div in child_divs:
                    div_text = div.get_text(strip=True)
                    pct_match = re.search(r"(\d+)%", div_text)
                    if pct_match:
                        percentages.append(float(pct_match.group(1)))

                if len(percentages) >= 2:
                    handle_pct = percentages[0]
                    bets_pct = percentages[1]

                    if is_over:
                        splits_data["total_over_handle_pct"] = handle_pct
                        splits_data["total_over_bets_pct"] = bets_pct
                    else:
                        splits_data["total_under_handle_pct"] = handle_pct
                        splits_data["total_under_bets_pct"] = bets_pct

            except Exception as e:
                logger.debug(f"Failed to parse odds div: {e}")
                continue

        return splits_data if splits_data else None

    def match_game_to_splits(
        self, splits_data: list[dict], home_team_name: str, away_team_name: str, game_date: date
    ) -> Optional[dict]:
        """
        Match betting splits data to a game by team names and date.

        Args:
            splits_data: List of splits dictionaries from scraper
            home_team_name: Home team name from database
            away_team_name: Away team name from database
            game_date: Game date

        Returns:
            Matched splits dictionary or None
        """
        # Normalize team names for matching
        home_keys = team_name_keys(home_team_name)
        away_keys = team_name_keys(away_team_name)

        for split in splits_data:
            split_home_keys = team_name_keys(split.get("home_team", ""))
            split_away_keys = team_name_keys(split.get("away_team", ""))

            # Try to match by team names
            if _keys_match(home_keys, split_home_keys) and _keys_match(away_keys, split_away_keys):
                # Also check date if available
                if split.get("date_str"):
                    # Parse date string (MM/DD format)
                    try:
                        month, day = split["date_str"].split("/")
                        split_date = date(game_date.year, int(month), int(day))
                        if abs((split_date - game_date).days) <= 1:  # Allow 1 day difference
                            return split
                    except:
                        pass

                # If date matches or not provided, return match
                return split

        return None


def save_betting_splits_to_db(game_id: int, splits_data: dict) -> bool:
    """
    Save betting splits to database.

    Args:
        game_id: Game ID
        splits_data: Dictionary with splits data

    Returns:
        True if saved successfully
    """
    with get_connection() as conn:
        try:
            conn.execute("ALTER TABLE betting_splits ADD COLUMN spread_line_home FLOAT")
        except Exception:
            pass
        conn.execute(
            """
            INSERT INTO betting_splits (
                id, game_id, snapshot_timestamp,
                spread_line, spread_line_home, total_line,
                spread_favored_handle_pct, spread_favored_bets_pct,
                spread_underdog_handle_pct, spread_underdog_bets_pct,
                total_over_handle_pct, total_over_bets_pct,
                total_under_handle_pct, total_under_bets_pct
            ) VALUES (
                nextval('seq_betting_splits'), ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                game_id,
                datetime.utcnow().isoformat(),
                splits_data.get("spread_line"),
                splits_data.get("spread_line_home"),
                splits_data.get("total_line"),
                splits_data.get("spread_favored_handle_pct"),
                splits_data.get("spread_favored_bets_pct"),
                splits_data.get("spread_underdog_handle_pct"),
                splits_data.get("spread_underdog_bets_pct"),
                splits_data.get("total_over_handle_pct"),
                splits_data.get("total_over_bets_pct"),
                splits_data.get("total_under_handle_pct"),
                splits_data.get("total_under_bets_pct"),
            ),
        )

    logger.info("Betting splits saved to database", game_id=game_id)
    return True


def get_latest_betting_splits(game_id: int) -> Optional[dict]:
    """
    Get latest betting splits for a game.

    Args:
        game_id: Game ID

    Returns:
        Dictionary with splits data or None
    """
    with get_connection() as conn:
        result = conn.execute(
            """
            SELECT
                spread_favored_handle_pct, spread_favored_bets_pct,
                spread_underdog_handle_pct, spread_underdog_bets_pct,
                total_over_handle_pct, total_over_bets_pct,
                total_under_handle_pct, total_under_bets_pct
            FROM betting_splits
            WHERE game_id = ?
            ORDER BY snapshot_timestamp DESC
            LIMIT 1
            """,
            (game_id,),
        ).fetchone()

    if result and result[0] is not None:
        return {
            "spread_favored_handle_pct": result[0],
            "spread_favored_bets_pct": result[1],
            "spread_underdog_handle_pct": result[2],
            "spread_underdog_bets_pct": result[3],
            "total_over_handle_pct": result[4],
            "total_over_bets_pct": result[5],
            "total_under_handle_pct": result[6],
            "total_under_bets_pct": result[7],
        }

    return None

