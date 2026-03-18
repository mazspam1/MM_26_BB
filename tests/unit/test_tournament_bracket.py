"""Tests for live NCAA bracket slot normalization."""

from packages.ingest.tournament_bracket import (
    REGION_MIDWEST,
    ROUND_FIRST_FOUR,
    ROUND_OF_64,
    build_live_bracket_slots,
)


def test_build_live_bracket_slots_preserves_play_in_edges() -> None:
    official_contests = [
        {
            "bracketId": 101,
            "gameStateCode": 1,
            "gamestateDisplay": "",
            "statusCodeDisplay": "pre",
            "startDate": "03/17/2026",
            "victorGamePosition": "Bottom",
            "victorBracketPositionId": 225,
            "mmlVideo": False,
            "round": {"roundNumber": 1},
            "region": {"sectionId": 1, "position": "TT", "title": "", "abbreviation": ""},
            "teams": [
                {
                    "isTop": True,
                    "seed": 16,
                    "nameShort": "UMBC",
                    "seoname": "umbc",
                    "nickname": "Retrievers",
                    "isWinner": False,
                },
                {
                    "isTop": False,
                    "seed": 16,
                    "nameShort": "Howard",
                    "seoname": "howard",
                    "nickname": "Bison",
                    "isWinner": False,
                },
            ],
        },
        {
            "bracketId": 225,
            "gameStateCode": 1,
            "gamestateDisplay": "",
            "statusCodeDisplay": "pre",
            "startDate": "03/19/2026",
            "victorGamePosition": "Top",
            "victorBracketPositionId": 313,
            "mmlVideo": False,
            "round": {"roundNumber": 2},
            "region": {
                "sectionId": 5,
                "position": "BR",
                "title": " MIDWEST",
                "abbreviation": "MW ",
            },
            "teams": [
                {
                    "isTop": True,
                    "seed": 1,
                    "nameShort": "Michigan",
                    "seoname": "michigan",
                    "nickname": "Wolverines",
                    "isWinner": False,
                },
            ],
        },
    ]
    score_contests = [
        {
            "bracketId": 101,
            "contestId": 9001,
            "startDate": "03/17/2026",
            "location": {"venue": "UD Arena"},
            "victorGamePosition": "Bottom",
            "victorBracketPositionId": 225,
            "round": {"roundNumber": 1},
            "teams": official_contests[0]["teams"],
            "winnerOf": [],
        },
        {
            "bracketId": 225,
            "contestId": 9002,
            "startDate": "03/19/2026",
            "location": {"venue": "KeyBank Center"},
            "victorGamePosition": "Top",
            "victorBracketPositionId": 313,
            "round": {"roundNumber": 2},
            "teams": official_contests[1]["teams"],
            "winnerOf": [
                {
                    "bracketId": 101,
                    "homeSeed": 16,
                    "visitSeed": 16,
                    "isTop": False,
                }
            ],
        },
    ]

    slots = build_live_bracket_slots(
        year=2026,
        official_contests=official_contests,
        score_contests=score_contests,
        team_lookup={"umbc": 1, "howard": 2, "michigan": 3},
    )

    play_in = next(slot for slot in slots if slot.slot_id == 101)
    round_of_64 = next(slot for slot in slots if slot.slot_id == 225)

    assert play_in.round == ROUND_FIRST_FOUR
    assert play_in.next_slot_id == 225
    assert play_in.victor_game_position == "Bottom"

    assert round_of_64.round == ROUND_OF_64
    assert round_of_64.region_id == REGION_MIDWEST
    assert round_of_64.team_a_id == 3
    assert round_of_64.team_b_id is None
    assert round_of_64.seed_b == 16
