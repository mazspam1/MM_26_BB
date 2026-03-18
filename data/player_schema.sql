-- Player data schema for March Madness modeling

-- Player basic info
CREATE TABLE IF NOT EXISTS players (
    player_id VARCHAR PRIMARY KEY,
    team_id INTEGER,
    team_name VARCHAR,
    first_name VARCHAR,
    last_name VARCHAR,
    display_name VARCHAR,
    position VARCHAR,
    jersey VARCHAR,
    height_inches INTEGER,
    weight_lbs INTEGER,
    class_year VARCHAR,
    is_active BOOLEAN DEFAULT true,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player game-by-game statistics
CREATE TABLE IF NOT EXISTS player_game_stats (
    id INTEGER PRIMARY KEY,
    player_id VARCHAR,
    game_id VARCHAR,
    team_id INTEGER,
    opponent_id INTEGER,
    game_date DATE,
    minutes_played FLOAT,
    points INTEGER,
    field_goals_made INTEGER,
    field_goals_attempted INTEGER,
    three_pointers_made INTEGER,
    three_pointers_attempted INTEGER,
    free_throws_made INTEGER,
    free_throws_attempted INTEGER,
    offensive_rebounds INTEGER,
    defensive_rebounds INTEGER,
    total_rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    personal_fouls INTEGER,
    plus_minus INTEGER,
    starter BOOLEAN,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player season aggregates
CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id VARCHAR,
    season INTEGER,
    team_id INTEGER,
    games_played INTEGER,
    games_started INTEGER,
    minutes_per_game FLOAT,
    points_per_game FLOAT,
    rebounds_per_game FLOAT,
    assists_per_game FLOAT,
    steals_per_game FLOAT,
    blocks_per_game FLOAT,
    turnovers_per_game FLOAT,
    field_goal_pct FLOAT,
    three_point_pct FLOAT,
    free_throw_pct FLOAT,
    usage_rate FLOAT,
    effective_fg_pct FLOAT,
    true_shooting_pct FLOAT,
    offensive_rating FLOAT,
    defensive_rating FLOAT,
    player_efficiency_rating FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, season)
);

-- Team player composition (for matchup analysis)
CREATE TABLE IF NOT EXISTS team_player_summary (
    team_id INTEGER,
    team_name VARCHAR,
    season INTEGER,
    n_players INTEGER,
    n_scholarship INTEGER,
    avg_height_inches FLOAT,
    avg_weight_lbs FLOAT,
    avg_class_year FLOAT,
    top_scorer_id VARCHAR,
    top_scorer_ppg FLOAT,
    top_rebounder_id VARCHAR,
    top_rebounder_rpg FLOAT,
    top_assists_id VARCHAR,
    top_assists_apg FLOAT,
    three_point_heavy BOOLEAN,
    interior_presence BOOLEAN,
    depth_rating FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (team_id, season)
);
