-- CBB Lines Database Schema
-- DuckDB schema for NCAA Basketball spread/total prediction system
-- All tables use append-only design for historical tracking where appropriate

-- =============================================================================
-- CORE ENTITIES
-- =============================================================================

-- Teams master table
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    abbreviation VARCHAR(10) NOT NULL,
    conference VARCHAR(50) NOT NULL,
    logo_url VARCHAR,
    color VARCHAR(10),
    alternate_color VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Team conference mapping (ESPN group IDs)
CREATE TABLE IF NOT EXISTS team_conference_ids (
    team_id INTEGER PRIMARY KEY,
    conference_id INTEGER,
    conference_name VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_team_conf_team FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Games fact table
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    game_date DATE NOT NULL,
    game_datetime TIMESTAMP,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_team_name VARCHAR,
    away_team_name VARCHAR,
    venue_type VARCHAR(10) NOT NULL DEFAULT 'home',
    neutral_site BOOLEAN DEFAULT FALSE,
    venue_name VARCHAR,
    home_score INTEGER,
    away_score INTEGER,
    status VARCHAR(20) DEFAULT 'scheduled',
    conference_game BOOLEAN DEFAULT FALSE,
    season_phase VARCHAR(20) DEFAULT 'non_conference',
    estimated_possessions FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_home_team FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    CONSTRAINT fk_away_team FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team_id);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team_id);

-- Team box scores
CREATE TABLE IF NOT EXISTS box_scores (
    game_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    is_home BOOLEAN NOT NULL,
    field_goals_made INTEGER NOT NULL,
    field_goals_attempted INTEGER NOT NULL,
    three_pointers_made INTEGER NOT NULL,
    three_pointers_attempted INTEGER NOT NULL,
    free_throws_made INTEGER NOT NULL,
    free_throws_attempted INTEGER NOT NULL,
    offensive_rebounds INTEGER NOT NULL,
    defensive_rebounds INTEGER NOT NULL,
    turnovers INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    steals INTEGER NOT NULL,
    blocks INTEGER NOT NULL,
    personal_fouls INTEGER NOT NULL,
    points INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, team_id),
    CONSTRAINT fk_box_game FOREIGN KEY (game_id) REFERENCES games(game_id),
    CONSTRAINT fk_box_team FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- =============================================================================
-- TIME-SERIES TABLES (Append-only snapshots)
-- =============================================================================

-- Team strength snapshots (daily recalculation)
CREATE TABLE IF NOT EXISTS team_strengths (
    team_id INTEGER NOT NULL,
    as_of_date VARCHAR NOT NULL,
    adj_offensive_efficiency FLOAT NOT NULL,
    adj_defensive_efficiency FLOAT NOT NULL,
    adj_tempo FLOAT NOT NULL,
    adj_em FLOAT,
    off_efg FLOAT,
    off_tov FLOAT,
    off_orb FLOAT,
    off_ftr FLOAT,
    def_efg FLOAT,
    def_tov FLOAT,
    def_drb FLOAT,
    def_ftr FLOAT,
    games_played INTEGER NOT NULL,
    sos_off FLOAT,
    sos_def FLOAT,
    home_off_delta FLOAT,
    home_def_delta FLOAT,
    away_off_delta FLOAT,
    away_def_delta FLOAT,
    home_games_played INTEGER,
    away_games_played INTEGER,
    off_rating_std FLOAT,
    def_rating_std FLOAT,
    tempo_std FLOAT,
    PRIMARY KEY (team_id, as_of_date),
    CONSTRAINT fk_strength_team FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_team_strengths_date ON team_strengths(as_of_date);

-- Line snapshots (append-only, timestamped odds)
CREATE TABLE IF NOT EXISTS line_snapshots (
    id INTEGER PRIMARY KEY,
    game_id INTEGER NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    snapshot_timestamp TIMESTAMP NOT NULL,
    snapshot_type VARCHAR(10) NOT NULL,  -- 'open', 'current', 'close'
    spread_home FLOAT,
    spread_home_price INTEGER,
    spread_away FLOAT,
    spread_away_price INTEGER,
    total_line FLOAT,
    over_price INTEGER,
    under_price INTEGER,
    home_ml INTEGER,
    away_ml INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_line_game FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_line_snapshots_game ON line_snapshots(game_id);
CREATE INDEX IF NOT EXISTS idx_line_snapshots_timestamp ON line_snapshots(snapshot_timestamp);
CREATE INDEX IF NOT EXISTS idx_line_snapshots_type ON line_snapshots(snapshot_type);

-- Create sequence for line_snapshots id
CREATE SEQUENCE IF NOT EXISTS seq_line_snapshots START 1;

-- Betting splits (public betting data from DraftKings)
CREATE TABLE IF NOT EXISTS betting_splits (
    id INTEGER PRIMARY KEY,
    game_id INTEGER NOT NULL,
    snapshot_timestamp TIMESTAMP NOT NULL,
    -- Line values from DraftKings
    spread_line FLOAT,  -- e.g., -31.5 (favored team spread)
    spread_line_home FLOAT,  -- e.g., +31.5 (positive = home favored)
    total_line FLOAT,   -- e.g., 149.5 (O/U total)
    -- Spread splits
    spread_favored_handle_pct FLOAT,
    spread_favored_bets_pct FLOAT,
    spread_underdog_handle_pct FLOAT,
    spread_underdog_bets_pct FLOAT,
    -- Total splits
    total_over_handle_pct FLOAT,
    total_over_bets_pct FLOAT,
    total_under_handle_pct FLOAT,
    total_under_bets_pct FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_splits_game FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_betting_splits_game ON betting_splits(game_id);
CREATE INDEX IF NOT EXISTS idx_betting_splits_timestamp ON betting_splits(snapshot_timestamp);

-- Create sequence for betting_splits id
CREATE SEQUENCE IF NOT EXISTS seq_betting_splits START 1;

-- =============================================================================
-- MODEL OUTPUTS
-- =============================================================================

-- Predictions output
CREATE TABLE IF NOT EXISTS predictions (
    game_id INTEGER NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    proj_home_score FLOAT NOT NULL,
    proj_away_score FLOAT NOT NULL,
    proj_spread FLOAT NOT NULL,
    proj_total FLOAT NOT NULL,
    proj_possessions FLOAT NOT NULL,
    home_win_prob FLOAT NOT NULL,
    efficiency_spread FLOAT,
    hca_adjustment FLOAT,
    travel_adjustment FLOAT,
    rest_adjustment FLOAT,
    four_factors_adjustment FLOAT,
    spread_ci_50_lower FLOAT,
    spread_ci_50_upper FLOAT,
    spread_ci_80_lower FLOAT,
    spread_ci_80_upper FLOAT,
    spread_ci_95_lower FLOAT,
    spread_ci_95_upper FLOAT,
    total_ci_50_lower FLOAT,
    total_ci_50_upper FLOAT,
    total_ci_80_lower FLOAT,
    total_ci_80_upper FLOAT,
    total_ci_95_lower FLOAT,
    total_ci_95_upper FLOAT,
    market_spread FLOAT,
    edge_vs_market_spread FLOAT,
    market_total FLOAT,
    edge_vs_market_total FLOAT,
    recommended_side VARCHAR(20),
    recommended_units FLOAT,
    confidence_rating VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, prediction_timestamp),
    CONSTRAINT fk_pred_game FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_version ON predictions(model_version);

-- Backtest predictions (append-only, run-scoped)
CREATE TABLE IF NOT EXISTS backtest_predictions (
    run_id VARCHAR NOT NULL,
    game_id INTEGER NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    proj_home_score FLOAT NOT NULL,
    proj_away_score FLOAT NOT NULL,
    proj_spread FLOAT NOT NULL,
    proj_total FLOAT NOT NULL,
    proj_possessions FLOAT NOT NULL,
    home_win_prob FLOAT NOT NULL,
    spread_ci_50_lower FLOAT,
    spread_ci_50_upper FLOAT,
    spread_ci_80_lower FLOAT,
    spread_ci_80_upper FLOAT,
    spread_ci_95_lower FLOAT,
    spread_ci_95_upper FLOAT,
    total_ci_50_lower FLOAT,
    total_ci_50_upper FLOAT,
    total_ci_80_lower FLOAT,
    total_ci_80_upper FLOAT,
    total_ci_95_lower FLOAT,
    total_ci_95_upper FLOAT,
    market_spread FLOAT,
    edge_vs_market_spread FLOAT,
    market_total FLOAT,
    edge_vs_market_total FLOAT,
    closing_spread FLOAT,
    closing_total FLOAT,
    spread_clv FLOAT,
    total_clv FLOAT,
    recommended_side VARCHAR(20),
    recommended_units FLOAT,
    confidence_rating VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, game_id),
    CONSTRAINT fk_backtest_game FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_backtest_predictions_run ON backtest_predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_predictions_game ON backtest_predictions(game_id);

-- Backtest run summaries
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    edge_threshold FLOAT DEFAULT 0.0,
    total_games INTEGER,
    games_with_lines INTEGER,
    spread_mae FLOAT,
    spread_rmse FLOAT,
    total_mae FLOAT,
    total_rmse FLOAT,
    spread_50_coverage FLOAT,
    spread_80_coverage FLOAT,
    spread_95_coverage FLOAT,
    mean_spread_clv FLOAT,
    mean_total_clv FLOAT,
    clv_positive_rate FLOAT,
    spread_wins INTEGER,
    spread_losses INTEGER,
    spread_pushes INTEGER,
    total_wins INTEGER,
    total_losses INTEGER,
    total_pushes INTEGER,
    simulated_roi FLOAT
);

-- Backtest segment summaries (diagnostics)
CREATE TABLE IF NOT EXISTS backtest_segments (
    run_id VARCHAR NOT NULL,
    segment_type VARCHAR NOT NULL,
    segment_value VARCHAR NOT NULL,
    total_games INTEGER,
    market_spread_count INTEGER,
    market_total_count INTEGER,
    closing_spread_count INTEGER,
    closing_total_count INTEGER,
    spread_mae FLOAT,
    spread_rmse FLOAT,
    total_mae FLOAT,
    total_rmse FLOAT,
    spread_50_coverage FLOAT,
    spread_80_coverage FLOAT,
    spread_95_coverage FLOAT,
    total_50_coverage FLOAT,
    total_80_coverage FLOAT,
    total_95_coverage FLOAT,
    mean_spread_clv FLOAT,
    mean_total_clv FLOAT,
    clv_positive_rate FLOAT,
    spread_wins INTEGER,
    spread_losses INTEGER,
    spread_pushes INTEGER,
    total_wins INTEGER,
    total_losses INTEGER,
    total_pushes INTEGER,
    simulated_roi FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, segment_type, segment_value)
);

CREATE INDEX IF NOT EXISTS idx_backtest_segments_run ON backtest_segments(run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_segments_type ON backtest_segments(segment_type);

-- CLV tracking
CREATE TABLE IF NOT EXISTS clv_reports (
    game_id INTEGER NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    bet_timestamp TIMESTAMP,
    our_spread FLOAT NOT NULL,
    our_total FLOAT NOT NULL,
    market_spread_at_bet FLOAT NOT NULL,
    market_total_at_bet FLOAT NOT NULL,
    closing_spread FLOAT NOT NULL,
    closing_total FLOAT NOT NULL,
    spread_clv FLOAT NOT NULL,
    total_clv FLOAT NOT NULL,
    actual_spread FLOAT,
    actual_total INTEGER,
    spread_bet_side VARCHAR(10),
    spread_bet_won BOOLEAN,
    total_bet_side VARCHAR(10),
    total_bet_won BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, prediction_timestamp),
    CONSTRAINT fk_clv_game FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_clv_game ON clv_reports(game_id);

-- =============================================================================
-- FEATURE ENGINEERING
-- =============================================================================

-- Feature snapshots for each game
CREATE TABLE IF NOT EXISTS feature_rows (
    game_id INTEGER NOT NULL,
    feature_timestamp TIMESTAMP NOT NULL,
    home_adj_off_eff FLOAT NOT NULL,
    home_adj_def_eff FLOAT NOT NULL,
    home_adj_tempo FLOAT NOT NULL,
    home_adj_em FLOAT NOT NULL,
    away_adj_off_eff FLOAT NOT NULL,
    away_adj_def_eff FLOAT NOT NULL,
    away_adj_tempo FLOAT NOT NULL,
    away_adj_em FLOAT NOT NULL,
    home_court_global FLOAT DEFAULT 3.5,
    home_court_team_effect FLOAT DEFAULT 0.0,
    is_neutral BOOLEAN DEFAULT FALSE,
    efg_diff FLOAT,
    to_diff FLOAT,
    orb_diff FLOAT,
    ftr_diff FLOAT,
    home_rest_days INTEGER,
    away_rest_days INTEGER,
    rest_advantage INTEGER,
    season_phase VARCHAR(20) DEFAULT 'non_conference',
    games_into_season_home INTEGER DEFAULT 0,
    games_into_season_away INTEGER DEFAULT 0,
    opening_spread FLOAT,
    current_spread FLOAT,
    spread_movement FLOAT,
    book_disagreement FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, feature_timestamp),
    CONSTRAINT fk_feature_game FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Latest team strengths view
CREATE OR REPLACE VIEW v_latest_team_strengths AS
SELECT ts.*
FROM team_strengths ts
INNER JOIN (
    SELECT team_id, MAX(as_of_date) as max_date
    FROM team_strengths
    GROUP BY team_id
) latest ON ts.team_id = latest.team_id AND ts.as_of_date = latest.max_date;

-- Latest predictions view
CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT p.*
FROM predictions p
INNER JOIN (
    SELECT game_id, MAX(prediction_timestamp) as max_ts
    FROM predictions
    GROUP BY game_id
) latest ON p.game_id = latest.game_id AND p.prediction_timestamp = latest.max_ts;

-- Games with latest predictions
CREATE OR REPLACE VIEW v_games_with_predictions AS
SELECT
    g.*,
    p.proj_spread,
    p.proj_total,
    p.home_win_prob,
    p.edge_vs_market_spread,
    p.edge_vs_market_total,
    p.recommended_side,
    p.recommended_units,
    p.model_version
FROM games g
LEFT JOIN v_latest_predictions p ON g.game_id = p.game_id;

-- Today's slate view
CREATE OR REPLACE VIEW v_todays_slate AS
SELECT * FROM v_games_with_predictions
WHERE game_date = CURRENT_DATE
ORDER BY game_datetime;
