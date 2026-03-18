-- Tournament-specific schema additions for March Madness
-- Tables only - views created separately after all tables exist

-- Tournament years
CREATE TABLE IF NOT EXISTS tournament_years (
    year INTEGER PRIMARY KEY,
    start_date DATE NOT NULL,
    championship_date DATE NOT NULL,
    num_teams INTEGER DEFAULT 68,
    status VARCHAR(20) DEFAULT 'upcoming',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tournament regions
CREATE TABLE IF NOT EXISTS tournament_regions (
    region_id INTEGER PRIMARY KEY,
    year INTEGER NOT NULL,
    region_name VARCHAR(50) NOT NULL,
    display_order INTEGER DEFAULT 0
);

-- Tournament bracket slots
CREATE TABLE IF NOT EXISTS tournament_bracket (
    slot_id INTEGER PRIMARY KEY,
    year INTEGER NOT NULL,
    region_id INTEGER,
    round INTEGER NOT NULL,
    game_in_round INTEGER NOT NULL,
    seed_a INTEGER,
    seed_b INTEGER,
    team_a_id INTEGER,
    team_b_id INTEGER,
    winner_team_id INTEGER,
    game_id INTEGER,
    game_date DATE,
    venue VARCHAR(100),
    is_first_four BOOLEAN DEFAULT FALSE,
    next_slot_id INTEGER,
    victor_game_position VARCHAR(10)
);

CREATE INDEX IF NOT EXISTS idx_bracket_year ON tournament_bracket(year);
CREATE INDEX IF NOT EXISTS idx_bracket_round ON tournament_bracket(round);

-- Tournament predictions
CREATE TABLE IF NOT EXISTS tournament_predictions (
    prediction_id VARCHAR PRIMARY KEY,
    slot_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    round INTEGER NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    higher_seed INTEGER,
    lower_seed INTEGER,
    higher_seed_team_id INTEGER,
    lower_seed_team_id INTEGER,
    proj_higher_score FLOAT NOT NULL,
    proj_lower_score FLOAT NOT NULL,
    proj_spread FLOAT NOT NULL,
    proj_total FLOAT NOT NULL,
    proj_possessions FLOAT,
    higher_seed_win_prob FLOAT NOT NULL,
    upset_prob FLOAT NOT NULL,
    cover_prob FLOAT,
    spread_std FLOAT,
    total_std FLOAT,
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
    market_total FLOAT,
    edge_vs_market_spread FLOAT,
    edge_vs_market_total FLOAT,
    tournament_adjustment FLOAT,
    seed_momentum_factor FLOAT,
    fatigue_factor FLOAT,
    recommended_side VARCHAR(20),
    recommended_units FLOAT,
    confidence_rating VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tourn_pred_year ON tournament_predictions(year);
CREATE INDEX IF NOT EXISTS idx_tourn_pred_slot ON tournament_predictions(slot_id);

-- Bracket simulations
CREATE TABLE IF NOT EXISTS bracket_simulations (
    sim_id VARCHAR PRIMARY KEY,
    year INTEGER NOT NULL,
    num_simulations INTEGER NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    simulation_timestamp TIMESTAMP NOT NULL,
    total_runtime_ms INTEGER,
    gpu_used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Team advancement probabilities
CREATE TABLE IF NOT EXISTS team_advancement_probs (
    sim_id VARCHAR NOT NULL,
    team_id INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    region_id INTEGER,
    prob_round_of_64 FLOAT,
    prob_round_of_32 FLOAT,
    prob_sweet_16 FLOAT,
    prob_elite_8 FLOAT,
    prob_final_four FLOAT,
    prob_championship FLOAT,
    prob_champion FLOAT,
    expected_wins FLOAT,
    median_round_reached INTEGER,
    PRIMARY KEY (sim_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_advancement_sim ON team_advancement_probs(sim_id);
