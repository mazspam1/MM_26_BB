"""
Streamlit dashboard for NCAAB predictions.

Views:
- Today's Slate: Games with predictions and edges
- Backtest Results: Historical performance metrics
- CLV Analysis: Closing Line Value breakdown
- Team Ratings: Current team strength rankings

Run with: streamlit run apps/dashboard/app.py
"""

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from packages.common.config import get_settings
from packages.common.database import get_connection, init_database
from packages.models.enhanced_predictor import (
    EnhancedPredictor,
    MODEL_VERSION,
    create_enhanced_predictor,
)
from packages.features.kenpom_ratings import TeamRatings

# Page configuration
st.set_page_config(
    page_title="NCAAB Predictions",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database (skip schema init - tables created by pipeline)
# init_database()  # Disabled - schema managed by enhanced pipeline

def local_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        
        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #262730;
            border: 1px solid #41444c;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Game Cards */
        .game-card {
            background-color: #1f2937;
            border: 1px solid #374151;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        
        .status-badge {
            background-color: #059669;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #ef4444;
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="NCAAB Predictions | PhD Tier",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    local_css()
    
    st.title("🏀 NCAAB Intelligent Spread Engine")
    st.caption("PhD-Level Bayesian Inference • Heteroscedastic Uncertainty • Real-Time Market Integration")


    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Today's Slate", "Backtest Results", "CLV Analysis", "Team Ratings"],
    )

    if page == "Today's Slate":
        render_slate_page()
    elif page == "Backtest Results":
        render_backtest_page()
    elif page == "CLV Analysis":
        render_clv_page()
    elif page == "Team Ratings":
        render_ratings_page()


def render_slate_page():
    """Render today's slate with predictions."""
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <h2 style="margin: 0; margin-right: 15px;">Today's Slate</h2>
            <div class="status-badge">
                <span class="live-indicator"></span>Live Market Data
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Date selector
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        target_date = st.date_input("Select Date", value=date.today())
    with col2:
        min_spread = st.number_input("Min Spread (pts)", min_value=0.0, max_value=30.0, value=0.0, step=1.0)

    # Fetch games
    with st.spinner("Fetching games from ESPN..."):
        games_df = fetch_slate(target_date)

    if games_df.empty:
        st.info(f"No games with ratings found for {target_date}")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Games", len(games_df))
    with col2:
        avg_spread = games_df["proj_spread"].abs().mean()
        st.metric("Avg Spread", f"{avg_spread:.1f}")
    with col3:
        st.metric("Model Version", MODEL_VERSION)
    with col4:
        # Calculate coverage of Vegas lines
        if "market_spread" in games_df.columns:
            vegas_coverage = games_df["market_spread"].notna().mean()
            st.metric("Vegas Coverage", f"{vegas_coverage:.0%}")
        else:
            st.metric("Vegas Coverage", "0%")

    # Filter by minimum spread if specified
    if min_spread > 0:
        games_df = games_df[games_df["proj_spread"].abs() >= min_spread]

    # Sort by spread magnitude (biggest mismatches first)
    games_df = games_df.sort_values("proj_spread", key=abs, ascending=False)

    # Display games table
    st.subheader("Games")

    # Check which columns are available
    available_cols = games_df.columns.tolist()

    # Build display columns based on what's available
    base_cols = ["away_team_name", "home_team_name", "proj_spread"]

    # Add Vegas columns if available
    if "market_spread" in available_cols:
        base_cols.extend(["market_spread", "edge_vs_spread"])

    base_cols.extend(["proj_total", "home_win_prob"])

    # Add betting splits if available
    if "spread_favored_handle_pct" in available_cols:
        base_cols.append("spread_favored_handle_pct")

    base_cols.extend(["home_adj_em", "away_adj_em"])

    display_df = games_df[[c for c in base_cols if c in available_cols]].copy()

    # Rename columns
    col_rename = {
        "away_team_name": "Away",
        "home_team_name": "Home",
        "proj_spread": "Model",
        "market_spread": "Vegas",
        "edge_vs_spread": "Edge",
        "proj_total": "Total",
        "home_win_prob": "Win%",
        "spread_favored_handle_pct": "Fav$%",
        "home_adj_em": "H-EM",
        "away_adj_em": "A-EM",
    }
    display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})

    # Format columns
    if "Model" in display_df.columns:
        display_df["Model"] = display_df["Model"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "")
    if "Vegas" in display_df.columns:
        display_df["Vegas"] = display_df["Vegas"].apply(
            lambda x: f"{-abs(x):.1f}" if pd.notna(x) else "-"
        )
    if "Edge" in display_df.columns:
        display_df["Edge"] = display_df["Edge"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "-")
    if "Total" in display_df.columns:
        display_df["Total"] = display_df["Total"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "")
    if "Win%" in display_df.columns:
        display_df["Win%"] = display_df["Win%"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "")
    if "Fav$%" in display_df.columns:
        display_df["Fav$%"] = display_df["Fav$%"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")
    if "H-EM" in display_df.columns:
        display_df["H-EM"] = display_df["H-EM"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "")
    if "A-EM" in display_df.columns:
        display_df["A-EM"] = display_df["A-EM"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "")

    # Final cleanup - remove rows with all null predictions
    display_df = display_df[display_df["Model"] != ""]

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
    )

    # Best picks section - games with large spreads
    st.subheader("Top Predicted Mismatches")

    # Get games with large predicted spreads
    picks = games_df[games_df["proj_spread"].abs() >= 10].copy()

    if not picks.empty:
        picks = picks.sort_values("proj_spread", key=abs, ascending=False)

        for _, pick in picks.head(8).iterrows():
            render_enhanced_pick_card(pick)
    else:
        st.info("No large spread predictions for this date")


def render_enhanced_pick_card(pick):
    """Render a single enhanced pick card."""
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

        with col1:
            matchup = f"{pick['away_team_name']} @ {pick['home_team_name']}"
            st.markdown(f"**{matchup}**")

        with col2:
            spread = pick["proj_spread"]
            if spread > 0:
                fav = pick["home_team_name"][:12]
                line = f"{fav} -{abs(spread):.1f}"
            else:
                fav = pick["away_team_name"][:12]
                line = f"{fav} -{abs(spread):.1f}"
            st.markdown(f"**{line}**")

        with col3:
            total = pick["proj_total"]
            st.markdown(f"Total: **{total:.1f}**")

        with col4:
            win_prob = pick["home_win_prob"]
            home_em = pick["home_adj_em"]
            away_em = pick["away_adj_em"]
            em_diff = home_em - away_em
            if em_diff > 20:
                st.markdown(f":green[EM Diff: {em_diff:+.1f}]")
            elif em_diff > 10:
                st.markdown(f":orange[EM Diff: {em_diff:+.1f}]")
            else:
                st.markdown(f"EM Diff: {em_diff:+.1f}")

        # Show uncertainty
        std_col1, std_col2 = st.columns(2)
        with std_col1:
            st.caption(f"Spread Uncertainty (σ): {pick.get('spread_std', 0):.2f}")
        with std_col2:
            st.caption(f"Win Probability: {pick.get('home_win_prob', 0.5):.1%}")

        st.divider()


def render_pick_card(pick):
    """Render a single pick card."""
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 2])

        with col1:
            matchup = f"{pick['away_team_name']} @ {pick['home_team_name']}"
            st.markdown(f"**{matchup}**")

        with col2:
            side = pick["recommended_side"]
            if "home" in side:
                rec_team = pick["home_team_name"]
            elif "away" in side:
                rec_team = pick["away_team_name"]
            else:
                rec_team = side.upper()

            edge = pick.get("edge_vs_spread", 0) or 0
            st.markdown(f"**{rec_team}** ({edge:+.1f} edge)")

        with col3:
            conf = pick.get("confidence_rating", "")
            units = pick.get("recommended_units", 0) or 0
            if conf == "high":
                st.markdown(f":green[{conf.upper()}] - {units:.1f}u")
            elif conf == "medium":
                st.markdown(f":orange[{conf.upper()}] - {units:.1f}u")
            else:
                st.markdown(f"{conf.upper()} - {units:.1f}u")

        st.divider()


def render_backtest_page():
    """Render backtest results page."""
    st.header("Backtest Results")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=30),
            key="backtest_start",
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            key="backtest_end",
        )

    run_meta = fetch_latest_backtest_run()
    if not run_meta:
        st.info("No backtest runs found. Run: .\\start.ps1 backtest")
        return

    st.caption(
        f"Latest run: {run_meta['run_id']} ({run_meta['created_at']}) "
        f"| Range: {run_meta['start_date']} -> {run_meta['end_date']}"
    )

    # Fetch backtest data
    results_df = fetch_backtest_results(start_date, end_date, run_id=run_meta["run_id"])

    if results_df.empty:
        st.info("No completed games with predictions in this date range")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    spread_errors = results_df["spread_error"]
    total_errors = results_df["total_error"]

    with col1:
        st.metric("Games", len(results_df))
    with col2:
        st.metric("Spread MAE", f"{spread_errors.abs().mean():.2f}")
    with col3:
        st.metric("Total MAE", f"{total_errors.abs().mean():.2f}")
    with col4:
        spread_rmse = (spread_errors ** 2).mean() ** 0.5
        st.metric("Spread RMSE", f"{spread_rmse:.2f}")

    # Error distribution plots
    st.subheader("Error Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            results_df,
            x="spread_error",
            nbins=30,
            title="Spread Prediction Errors",
            labels={"spread_error": "Error (Predicted - Actual)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            results_df,
            x="total_error",
            nbins=30,
            title="Total Prediction Errors",
            labels={"total_error": "Error (Predicted - Actual)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    # Calibration plot
    st.subheader("Calibration")

    # Check coverage at different CI levels
    if "spread_ci_50_lower" in results_df.columns:
        in_50 = (
            (results_df["actual_spread"] >= results_df["spread_ci_50_lower"]) &
            (results_df["actual_spread"] <= results_df["spread_ci_50_upper"])
        ).mean()
        in_80 = (
            (results_df["actual_spread"] >= results_df["spread_ci_80_lower"]) &
            (results_df["actual_spread"] <= results_df["spread_ci_80_upper"])
        ).mean()
        in_95 = (
            (results_df["actual_spread"] >= results_df["spread_ci_95_lower"]) &
            (results_df["actual_spread"] <= results_df["spread_ci_95_upper"])
        ).mean()

        cal_df = pd.DataFrame({
            "Nominal": [0.50, 0.80, 0.95],
            "Actual": [in_50, in_80, in_95],
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Actual", x=["50%", "80%", "95%"], y=cal_df["Actual"]))
        fig.add_trace(go.Scatter(
            name="Perfect",
            x=["50%", "80%", "95%"],
            y=[0.50, 0.80, 0.95],
            mode="lines+markers",
            line=dict(dash="dash"),
        ))
        fig.update_layout(title="Spread Confidence Interval Coverage")
        st.plotly_chart(fig, use_container_width=True)

    # Rolling performance
    st.subheader("Rolling Performance")

    results_df["game_date"] = pd.to_datetime(results_df["game_date"])
    daily_mae = results_df.groupby("game_date")["spread_error"].apply(lambda x: x.abs().mean())

    fig = px.line(
        x=daily_mae.index,
        y=daily_mae.values,
        title="Daily Spread MAE",
        labels={"x": "Date", "y": "MAE"},
    )
    fig.add_hline(y=daily_mae.mean(), line_dash="dash", annotation_text="Average")
    st.plotly_chart(fig, use_container_width=True)

    # Segment diagnostics
    st.subheader("Segment Diagnostics")

    segment_options = {
        "Conference vs Non-Conference": "conference_game",
        "Season Timing (Early vs Late)": "season_timing",
        "Spread Buckets": "spread_bucket",
        "Tier Matchups": "tier_matchup",
    }
    selected_label = st.selectbox("Segment View", list(segment_options.keys()))
    segment_type = segment_options[selected_label]

    segment_df = fetch_backtest_segments(run_meta["run_id"], segment_type=segment_type)
    if segment_df.empty:
        st.info("No segment diagnostics available for this run.")
        return

    total_games = segment_df["total_games"].replace(0, pd.NA)
    segment_df["spread_line_rate"] = (
        segment_df["market_spread_count"] / total_games
    ).fillna(0.0)
    segment_df["total_line_rate"] = (
        segment_df["market_total_count"] / total_games
    ).fillna(0.0)
    segment_df["spread_80_drift"] = segment_df["spread_80_coverage"] - 0.80
    segment_df["total_80_drift"] = segment_df["total_80_coverage"] - 0.80

    display_df = segment_df[
        [
            "segment_value",
            "total_games",
            "spread_line_rate",
            "total_line_rate",
            "spread_mae",
            "total_mae",
            "spread_80_drift",
            "total_80_drift",
            "mean_spread_clv",
            "clv_positive_rate",
            "simulated_roi",
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "segment_value": "Segment",
            "total_games": "Games",
            "spread_line_rate": "Spread Line%",
            "total_line_rate": "Total Line%",
            "spread_mae": "Spread MAE",
            "total_mae": "Total MAE",
            "spread_80_drift": "Spread 80 Drift",
            "total_80_drift": "Total 80 Drift",
            "mean_spread_clv": "Mean Spread CLV",
            "clv_positive_rate": "CLV+ Rate",
            "simulated_roi": "Sim ROI",
        }
    )

    display_df["Spread Line%"] = display_df["Spread Line%"].apply(lambda x: f"{x:.0%}")
    display_df["Total Line%"] = display_df["Total Line%"].apply(lambda x: f"{x:.0%}")
    display_df["Spread MAE"] = display_df["Spread MAE"].apply(lambda x: f"{x:.2f}")
    display_df["Total MAE"] = display_df["Total MAE"].apply(lambda x: f"{x:.2f}")
    display_df["Spread 80 Drift"] = display_df["Spread 80 Drift"].apply(lambda x: f"{x:+.1%}")
    display_df["Total 80 Drift"] = display_df["Total 80 Drift"].apply(lambda x: f"{x:+.1%}")
    display_df["Mean Spread CLV"] = display_df["Mean Spread CLV"].apply(lambda x: f"{x:+.2f}")
    display_df["CLV+ Rate"] = display_df["CLV+ Rate"].apply(lambda x: f"{x:.0%}")
    display_df["Sim ROI"] = display_df["Sim ROI"].apply(lambda x: f"{x:+.1%}")

    st.dataframe(display_df, hide_index=True, use_container_width=True)


def render_clv_page():
    """Render CLV analysis page."""
    st.header("Closing Line Value Analysis")

    # Fetch CLV data
    clv_df = fetch_clv_data()

    if clv_df.empty:
        st.info("No CLV data available yet. CLV is calculated after games complete with closing lines captured.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Predictions", len(clv_df))
    with col2:
        mean_clv = clv_df["clv_spread"].mean()
        st.metric("Mean CLV", f"{mean_clv:+.2f}")
    with col3:
        clv_pos = (clv_df["clv_spread"] > 0).mean()
        st.metric("CLV+ Rate", f"{clv_pos:.1%}")
    with col4:
        median_clv = clv_df["clv_spread"].median()
        st.metric("Median CLV", f"{median_clv:+.2f}")

    # CLV distribution
    st.subheader("CLV Distribution")

    fig = px.histogram(
        clv_df,
        x="clv_spread",
        nbins=30,
        title="Spread CLV Distribution",
        labels={"clv_spread": "CLV (points)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # CLV by confidence level
    if "confidence_rating" in clv_df.columns:
        st.subheader("CLV by Confidence Level")

        conf_clv = clv_df.groupby("confidence_rating")["clv_spread"].agg(["mean", "count"])

        fig = px.bar(
            x=conf_clv.index,
            y=conf_clv["mean"],
            title="Mean CLV by Confidence Rating",
            labels={"x": "Confidence", "y": "Mean CLV"},
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)


def render_ratings_page():
    """Render team ratings page."""
    st.header("Team Ratings")

    # Fetch team ratings
    ratings_df = fetch_team_ratings()

    if ratings_df.empty:
        st.info("No team ratings available yet. Ratings are calculated after processing game data.")
        return

    # Conference filter
    conferences = ["All"] + sorted(ratings_df["conference"].unique().tolist())
    selected_conf = st.selectbox("Filter by Conference", conferences)

    if selected_conf != "All":
        ratings_df = ratings_df[ratings_df["conference"] == selected_conf]

    # Sort selector
    sort_by = st.selectbox(
        "Sort by",
        ["Efficiency Margin", "Adjusted Offense", "Adjusted Defense", "Tempo"],
    )

    sort_col = {
        "Efficiency Margin": "adj_em",
        "Adjusted Offense": "adj_off",
        "Adjusted Defense": "adj_def",
        "Tempo": "adj_tempo",
    }[sort_by]

    ascending = sort_by == "Adjusted Defense"  # Lower is better for defense
    ratings_df = ratings_df.sort_values(sort_col, ascending=ascending)

    # Add rank column
    ratings_df["Rank"] = range(1, len(ratings_df) + 1)

    # Display table
    display_df = ratings_df[[
        "Rank", "name", "conference",
        "adj_off", "adj_def", "adj_em", "adj_tempo", "games_played"
    ]].copy()

    display_df.columns = [
        "Rank", "Team", "Conference",
        "Adj Off", "Adj Def", "Adj EM", "Tempo", "Games"
    ]

    # Format numbers
    display_df["Adj Off"] = display_df["Adj Off"].apply(lambda x: f"{x:.1f}")
    display_df["Adj Def"] = display_df["Adj Def"].apply(lambda x: f"{x:.1f}")
    display_df["Adj EM"] = display_df["Adj EM"].apply(lambda x: f"{x:+.1f}")
    display_df["Tempo"] = display_df["Tempo"].apply(lambda x: f"{x:.1f}")

    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # Scatter plot of offense vs defense
    st.subheader("Offense vs Defense")

    fig = px.scatter(
        ratings_df,
        x="adj_off",
        y="adj_def",
        hover_name="name",
        color="conference",
        title="Adjusted Offense vs Defense (top-left is best)",
        labels={"adj_off": "Adjusted Offense", "adj_def": "Adjusted Defense"},
    )
    # Invert y-axis (lower defense is better)
    fig.update_yaxes(autorange="reversed")
    fig.add_hline(y=100, line_dash="dash", line_color="gray")
    fig.add_vline(x=100, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)


# Cached predictor and ratings
@st.cache_resource
def get_predictor():
    """Get cached enhanced predictor."""
    return create_enhanced_predictor()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ratings() -> dict:
    """Get cached ratings from database."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                team_id, as_of_date,
                adj_offensive_efficiency, adj_defensive_efficiency, adj_tempo, adj_em,
                off_efg, off_tov, off_orb, off_ftr,
                def_efg, def_tov, def_drb, def_ftr,
                games_played, sos_off, sos_def,
                home_off_delta, home_def_delta, away_off_delta, away_def_delta,
                home_games_played, away_games_played,
                off_rating_std, def_rating_std, tempo_std
            FROM team_strengths
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM team_strengths)
        """
        ).fetchall()

    if not rows:
        return {}

    ratings = {}
    for row in rows:
        ratings[row[0]] = TeamRatings(
            team_id=row[0],
            adj_off=row[2],
            adj_def=row[3],
            adj_tempo=row[4],
            adj_em=row[5],
            adj_efg=row[6],
            adj_tov=row[7],
            adj_orb=row[8],
            adj_ftr=row[9],
            adj_efg_def=row[10],
            adj_tov_def=row[11],
            adj_drb=row[12],
            adj_ftr_def=row[13],
            games_played=row[14],
            sos_off=row[15],
            sos_def=row[16],
            as_of_date=date.fromisoformat(row[1]) if isinstance(row[1], str) else row[1],
            home_off_delta=row[17],
            home_def_delta=row[18],
            away_off_delta=row[19],
            away_def_delta=row[20],
            home_games_played=row[21],
            away_games_played=row[22],
            off_std=row[23],
            def_std=row[24],
            tempo_std=row[25],
        )

    return ratings


# Data fetching functions
def fetch_slate(target_date: date) -> pd.DataFrame:
    """Fetch games and predictions from database (includes Vegas lines)."""

    # First try to get from database (includes betting splits)
    with get_connection() as conn:
        query = """
        WITH latest_predictions AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY prediction_timestamp DESC) AS rn
                FROM predictions
            )
            WHERE rn = 1
        ),
        latest_splits AS (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY snapshot_timestamp DESC) AS rn
            FROM betting_splits
        )
        SELECT
            g.game_id, g.game_date, g.game_datetime,
            g.home_team_id, g.home_team_name,
            g.away_team_id, g.away_team_name,
            g.status, g.neutral_site,
            p.proj_home_score, p.proj_away_score,
            p.proj_spread, p.proj_total,
            p.home_win_prob,
            p.spread_ci_80_lower, p.spread_ci_80_upper,
            bs.spread_line_home as market_spread,
            bs.total_line as market_total,
            bs.spread_favored_handle_pct, bs.spread_favored_bets_pct,
            bs.total_over_handle_pct, bs.total_over_bets_pct,
            ts_home.adj_offensive_efficiency as home_adj_off,
            ts_home.adj_defensive_efficiency as home_adj_def,
            (ts_home.adj_offensive_efficiency - ts_home.adj_defensive_efficiency) as home_adj_em,
            ts_away.adj_offensive_efficiency as away_adj_off,
            ts_away.adj_defensive_efficiency as away_adj_def,
            (ts_away.adj_offensive_efficiency - ts_away.adj_defensive_efficiency) as away_adj_em
        FROM games g
        LEFT JOIN latest_predictions p ON g.game_id = p.game_id
        LEFT JOIN latest_splits bs ON g.game_id = bs.game_id AND bs.rn = 1
        LEFT JOIN team_strengths ts_home ON g.home_team_id = ts_home.team_id
        LEFT JOIN team_strengths ts_away ON g.away_team_id = ts_away.team_id
        WHERE g.game_date = ?
        ORDER BY g.game_datetime
        """
        df = pd.read_sql_query(query, conn, params=(target_date.isoformat(),))

    if not df.empty and df['proj_spread'].notna().any():
        # Compute edges using model convention (positive = home favored)
        if "market_spread" in df.columns:
            df["edge_vs_spread"] = df["proj_spread"] - df["market_spread"]
            df.loc[df["market_spread"].isna() | df["proj_spread"].isna(), "edge_vs_spread"] = None
        if "market_total" in df.columns:
            df["edge_vs_total"] = df["proj_total"] - df["market_total"]
            df.loc[df["market_total"].isna() | df["proj_total"].isna(), "edge_vs_total"] = None
        return df

    # Fallback to live prediction if no database data
    from packages.common.sportsdataverse_mbb import load_mbb

    date_str = target_date.strftime('%Y%m%d')
    try:
        schedule = load_mbb().espn_mbb_schedule(dates=date_str, groups=50, return_as_pandas=True)
    except Exception as e:
        st.error(f"Failed to fetch schedule: {e}")
        return pd.DataFrame()

    if len(schedule) == 0:
        return pd.DataFrame()

    ratings = get_ratings()
    predictor = get_predictor()

    if not ratings:
        st.warning("No ratings available. Run the pipeline first.")
        return pd.DataFrame()

    predictions = []
    for _, game in schedule.iterrows():
        try:
            home_id = int(game.get('home_id'))
            away_id = int(game.get('away_id'))
            game_id = int(game.get('game_id')) if game.get('game_id') else 0
        except (ValueError, TypeError):
            continue

        is_neutral = bool(game.get('neutral_site', False))
        home_name = game.get('home_display_name', f'Team {home_id}')
        away_name = game.get('away_display_name', f'Team {away_id}')
        game_time = game.get('start_date', '')

        if home_id not in ratings or away_id not in ratings:
            continue

        home_ratings = ratings[home_id]
        away_ratings = ratings[away_id]

        pred = predictor.predict_game(
            home_ratings=home_ratings,
            away_ratings=away_ratings,
            game_id=game_id,
            is_neutral=is_neutral,
        )

        predictions.append({
            'game_id': game_id,
            'game_date': target_date,
            'game_datetime': game_time,
            'home_team_id': home_id,
            'home_team_name': home_name,
            'away_team_id': away_id,
            'away_team_name': away_name,
            'status': game.get('status_type_description', 'Scheduled'),
            'neutral_site': is_neutral,
            'proj_home_score': pred.home_score,
            'proj_away_score': pred.away_score,
            'proj_spread': pred.spread,
            'proj_total': pred.total,
            'home_win_prob': pred.home_win_prob,
            'spread_ci_80_lower': pred.spread_ci_80[0],
            'spread_ci_80_upper': pred.spread_ci_80[1],
            'home_adj_em': home_ratings.adj_em,
            'away_adj_em': away_ratings.adj_em,
            'market_spread': None,
            'market_total': None,
            'edge_vs_spread': None,
            'edge_vs_total': None,
        })

    return pd.DataFrame(predictions)


def fetch_backtest_results(start_date: date, end_date: date, run_id: str) -> pd.DataFrame:
    """Fetch backtest results for date range."""
    with get_connection() as conn:
        query = """
        SELECT
            g.game_id, g.game_date,
            g.home_score, g.away_score,
            g.home_score - g.away_score as actual_spread,
            g.home_score + g.away_score as actual_total,
            b.proj_spread, b.proj_total,
            b.proj_spread - (g.home_score - g.away_score) as spread_error,
            b.proj_total - (g.home_score + g.away_score) as total_error,
            b.spread_ci_50_lower, b.spread_ci_50_upper,
            b.spread_ci_80_lower, b.spread_ci_80_upper,
            b.spread_ci_95_lower, b.spread_ci_95_upper
        FROM games g
        JOIN backtest_predictions b ON g.game_id = b.game_id
        WHERE g.status = 'final'
            AND g.home_score IS NOT NULL
            AND g.game_date BETWEEN ? AND ?
            AND b.run_id = ?
        ORDER BY g.game_date
        """
        df = pd.read_sql_query(
            query, conn,
            params=(start_date.isoformat(), end_date.isoformat(), run_id),
        )

    return df


def fetch_backtest_segments(run_id: str, segment_type: Optional[str] = None) -> pd.DataFrame:
    """Fetch backtest segment diagnostics for a run."""
    with get_connection() as conn:
        query = """
        SELECT
            segment_type, segment_value,
            total_games,
            market_spread_count, market_total_count,
            closing_spread_count, closing_total_count,
            spread_mae, spread_rmse, total_mae, total_rmse,
            spread_50_coverage, spread_80_coverage, spread_95_coverage,
            total_50_coverage, total_80_coverage, total_95_coverage,
            mean_spread_clv, mean_total_clv, clv_positive_rate,
            simulated_roi
        FROM backtest_segments
        WHERE run_id = ?
        """
        params: list = [run_id]
        if segment_type:
            query += " AND segment_type = ?"
            params.append(segment_type)
        query += " ORDER BY segment_value"

        df = pd.read_sql_query(query, conn, params=params)

    return df


def fetch_latest_backtest_run() -> Optional[dict]:
    """Fetch most recent backtest run metadata."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT run_id, created_at, start_date, end_date
            FROM backtest_runs
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        return None

    created_at = row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1])
    return {
        "run_id": row[0],
        "created_at": created_at,
        "start_date": row[2],
        "end_date": row[3],
    }


def fetch_clv_data() -> pd.DataFrame:
    """Fetch CLV analysis data."""
    with get_connection() as conn:
        query = """
        SELECT
            c.game_id, c.prediction_timestamp,
            c.spread_clv as clv_spread,
            c.total_clv as clv_total,
            COALESCE(bp.confidence_rating, p.confidence_rating) as confidence_rating,
            COALESCE(bp.recommended_side, p.recommended_side) as recommended_side
        FROM clv_reports c
        LEFT JOIN backtest_predictions bp
            ON c.game_id = bp.game_id AND c.prediction_timestamp = bp.prediction_timestamp
        LEFT JOIN predictions p
            ON c.game_id = p.game_id AND c.prediction_timestamp = p.prediction_timestamp
        ORDER BY c.prediction_timestamp DESC
        """
        df = pd.read_sql_query(query, conn)

    return df


def fetch_team_ratings() -> pd.DataFrame:
    """Fetch current team ratings."""
    # Get ratings directly from cached data
    ratings = get_ratings()

    if not ratings:
        return pd.DataFrame()

    # Get team names from ESPN
    from packages.common.sportsdataverse_mbb import load_mbb
    try:
        teams = load_mbb().espn_mbb_teams(groups=50, return_as_pandas=True)
        team_names = {}
        team_confs = {}
        for _, row in teams.iterrows():
            tid = int(row.get('team_id', 0))
            team_names[tid] = row.get('team_display_name', f'Team {tid}')
            team_confs[tid] = row.get('team_location', 'Unknown')  # Use location as proxy
    except:
        team_names = {}
        team_confs = {}

    # Build dataframe
    data = []
    for team_id, r in ratings.items():
        data.append({
            'team_id': team_id,
            'name': team_names.get(team_id, f'Team {team_id}'),
            'conference': team_confs.get(team_id, 'Unknown'),
            'adj_off': r.adj_off,
            'adj_def': r.adj_def,
            'adj_em': r.adj_em,
            'adj_tempo': r.adj_tempo,
            'games_played': r.games_played,
        })

    df = pd.DataFrame(data)
    return df.sort_values('adj_em', ascending=False)


if __name__ == "__main__":
    main()
