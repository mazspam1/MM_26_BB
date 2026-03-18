"""
Application configuration using pydantic-settings.

Loads settings from environment variables and .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Environment
    environment: Literal["development", "staging", "production"] = "development"

    # Database
    database_path: str = "data/cbb_lines.duckdb"

    # The Odds API
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # Model settings
    model_version: str = "v1.3.0-phd"
    home_court_advantage: float = 3.5
    league_avg_efficiency: float = 100.0
    model_calibration_path: str = "data/model_calibration.json"
    min_games_played: int = 8
    quality_report_dir: str = "data/reports"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 2500

    # Dashboard settings
    dashboard_port: int = 2501

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
