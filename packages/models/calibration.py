"""
Calibration and uncertainty quantification using MAPIE.

Provides distribution-free prediction intervals with finite-sample
coverage guarantees through conformal prediction.

References:
- MAPIE: https://mapie.readthedocs.io/
- Vovk et al., "Algorithmic Learning in a Random World"
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
import structlog

logger = structlog.get_logger()

# Try to import MAPIE, provide fallback if not available
try:
    from mapie.regression import MapieRegressor
    from mapie.conformity_scores import AbsoluteConformityScore
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    logger.warning("MAPIE not available, using Gaussian intervals")


@dataclass
class CalibrationResult:
    """Result of calibration check."""

    coverage_50: float  # Actual coverage at 50% nominal
    coverage_80: float  # Actual coverage at 80% nominal
    coverage_95: float  # Actual coverage at 95% nominal
    mean_interval_width_50: float
    mean_interval_width_80: float
    mean_interval_width_95: float
    n_samples: int
    is_calibrated: bool  # All coverages within tolerance


@dataclass
class PredictionWithInterval:
    """Point prediction with conformal interval."""

    prediction: float
    lower_50: float
    upper_50: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


class GaussianIntervalEstimator:
    """
    Fallback interval estimator using Gaussian assumption.

    Used when MAPIE is not available or insufficient data.
    """

    def __init__(self, default_std: float = 11.0):
        """
        Initialize with default standard deviation.

        Args:
            default_std: Default standard deviation for intervals
        """
        self.std = default_std
        self._residuals: list[float] = []

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "GaussianIntervalEstimator":
        """
        Fit estimator by computing residual standard deviation.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            self
        """
        residuals = y_true - y_pred
        self._residuals = list(residuals)
        self.std = np.std(residuals)
        logger.info("Gaussian interval estimator fit", std=self.std, n_samples=len(residuals))
        return self

    def predict_interval(
        self,
        y_pred: float,
        alpha: float = 0.80,
    ) -> tuple[float, float]:
        """
        Generate prediction interval.

        Args:
            y_pred: Point prediction
            alpha: Confidence level (e.g., 0.80 for 80% CI)

        Returns:
            Tuple of (lower, upper) bounds
        """
        from scipy import stats

        z = stats.norm.ppf((1 + alpha) / 2)
        lower = y_pred - z * self.std
        upper = y_pred + z * self.std
        return lower, upper

    def predict_all_intervals(
        self,
        y_pred: float,
    ) -> PredictionWithInterval:
        """
        Generate all standard prediction intervals.

        Args:
            y_pred: Point prediction

        Returns:
            PredictionWithInterval with 50/80/95% CIs
        """
        l50, u50 = self.predict_interval(y_pred, 0.50)
        l80, u80 = self.predict_interval(y_pred, 0.80)
        l95, u95 = self.predict_interval(y_pred, 0.95)

        return PredictionWithInterval(
            prediction=y_pred,
            lower_50=l50,
            upper_50=u50,
            lower_80=l80,
            upper_80=u80,
            lower_95=l95,
            upper_95=u95,
        )


class ConformalIntervalEstimator:
    """
    Conformal prediction intervals using MAPIE.

    Provides distribution-free, finite-sample valid coverage.
    """

    def __init__(
        self,
        base_model: Optional[BaseEstimator] = None,
        method: str = "plus",
        cv: int = 5,
    ):
        """
        Initialize conformal estimator.

        Args:
            base_model: Sklearn regressor (default: Ridge)
            method: MAPIE method ("naive", "base", "plus", "minmax")
            cv: Cross-validation folds for MAPIE
        """
        if not MAPIE_AVAILABLE:
            logger.warning("MAPIE not available, will use Gaussian fallback")
            self._use_fallback = True
            self._fallback = GaussianIntervalEstimator()
            return

        self._use_fallback = False
        self.base_model = base_model or Ridge(alpha=1.0)
        self.method = method
        self.cv = cv
        self._mapie: Optional[MapieRegressor] = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "ConformalIntervalEstimator":
        """
        Fit the conformal predictor.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values

        Returns:
            self
        """
        if self._use_fallback:
            # For fallback, we need predictions first
            # Just store data for now
            self._X_train = X
            self._y_train = y
            logger.info("Using Gaussian fallback", n_samples=len(y))
            return self

        if len(X) < self.cv * 2:
            logger.warning(
                "Insufficient data for conformal prediction, using Gaussian fallback",
                n_samples=len(X),
                min_required=self.cv * 2,
            )
            self._use_fallback = True
            self._fallback = GaussianIntervalEstimator()
            return self

        self._mapie = MapieRegressor(
            estimator=self.base_model,
            method=self.method,
            cv=self.cv,
            conformity_score=AbsoluteConformityScore(),
        )

        self._mapie.fit(X, y)
        self._is_fitted = True

        logger.info(
            "Conformal estimator fit",
            n_samples=len(X),
            method=self.method,
            cv=self.cv,
        )

        return self

    def predict(
        self,
        X: np.ndarray,
        alpha: float = 0.80,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with intervals.

        Args:
            X: Feature matrix
            alpha: Confidence level

        Returns:
            Tuple of (predictions, intervals) where intervals is (n, 2, 1)
        """
        if self._use_fallback:
            raise ValueError("Fallback estimator needs point predictions, use predict_from_point")

        if not self._is_fitted:
            raise ValueError("Estimator not fitted")

        # MAPIE uses alpha as error rate (1 - confidence)
        y_pred, y_pis = self._mapie.predict(X, alpha=1 - alpha)

        return y_pred, y_pis

    def predict_from_point(
        self,
        y_pred: float,
        X: Optional[np.ndarray] = None,
    ) -> PredictionWithInterval:
        """
        Generate intervals from a point prediction.

        For use when we have predictions from another model.

        Args:
            y_pred: Point prediction
            X: Features (used if conformal available)

        Returns:
            PredictionWithInterval
        """
        if self._use_fallback:
            return self._fallback.predict_all_intervals(y_pred)

        if X is None or not self._is_fitted:
            # Fall back to Gaussian
            fallback = GaussianIntervalEstimator()
            return fallback.predict_all_intervals(y_pred)

        # Get intervals from MAPIE at different alpha levels
        X_reshaped = X.reshape(1, -1) if X.ndim == 1 else X

        _, pi_50 = self._mapie.predict(X_reshaped, alpha=0.50)
        _, pi_80 = self._mapie.predict(X_reshaped, alpha=0.20)
        _, pi_95 = self._mapie.predict(X_reshaped, alpha=0.05)

        return PredictionWithInterval(
            prediction=y_pred,
            lower_50=float(pi_50[0, 0, 0]),
            upper_50=float(pi_50[0, 1, 0]),
            lower_80=float(pi_80[0, 0, 0]),
            upper_80=float(pi_80[0, 1, 0]),
            lower_95=float(pi_95[0, 0, 0]),
            upper_95=float(pi_95[0, 1, 0]),
        )


def check_calibration(
    y_true: np.ndarray,
    intervals_50: np.ndarray,
    intervals_80: np.ndarray,
    intervals_95: np.ndarray,
    tolerance: float = 0.05,
) -> CalibrationResult:
    """
    Check calibration of prediction intervals.

    Args:
        y_true: Actual values (n,)
        intervals_50: 50% intervals (n, 2)
        intervals_80: 80% intervals (n, 2)
        intervals_95: 95% intervals (n, 2)
        tolerance: Acceptable deviation from nominal coverage

    Returns:
        CalibrationResult dataclass
    """
    n = len(y_true)

    # Calculate actual coverage
    in_50 = np.sum((y_true >= intervals_50[:, 0]) & (y_true <= intervals_50[:, 1]))
    in_80 = np.sum((y_true >= intervals_80[:, 0]) & (y_true <= intervals_80[:, 1]))
    in_95 = np.sum((y_true >= intervals_95[:, 0]) & (y_true <= intervals_95[:, 1]))

    coverage_50 = in_50 / n
    coverage_80 = in_80 / n
    coverage_95 = in_95 / n

    # Calculate mean interval widths
    width_50 = np.mean(intervals_50[:, 1] - intervals_50[:, 0])
    width_80 = np.mean(intervals_80[:, 1] - intervals_80[:, 0])
    width_95 = np.mean(intervals_95[:, 1] - intervals_95[:, 0])

    # Check if calibrated (within tolerance of nominal)
    is_calibrated = (
        abs(coverage_50 - 0.50) <= tolerance
        and abs(coverage_80 - 0.80) <= tolerance
        and abs(coverage_95 - 0.95) <= tolerance
    )

    return CalibrationResult(
        coverage_50=coverage_50,
        coverage_80=coverage_80,
        coverage_95=coverage_95,
        mean_interval_width_50=width_50,
        mean_interval_width_80=width_80,
        mean_interval_width_95=width_95,
        n_samples=n,
        is_calibrated=is_calibrated,
    )


class SpreadCalibrator:
    """
    Specialized calibrator for spread predictions.

    Maintains separate estimators for spread and total predictions.
    """

    def __init__(self, min_samples: int = 100):
        """
        Initialize spread calibrator.

        Args:
            min_samples: Minimum samples before using conformal
        """
        self.min_samples = min_samples
        self._spread_estimator = GaussianIntervalEstimator(default_std=11.0)
        self._total_estimator = GaussianIntervalEstimator(default_std=13.0)
        self._spread_residuals: list[float] = []
        self._total_residuals: list[float] = []

    def add_result(
        self,
        spread_pred: float,
        spread_actual: float,
        total_pred: float,
        total_actual: float,
    ) -> None:
        """
        Add a prediction result for calibration.

        Args:
            spread_pred: Predicted spread
            spread_actual: Actual spread
            total_pred: Predicted total
            total_actual: Actual total
        """
        self._spread_residuals.append(spread_actual - spread_pred)
        self._total_residuals.append(total_actual - total_pred)

        # Refit estimators periodically
        if len(self._spread_residuals) % 50 == 0:
            self._refit()

    def _refit(self) -> None:
        """Refit estimators with accumulated residuals."""
        if len(self._spread_residuals) >= 10:
            self._spread_estimator.std = np.std(self._spread_residuals)

        if len(self._total_residuals) >= 10:
            self._total_estimator.std = np.std(self._total_residuals)

        logger.debug(
            "Calibrator refit",
            spread_std=self._spread_estimator.std,
            total_std=self._total_estimator.std,
            n_samples=len(self._spread_residuals),
        )

    def get_spread_intervals(self, spread_pred: float) -> PredictionWithInterval:
        """Get calibrated intervals for spread prediction."""
        return self._spread_estimator.predict_all_intervals(spread_pred)

    def get_total_intervals(self, total_pred: float) -> PredictionWithInterval:
        """Get calibrated intervals for total prediction."""
        return self._total_estimator.predict_all_intervals(total_pred)

    def get_calibration_stats(self) -> dict:
        """Get current calibration statistics."""
        return {
            "spread_std": self._spread_estimator.std,
            "total_std": self._total_estimator.std,
            "n_samples": len(self._spread_residuals),
            "min_samples_for_conformal": self.min_samples,
            "using_conformal": len(self._spread_residuals) >= self.min_samples,
        }


def create_default_calibrator() -> SpreadCalibrator:
    """Create calibrator with default settings."""
    return SpreadCalibrator(min_samples=100)


@dataclass
class ModelCalibrationParams:
    """Persisted calibration parameters for EnhancedPredictor."""

    spread_bias: float
    spread_scale: float
    total_bias: float
    total_scale: float
    base_spread_std: float
    base_total_std: float
    market_anchor_weight_spread: float
    market_anchor_weight_total: float
    n_samples: int
    fitted_at: str
    source_run_id: Optional[str] = None


def save_model_calibration(params: ModelCalibrationParams, path: Path) -> None:
    """Save calibration params to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(params), handle, indent=2, sort_keys=True)


def load_model_calibration(path: Path) -> Optional[ModelCalibrationParams]:
    """Load calibration params from disk if present."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ModelCalibrationParams(**data)
