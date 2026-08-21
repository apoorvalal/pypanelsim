"""Outcome-model surfaces used in the R and Python panel reproduction.

All functions accept unit-by-period outcome and treatment matrices and return a
same-shaped estimate of the untreated-outcome surface. Treated cells are never
used as observed outcomes during fitting.
"""

from __future__ import annotations

import crabbymetrics as cm
import numpy as np


def _validate_panel(
    outcome: np.ndarray, treatment: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if outcome.ndim != 2 or treatment.shape != outcome.shape:
        raise ValueError(
            "outcome and treatment must be same-shaped two-dimensional arrays"
        )
    if not np.isfinite(outcome).all():
        raise ValueError("outcome must contain only finite values")
    if not np.isin(treatment, (0.0, 1.0)).all():
        raise ValueError("treatment must contain only zero and one")
    if not (treatment < 0.5).any(axis=1).all():
        raise ValueError("each unit must have an untreated observation")
    return outcome, treatment


def _additive_surface(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 2_000,
) -> np.ndarray:
    """Project observed cells on additive unit and period effects."""

    if values.shape != observed.shape:
        raise ValueError("values and observed must have the same shape")
    if not observed.any():
        raise ValueError("at least one observed cell is required")
    intercept = float(values[observed].mean())
    unit_effect = np.zeros(values.shape[0], dtype=float)
    period_effect = np.zeros(values.shape[1], dtype=float)
    fitted = np.full(values.shape, intercept, dtype=float)
    for _ in range(max_iterations):
        previous = fitted
        for unit in range(values.shape[0]):
            mask = observed[unit]
            if mask.any():
                unit_effect[unit] = np.mean(
                    values[unit, mask] - intercept - period_effect[mask]
                )
        for period in range(values.shape[1]):
            mask = observed[:, period]
            if mask.any():
                period_effect[period] = np.mean(
                    values[mask, period] - intercept - unit_effect[mask]
                )
        unit_shift = float(unit_effect.mean())
        unit_effect -= unit_shift
        intercept += unit_shift
        period_shift = float(period_effect.mean())
        period_effect -= period_shift
        intercept += period_shift
        fitted = intercept + unit_effect[:, None] + period_effect[None, :]
        relative_change = np.linalg.norm(fitted - previous) / (
            np.linalg.norm(previous) + 1e-12
        )
        if relative_change < tolerance:
            return fitted
    raise RuntimeError("additive unit and period effects did not converge")


def two_way_fe_surface(outcome: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Fit the R experiment's two-way fixed-effects nuisance surface."""

    outcome, treatment = _validate_panel(outcome, treatment)
    return _additive_surface(outcome, treatment < 0.5)


def interactive_fixed_effects_surface(
    outcome: np.ndarray,
    treatment: np.ndarray,
    *,
    rank: int = 2,
    force: int = 3,
    tolerance: float = 1e-4,
    max_iterations: int = 4_000,
) -> np.ndarray:
    """Fit the historical ``fect(method='ife')`` missing-cell EM algorithm.

    The E-step fills treated cells with the current counterfactual surface. The
    M-step applies the rank-constrained interactive fixed-effects projection in
    :class:`crabbymetrics.InteractiveFixedEffects`.
    """

    outcome, treatment = _validate_panel(outcome, treatment)
    observed = (treatment < 0.5).T
    y_time_unit = outcome.T
    fitted = two_way_fe_surface(outcome, treatment).T
    model = cm.InteractiveFixedEffects(rank=rank, force=force)
    for _ in range(max_iterations + 1):
        filled = np.where(observed, y_time_unit, fitted)
        model.fit(filled)
        updated = np.asarray(model.predict(), dtype=float)
        relative_change = np.linalg.norm(updated - fitted) / (
            np.linalg.norm(fitted) + 1e-12
        )
        fitted = updated
        if relative_change < tolerance:
            return fitted.T
    # Historical fect returns the last iterate when its iteration budget is
    # exhausted. Preserve that contract for the archived experiment.
    return fitted.T


def generalized_synthetic_control_surface(
    outcome: np.ndarray,
    treatment: np.ndarray,
    *,
    rank: int = 2,
    force: int = 3,
) -> np.ndarray:
    """Fit the historical fixed-rank generalized synthetic-control surface.

    Factors and common time effects are estimated on never-treated controls.
    Each treated unit's loading and unit intercept are then estimated on its
    untreated pre-treatment observations.
    """

    outcome, treatment = _validate_panel(outcome, treatment)
    controls = np.flatnonzero(treatment.sum(axis=1) == 0.0)
    treated = np.flatnonzero(treatment.sum(axis=1) > 0.0)
    if controls.size == 0 or treated.size == 0:
        raise ValueError(
            "generalized synthetic control requires treated and never-treated units"
        )

    control_model = cm.InteractiveFixedEffects(rank=rank, force=force)
    control_model.fit(outcome[controls].T)
    control_summary = control_model.summary()
    factor = np.asarray(control_summary["factor"], dtype=float)
    mu = float(control_summary["mu"])
    time_effect = np.asarray(control_summary["xi"], dtype=float)
    surface = np.empty_like(outcome)
    surface[controls] = np.asarray(control_model.predict(), dtype=float).T

    design = factor
    if force in (1, 3):
        design = np.column_stack((design, np.ones(outcome.shape[1])))
    for unit in treated:
        pre = treatment[unit] < 0.5
        target = outcome[unit] - mu
        if force in (2, 3):
            target = target - time_effect
        coefficients, *_ = np.linalg.lstsq(design[pre], target[pre], rcond=None)
        fitted = mu + design @ coefficients
        if force in (2, 3):
            fitted = fitted + time_effect
        surface[unit] = fitted
    return surface


def mcpanel_surface(
    outcome: np.ndarray,
    treatment: np.ndarray,
    *,
    lambda_l: float = 0.002,
    tolerance: float = 1e-5,
    max_iterations: int = 1_000,
) -> np.ndarray:
    """Fit the fixed-penalty MCPanel nuclear-norm outcome model."""

    outcome, treatment = _validate_panel(outcome, treatment)
    model = cm.MatrixCompletion(
        lambda_l=lambda_l,
        fit_unit_effects=True,
        fit_time_effects=True,
        max_iterations=max_iterations,
        effect_iterations=1,
        tolerance=tolerance,
    )
    model.fit(outcome, treatment)
    return np.asarray(model.predict(), dtype=float)


def _two_way_residualize(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    tolerance: float = 1e-11,
    max_iterations: int = 2_000,
) -> np.ndarray:
    """Apply alternating unit and period demeaning on an unbalanced mask."""

    residual = np.where(observed, values, 0.0).astype(float, copy=True)
    for _ in range(max_iterations):
        previous = residual.copy()
        for unit in range(values.shape[0]):
            mask = observed[unit]
            if mask.any():
                residual[unit, mask] -= residual[unit, mask].mean()
        for period in range(values.shape[1]):
            mask = observed[:, period]
            if mask.any():
                residual[mask, period] -= residual[mask, period].mean()
        change = np.linalg.norm(residual - previous) / (
            np.linalg.norm(previous) + 1e-12
        )
        if change < tolerance:
            return residual
    raise RuntimeError("two-way residualization did not converge")


def panel_var_surface(
    outcome: np.ndarray,
    treatment: np.ndarray,
    *,
    lags: int = 1,
) -> np.ndarray:
    """Fit the R experiment's pooled panel VAR outcome model.

    The model regresses the outcome on ``lags`` of the outcome and additive
    unit and period effects. The fit uses untreated cells with complete lag
    histories. Treated post-period outcomes are then forecast recursively.
    """

    outcome, treatment = _validate_panel(outcome, treatment)
    if not isinstance(lags, int) or lags < 1 or lags >= outcome.shape[1]:
        raise ValueError(
            "lags must be a positive integer smaller than the number of periods"
        )

    lagged = np.zeros((lags, *outcome.shape), dtype=float)
    for lag in range(1, lags + 1):
        lagged[lag - 1, :, lag:] = outcome[:, :-lag]
    observed = treatment < 0.5
    observed[:, :lags] = False
    y_residual = _two_way_residualize(outcome, observed)
    x_residuals = np.stack(
        [_two_way_residualize(lagged[index], observed) for index in range(lags)]
    )
    design = np.column_stack([residual[observed] for residual in x_residuals])
    coefficients, _, model_rank, _ = np.linalg.lstsq(
        design, y_residual[observed], rcond=None
    )
    if model_rank < lags:
        raise RuntimeError("panel VAR lag design is rank deficient")
    adjusted = outcome - np.tensordot(coefficients, lagged, axes=(0, 0))
    additive = _additive_surface(adjusted, observed)

    surface = np.zeros_like(outcome)
    surface[:, lags:] = additive[:, lags:]
    for lag, coefficient in enumerate(coefficients, start=1):
        surface[:, lags:] += coefficient * outcome[:, lags - lag : -lag]

    for unit in range(outcome.shape[0]):
        treated_periods = np.flatnonzero(treatment[unit] > 0.5)
        if treated_periods.size:
            history = outcome[unit].copy()
            for period in treated_periods:
                if period < lags:
                    surface[unit, period] = 0.0
                    history[period] = 0.0
                    continue
                prediction = additive[unit, period]
                for lag, coefficient in enumerate(coefficients, start=1):
                    prediction += coefficient * history[period - lag]
                surface[unit, period] = prediction
                history[period] = prediction
    return surface


def _fit_ar_candidate(
    series: np.ndarray, difference: int, order: int
) -> tuple[float, np.ndarray]:
    transformed = np.diff(series, n=difference) if difference else series.copy()
    if transformed.size <= order + 2:
        return np.inf, np.empty(0)
    response = transformed[order:]
    columns = [np.ones(response.size)]
    for lag in range(1, order + 1):
        columns.append(transformed[order - lag : transformed.size - lag])
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    variance = float(np.mean(residual**2))
    if variance <= 0.0 or not np.isfinite(variance):
        return np.inf, coefficients
    parameter_count = coefficients.size + 1
    aic = response.size * np.log(variance) + 2.0 * parameter_count
    return float(aic), coefficients


def _ar_one_step_and_forecast(
    series: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    candidates = []
    for difference in (0, 1):
        for order in (0, 1, 2):
            aic, coefficients = _fit_ar_candidate(series, difference, order)
            candidates.append((aic, difference, order, coefficients))
    _, difference, order, coefficients = min(candidates, key=lambda item: item[0])

    fitted = np.zeros_like(series)
    transformed = np.diff(series, n=difference) if difference else series.copy()
    transformed_fitted = np.full_like(transformed, coefficients[0])
    for index in range(order, transformed.size):
        value = coefficients[0]
        for lag in range(1, order + 1):
            value += coefficients[lag] * transformed[index - lag]
        transformed_fitted[index] = value
    if difference == 0:
        fitted[:] = transformed_fitted
    else:
        fitted[0] = series[0]
        fitted[1:] = series[:-1] + transformed_fitted

    history = list(transformed)
    transformed_forecast = []
    for _ in range(horizon):
        value = float(coefficients[0])
        for lag in range(1, order + 1):
            value += float(coefficients[lag]) * history[-lag]
        history.append(value)
        transformed_forecast.append(value)
    if difference == 0:
        forecast = np.asarray(transformed_forecast)
    else:
        forecast = np.empty(horizon)
        level = float(series[-1])
        for index, change in enumerate(transformed_forecast):
            level += change
            forecast[index] = level
    return fitted, forecast


def arima_surface(outcome: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Fit a per-unit automatic AR or integrated-AR forecasting surface.

    The search uses conditional Gaussian AIC over AR orders zero through two
    and integration orders zero and one. It is the Python analogue of the
    per-unit automatic ARIMA call in the R experiment.
    """

    outcome, treatment = _validate_panel(outcome, treatment)
    surface = np.empty_like(outcome)
    for unit in range(outcome.shape[0]):
        observed_count = int(np.sum(treatment[unit] < 0.5))
        fitted, forecast = _ar_one_step_and_forecast(
            outcome[unit, :observed_count], outcome.shape[1] - observed_count
        )
        surface[unit, :observed_count] = fitted
        surface[unit, observed_count:] = forecast
    return surface


OUTCOME_MODELS = {
    "fe": two_way_fe_surface,
    "ife": interactive_fixed_effects_surface,
    "MCPanel": mcpanel_surface,
    "gsynth": generalized_synthetic_control_surface,
    "VAR": panel_var_surface,
    "arima": arima_surface,
}
