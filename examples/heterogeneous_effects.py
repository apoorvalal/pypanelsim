"""Treatment-effect heterogeneity across units and time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pypanelsim import core, primitives


@dataclass(frozen=True, slots=True)
class FeatureDrivenOutcome:
    """Untreated outcomes that share the baseline effect modifiers."""

    noise_scale: float = 0.3

    def generate(self, context, rng):
        trend = context.time_features[:, 0]
        cycle = context.time_features[:, 1]
        values = (
            0.8 * context.observables[:, :1]
            + 0.5 * context.observables[:, 1:2] * trend
            + 1.1 * context.unobservables[:, :1] * cycle
        )
        values += rng.normal(scale=self.noise_scale, size=values.shape)
        return core.ComponentDraw(values, {"kind": "feature_driven_outcome"})


@dataclass(frozen=True, slots=True)
class TrendCycleTimeFeatures:
    """Deterministic trend and cycle shared by outcomes and effects."""

    def generate(self, dimensions, rng):
        del rng
        trend = np.linspace(-1.0, 1.0, dimensions.n_periods)
        cycle = np.sin(np.linspace(0.0, 2.5 * np.pi, dimensions.n_periods))
        return core.TimeFeatureDraw(
            np.column_stack((trend, cycle)),
            {"kind": "trend_cycle", "feature_names": ("trend", "cycle")},
        )


def heterogeneous_effect_dgp() -> core.PanelSimulator:
    """Build a randomized panel with tau_it = f(X_i, U_i, V_t)."""

    return core.PanelSimulator(
        name="heterogeneous_effects",
        dimensions=core.PanelDimensions(n_units=120, n_periods=40),
        feature_model=primitives.GaussianUnitFeatures(
            n_observables=2,
            n_unobservables=1,
        ),
        time_feature_model=TrendCycleTimeFeatures(),
        assignment=primitives.RandomizedSingleCohortAssignment(
            n_treated=40,
            adoption_period=28,
        ),
        outcome_model=FeatureDrivenOutcome(),
        effect_model=lambda x: (
            (
                1.0
                + 0.45 * x.observables[:, 0]
                - 0.30 * x.observables[:, 1]
                + 0.80 * x.unobservables[:, 0]
            )[:, None]
            * (1.0 + 0.35 * x.time_features[:, 0] + 0.25 * x.time_features[:, 1])[
                None, :
            ]
        ),
    )


def render_heterogeneous_effects(
    output: Path | None = None,
    *,
    seed: int = 20260821,
) -> Path:
    """Render Y, W, and the realized heterogeneous effect matrix."""

    if output is None:
        output = Path(__file__).with_name("heterogeneous_effects.png")
    panel = heterogeneous_effect_dgp().simulate(seed=seed)
    effect_surface = panel.effect_surface
    order = np.argsort(effect_surface.mean(axis=1), kind="stable")

    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    panels = (
        (panel.outcome, "Observed outcome Y", "coolwarm", None, None),
        (panel.treatment, "Treatment W", "Greys", 0.0, 1.0),
        (
            panel.treatment_effect,
            r"Realized effect $\tau_{it} W_{it}$",
            "coolwarm",
            None,
            None,
        ),
    )
    for axis, (values, title, cmap, lower, upper) in zip(axes, panels, strict=True):
        image = axis.matshow(
            values[order],
            aspect="auto",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
        )
        axis.set_title(title)
        axis.set_xlabel("period")
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    axes[0].set_ylabel(r"units, sorted by mean $\tau_{it}$")
    figure.suptitle(
        r"Unit-time effects: $\tau_{it}=f(X_i,U_i,V_t)$",
        fontsize=14,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


if __name__ == "__main__":
    print(render_heterogeneous_effects())
