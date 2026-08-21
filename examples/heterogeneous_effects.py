"""Treatment-effect heterogeneity from observed and latent unit features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pypanelsim as pps


@dataclass(frozen=True, slots=True)
class FeatureDrivenOutcome:
    """Untreated outcomes that share the baseline effect modifiers."""

    noise_scale: float = 0.3

    def generate(self, context, rng):
        time = np.linspace(-1.0, 1.0, context.dimensions.n_periods)
        cycle = np.sin(np.linspace(0.0, 2.5 * np.pi, context.dimensions.n_periods))
        values = (
            0.8 * context.observables[:, :1]
            + 0.5 * context.observables[:, 1:2] * time
            + 1.1 * context.unobservables[:, :1] * cycle
        )
        values += rng.normal(scale=self.noise_scale, size=values.shape)
        return pps.ComponentDraw(values, {"kind": "feature_driven_outcome"})


def heterogeneous_effect_dgp() -> pps.PanelSimulator:
    """Build a randomized panel with tau_i = f(X_i, U_i)."""

    return pps.PanelSimulator(
        name="heterogeneous_effects",
        dimensions=pps.PanelDimensions(n_units=120, n_periods=40),
        feature_model=pps.GaussianUnitFeatures(
            n_observables=2,
            n_unobservables=1,
        ),
        assignment=pps.RandomizedSingleCohortAssignment(
            n_treated=40,
            adoption_period=28,
        ),
        outcome_model=FeatureDrivenOutcome(),
        effect_model=lambda x: (
            1.0
            + 0.45 * x.observables[:, 0]
            - 0.30 * x.observables[:, 1]
            + 0.80 * x.unobservables[:, 0]
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
    unit_effects = panel.metadata["effect"]["unit_effects"]
    order = np.argsort(unit_effects, kind="stable")

    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    panels = (
        (panel.outcome, "Observed outcome Y", "coolwarm", None, None),
        (panel.treatment, "Treatment W", "Greys", 0.0, 1.0),
        (
            panel.treatment_effect,
            r"Realized effect $\tau_i W_{it}$",
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
    axes[0].set_ylabel(r"units, sorted by $\tau_i$")
    figure.suptitle(
        r"Heterogeneous effects: $\tau_i=1+0.45X_{i1}-0.30X_{i2}+0.80U_i$",
        fontsize=14,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


if __name__ == "__main__":
    print(render_heterogeneous_effects())
