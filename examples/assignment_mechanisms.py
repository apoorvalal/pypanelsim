"""Visual comparison of randomized, selected, and cohort assignment DGPs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pypanelsim import core, primitives


@dataclass(frozen=True, slots=True)
class FeatureDrivenOutcome:
    """Outcome process sharing observed and latent features with assignment."""

    noise_scale: float = 0.35

    def generate(self, context, rng):
        time = np.linspace(-1.0, 1.0, context.dimensions.n_periods)
        drift = np.linspace(0.0, 2.0, context.dimensions.n_periods)
        cycle = np.sin(np.linspace(0.0, 3.0 * np.pi, context.dimensions.n_periods))
        observed = (
            0.8 * context.observables[:, :1] + 0.6 * context.observables[:, 1:2] * time
        )
        latent = (
            1.3 * context.unobservables[:, :1] * drift
            + 0.9 * context.unobservables[:, 1:2] * cycle
        )
        noise = rng.normal(
            scale=self.noise_scale,
            size=(context.dimensions.n_units, context.dimensions.n_periods),
        )
        return core.ComponentDraw(
            observed + latent + noise,
            {"kind": "feature_driven_factor_outcome"},
        )


def assignment_dgps() -> dict[str, core.PanelSimulator]:
    """Construct four DGPs with a common feature-driven outcome process."""

    dimensions = core.PanelDimensions(n_units=120, n_periods=48)
    features = primitives.GaussianUnitFeatures(
        n_observables=2,
        n_unobservables=2,
    )
    outcome = FeatureDrivenOutcome()
    effect = primitives.LinearRampEffect(slope=0.12)

    def simulator(name, assignment):
        return core.PanelSimulator(
            name=name,
            dimensions=dimensions,
            feature_model=features,
            assignment=assignment,
            outcome_model=outcome,
            effect_model=effect,
        )

    return {
        "Randomized fixed cohort": simulator(
            "randomized",
            primitives.RandomizedSingleCohortAssignment(
                n_treated=30,
                adoption_period=32,
            ),
        ),
        "Logit selection on observed X": simulator(
            "selection_observed",
            primitives.BinaryLogitAssignment(
                adoption_period=32,
                intercept=-1.1,
                observable_coefficients=(1.2, -0.8),
                unobservable_coefficients=(0.0, 0.0),
            ),
        ),
        "Logit selection on latent factors": simulator(
            "selection_latent",
            primitives.BinaryLogitAssignment(
                adoption_period=32,
                intercept=-1.1,
                observable_coefficients=(0.0, 0.0),
                unobservable_coefficients=(1.4, -1.0),
            ),
        ),
        "Cohort GPS on observed X": simulator(
            "cohort_gps",
            primitives.GeneralizedPropensityAssignment(
                adoption_periods=(24, 30, 36),
                intercepts=(-0.7, -0.9, -1.1),
                observable_coefficients=(
                    (1.0, -0.5),
                    (0.2, 0.8),
                    (-0.8, 0.3),
                ),
                unobservable_coefficients=((0.0, 0.0),) * 3,
            ),
        ),
    }


def render_assignment_dgps(
    output: Path | None = None,
    *,
    seed: int = 20260820,
) -> Path:
    """Render outcome and treatment matrices for each assignment DGP."""

    if output is None:
        output = Path(__file__).with_name("assignment_mechanisms.png")
    simulators = assignment_dgps()
    figure, axes = plt.subplots(
        len(simulators),
        2,
        figsize=(12, 13),
        constrained_layout=True,
    )
    for row, (label, simulator) in enumerate(simulators.items()):
        panel = simulator.simulate(seed=seed)
        order = np.argsort(panel.adoption_times, kind="stable")
        outcome_image = axes[row, 0].matshow(
            panel.outcome[order],
            aspect="auto",
            cmap="coolwarm",
        )
        treatment_image = axes[row, 1].matshow(
            panel.treatment[order],
            aspect="auto",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 0].set_title(f"{label}: Y")
        axes[row, 1].set_title(f"{label}: W")
        axes[row, 0].set_ylabel("units, sorted by adoption")
        for axis in axes[row]:
            axis.set_xlabel("period")
        figure.colorbar(outcome_image, ax=axes[row, 0], fraction=0.025, pad=0.02)
        figure.colorbar(treatment_image, ax=axes[row, 1], fraction=0.025, pad=0.02)
    figure.suptitle("Panel DGPs under alternative assignment mechanisms", fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


if __name__ == "__main__":
    print(render_assignment_dgps())
