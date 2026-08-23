"""Compose a staggered-adoption panel with unit-specific trends."""

from dataclasses import dataclass

import numpy as np

from pypanelsim import core, primitives


@dataclass(frozen=True, slots=True)
class UnitTrendOutcome:
    """Untreated outcomes with unit intercepts, slopes, and white noise."""

    trend_scale: float = 0.05
    noise_scale: float = 0.5

    def generate(
        self,
        context: core.SimulationContext,
        rng: np.random.Generator,
    ) -> core.ComponentDraw:
        shape = (context.dimensions.n_units, context.dimensions.n_periods)
        intercepts = rng.normal(size=(context.dimensions.n_units, 1))
        slopes = rng.normal(
            scale=self.trend_scale,
            size=(context.dimensions.n_units, 1),
        )
        time = np.arange(context.dimensions.n_periods)[None, :]
        values = (
            intercepts
            + slopes * time
            + rng.normal(
                scale=self.noise_scale,
                size=shape,
            )
        )
        return core.ComponentDraw(
            values,
            {
                "trend_scale": self.trend_scale,
                "noise_scale": self.noise_scale,
            },
        )


simulator = core.PanelSimulator(
    name="staggered_unit_trends",
    dimensions=core.PanelDimensions(n_units=50, n_periods=20),
    assignment=primitives.StaggeredAdoption({40: 12, 41: 12, 42: 14, 43: 14}),
    outcome_model=UnitTrendOutcome(),
    effect_model=primitives.LinearRampEffect(slope=0.25),
)

panel = simulator.simulate(seed=20260819)
print(f"Generated {panel.name} with shape {panel.shape}")
print(f"Treated units: {panel.treated_units.tolist()}")
print(f"True ATT: {panel.true_att:.3f}")
