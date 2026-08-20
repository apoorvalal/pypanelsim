"""Pass one panel to an estimator that accepts outcome and treatment arrays."""

from dataclasses import dataclass, field

import numpy as np

import pypanelsim as pps


@dataclass(slots=True)
class TreatedCellMean:
    """Minimal downstream estimator used to demonstrate array interchange."""

    estimate_: float = field(init=False)

    def fit(self, outcome: np.ndarray, treatment: np.ndarray) -> "TreatedCellMean":
        if outcome.shape != treatment.shape:
            raise ValueError("outcome and treatment must have the same shape")
        self.estimate_ = float(outcome[treatment == 1.0].mean())
        return self


panel = pps.classic_factor(overlap=1.0, seed=42)
estimator = TreatedCellMean().fit(*panel.as_arrays())

print(f"Estimator input shape: {panel.shape}")
print(f"Estimated treated-cell mean: {estimator.estimate_:.3f}")
print(f"Simulation true ATT: {panel.true_att:.3f}")
