# pypanelsim

`pypanelsim` is a small, estimator-neutral Python library for synthetic panel
data. It separates treatment assignment, untreated outcomes, and treatment
effects into composable components. A simulation returns validated NumPy
arrays that can be passed to any downstream estimator.

The complete API, canonical designs, extension guide, and migration notes are
documented in this repository. The package has no estimator dependency.

```python
import pypanelsim as pps

simulator = pps.classic_factor_design(overlap=1.0)
panel = simulator.simulate(seed=42)

y, w = panel.as_arrays()
print(y.shape, w.shape, panel.true_att)
```

Development uses `uv`:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

