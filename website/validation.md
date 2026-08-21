# Validation record

## Package checks

The 0.1.0 repository was validated with:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run python examples/custom_dgp.py
uv run python examples/estimator_interop.py
uv run python examples/assignment_mechanisms.py
uv build
```

The source tree contains no crabbymetrics import. NumPy and Matplotlib are the
runtime dependencies declared in `pyproject.toml`.

The wheel was installed into a fresh temporary `uv` environment. A smoke test
then created a registered classic-factor design, generated a `(200, 50)` panel,
extracted estimator arrays, and recovered the canonical true ATT of 1.1.

## Migration parity

Before the embedded implementation was removed from the balancing reproduction
project, the old and new packages were loaded together. Each design used the
same small configuration and seed 101:

```python
CanonicalPanelConfig(
    n_control=12,
    n_treated=4,
    n_pre=8,
    n_post=3,
)
```

The audit compared `outcome`, `untreated_outcome`, `treatment`, and
`treatment_effect` with exact NumPy array equality.

| Design | Parameters | Result |
|---|---|---|
| Classic factor | `overlap=1.0` | Exact |
| Weak factor | `overlap=1.0` | Exact |
| Synthetic control | `active_share=0.25` | Exact |
| Factor-synthetic | `overlap=1.0` | Exact |
| Stationary time series | `coefficient=0.9`, `integrated=False` | Exact |
| Integrated time series | `coefficient=0.2`, `integrated=True` | Exact |
| Mixed factor | `overlap=1.0` | Exact |

The migration changes object ownership and metadata layout, not canonical
outcomes or treatment arrays.

## Downstream reproduction check

The separate balancing reproduction was changed to consume `pypanelsim` as an
editable sibling dependency. Its tests passed after removal of the embedded
source. Its self-contained Quarto report then executed all 25 code cells and
rendered successfully with the same cached 398-cell R/Python comparison.

This downstream check establishes array-contract compatibility with the
current crabbymetrics experiment runner. It does not make crabbymetrics a
runtime or test dependency of `pypanelsim`.

## What is not asserted

NumPy and R do not share seeded random streams. The migration audit verifies
Python-to-Python extraction parity. The separate reproduction report evaluates
R/Python agreement with same-panel estimator fixtures and Monte Carlo standard
errors.
