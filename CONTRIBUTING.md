# Contributing

## Environment

Use the repository `uv` environment:

```bash
uv sync --locked
```

Do not install development dependencies with `pip`.

## Required checks

Run all checks before committing:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

## Component requirements

- Use the provided NumPy generator. Do not use global random state.
- Return a full `(n_units, n_periods)` matrix.
- Keep treatment binary.
- Return realized effects only in treated cells.
- Store latent simulation truth in component metadata.
- Add seeded-reproducibility and invalid-input tests.
- Keep estimator imports out of `src/pypanelsim`.
- Use professional, emoji-free code, comments, logs, and error messages.

## Canonical design changes

Changes to canonical probability laws or draw order require a versioned
regression test and a clear changelog entry. Parameter-only extensions should
remain backward compatible when possible.
