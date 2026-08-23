"""Run the matched Python Monte Carlo experiments for the slide DGPs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import crabbymetrics as cm
import numpy as np

from pypanelsim import designs

from .outcome_models import OUTCOME_MODELS

CASES = {
    "classic_a0": (
        "classic_factor",
        "NN=200;a=0",
        designs.classic_factor,
        {"overlap": 0.0},
    ),
    "classic_a1": (
        "classic_factor",
        "NN=200;a=1",
        designs.classic_factor,
        {"overlap": 1.0},
    ),
    "weak_a0": (
        "weak_factor",
        "NN=200;a=0",
        designs.weak_factor,
        {"overlap": 0.0},
    ),
    "weak_a1": (
        "weak_factor",
        "NN=200;a=1",
        designs.weak_factor,
        {"overlap": 1.0},
    ),
    "mixed_a0": (
        "mixed_factor",
        "NN=200;a=0",
        designs.mixed_factor,
        {"overlap": 0.0},
    ),
    "mixed_a1": (
        "mixed_factor",
        "NN=200;a=1",
        designs.mixed_factor,
        {"overlap": 1.0},
    ),
    "time_a02_i0": (
        "time_series",
        "NN=200;a=0.2;I=0",
        designs.time_series,
        {"coefficient": 0.2, "integrated": False},
    ),
    "time_a02_i1": (
        "time_series",
        "NN=200;a=0.2;I=1",
        designs.time_series,
        {"coefficient": 0.2, "integrated": True},
    ),
    "time_a09_i0": (
        "time_series",
        "NN=200;a=0.9;I=0",
        designs.time_series,
        {"coefficient": 0.9, "integrated": False},
    ),
    "time_a09_i1": (
        "time_series",
        "NN=200;a=0.9;I=1",
        designs.time_series,
        {"coefficient": 0.9, "integrated": True},
    ),
    "synth_active01": (
        "synthetic_control",
        "NN=200;active=0.1;unitb=ridge",
        designs.synthetic_control,
        {"active_share": 0.1},
    ),
    "synth_active05": (
        "synthetic_control",
        "NN=200;active=0.5;unitb=ridge",
        designs.synthetic_control,
        {"active_share": 0.5},
    ),
    "synth_active01_penscm": (
        "synthetic_control",
        "NN=200;active=0.1;unitb=penSCM",
        designs.synthetic_control,
        {"active_share": 0.1},
    ),
    "synth_active05_penscm": (
        "synthetic_control",
        "NN=200;active=0.5;unitb=penSCM",
        designs.synthetic_control,
        {"active_share": 0.5},
    ),
}

CASE_UNIT_LOSS = {
    "synth_active01_penscm": "penalized_scm",
    "synth_active05_penscm": "penalized_scm",
}

OUTCOME_PARAMETERS = {
    "classic_factor": {
        "ife": {"tolerance": 1e-5, "max_iterations": 2_000},
        "VAR": {"lags": 2},
    },
    "weak_factor": {
        "ife": {"tolerance": 1e-5, "max_iterations": 2_000},
        "VAR": {"lags": 2},
    },
    "mixed_factor": {
        "ife": {"tolerance": 1e-4, "max_iterations": 4_000},
        "VAR": {"lags": 1},
    },
    "synthetic_control": {
        "ife": {"tolerance": 1e-4, "max_iterations": 4_000},
        "VAR": {"lags": 1},
    },
    "time_series": {
        "ife": {"tolerance": 1e-4, "max_iterations": 4_000},
        "VAR": {"lags": 4},
    },
}


BASE_ESTIMATORS = {
    "u-balance": {"balance": "unit"},
    "d-balance": {"balance": "double"},
}

OUTCOME_ESTIMATORS = {
    "outcome": {"balance": "none"},
    "aug-d-balance": {"balance": "double"},
    "aug-d-balance (residual)": {"balance": "double", "balance_on": "residual"},
    "aug-d-balance (individual)": {
        "balance": "double",
        "unit_target": "individual",
        "time_target": "period",
    },
    "aug-d-balance (residual individual)": {
        "balance": "double",
        "unit_target": "individual",
        "time_target": "period",
        "balance_on": "residual",
    },
}


def _fit_one(case_and_seed: tuple[str, int]) -> list[dict[str, object]]:
    case_name, seed = case_and_seed
    family, setting, generator, parameters = CASES[case_name]
    panel = generator(seed=seed, **parameters)
    rows = []

    for frame, options in BASE_ESTIMATORS.items():
        model = cm.AugmentedBalancing(
            max_iterations=1000,
            unit_loss=CASE_UNIT_LOSS.get(case_name, "ridge"),
            **options,
        )
        model.fit(panel.outcome, panel.treatment)
        estimate = float(model.summary()["att"])
        rows.append(
            {
                "family": family,
                "setting": setting,
                "seed": seed,
                "frame": frame,
                "outcome": "fe",
                "estimate": estimate,
                "truth": panel.true_att,
                "error": estimate - panel.true_att,
            }
        )

    outcome_names = ["fe", "ife", "MCPanel", "gsynth", "VAR"]
    if family == "time_series":
        outcome_names.append("arima")
    for outcome_name in outcome_names:
        outcome_model = OUTCOME_MODELS[outcome_name](
            panel.outcome,
            panel.treatment,
            **OUTCOME_PARAMETERS.get(family, {}).get(outcome_name, {}),
        )
        for frame, options in OUTCOME_ESTIMATORS.items():
            model = cm.AugmentedBalancing(
                max_iterations=1000,
                unit_loss=CASE_UNIT_LOSS.get(case_name, "ridge"),
                **options,
            )
            model.fit(panel.outcome, panel.treatment, outcome_model)
            estimate = float(model.summary()["att"])
            rows.append(
                {
                    "family": family,
                    "setting": setting,
                    "seed": seed,
                    "frame": frame,
                    "outcome": outcome_name,
                    "estimate": estimate,
                    "truth": panel.true_att,
                    "error": estimate - panel.true_att,
                }
            )
    return rows


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        key = tuple(
            str(row[field]) for field in ("family", "setting", "frame", "outcome")
        )
        groups.setdefault(key, []).append(float(row["error"]))
    summaries = []
    for (family, setting, frame, outcome), errors in sorted(groups.items()):
        values = np.asarray(errors)
        squared = values**2
        rmse = math.sqrt(float(squared.mean()))
        rmse_mcse = float(squared.std(ddof=1) / math.sqrt(values.size) / (2.0 * rmse))
        summaries.append(
            {
                "family": family,
                "setting": setting,
                "frame": frame,
                "outcome": outcome,
                "rmse": rmse,
                "rmse_mcse": rmse_mcse,
                "mean_bias": float(values.mean()),
                "mean_absolute_bias": float(np.abs(values).mean()),
                "n": values.size,
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--cases", nargs="+", choices=sorted(CASES), default=sorted(CASES)
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).parents[1] / "results",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="keep cached settings that are not included in this run",
    )
    arguments = parser.parse_args()
    if arguments.replications <= 1:
        parser.error("--replications must be greater than one")
    raw_path = arguments.output_directory / "python_replications.csv"
    summary_path = arguments.output_directory / "python_rmse.csv"
    retained: list[dict[str, object]] = []
    cached_selected: list[dict[str, object]] = []
    complete_draws: set[tuple[str, str, int]] = set()
    selected_settings = {(CASES[name][0], CASES[name][1]) for name in arguments.cases}
    if arguments.append and raw_path.exists():
        existing = _read_csv(raw_path)
        retained = [
            row
            for row in existing
            if (str(row["family"]), str(row["setting"])) not in selected_settings
        ]
        selected_existing = [
            row
            for row in existing
            if (str(row["family"]), str(row["setting"])) in selected_settings
        ]
        counts = Counter(
            (str(row["family"]), str(row["setting"]), int(row["seed"]))
            for row in selected_existing
        )
        for case_name in arguments.cases:
            family, setting, _, _ = CASES[case_name]
            expected_rows = 32 if family == "time_series" else 27
            for seed in range(1, arguments.replications + 1):
                key = (family, setting, seed)
                if counts[key] == expected_rows:
                    complete_draws.add(key)
        cached_selected = [
            row
            for row in selected_existing
            if (str(row["family"]), str(row["setting"]), int(row["seed"]))
            in complete_draws
        ]
    tasks = [
        (case_name, seed)
        for case_name in arguments.cases
        for seed in range(1, arguments.replications + 1)
        if (CASES[case_name][0], CASES[case_name][1], seed) not in complete_draws
    ]
    started = time.monotonic()
    rows: list[dict[str, object]] = cached_selected
    if complete_draws:
        print(
            f"Loaded {len(complete_draws)} complete panel draws from the result cache"
        )
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for completed, result in enumerate(executor.map(_fit_one, tasks), start=1):
            rows.extend(result)
            if completed % 25 == 0 or completed == len(tasks):
                elapsed = time.monotonic() - started
                print(
                    f"Completed {completed} of {len(tasks)} panel draws "
                    f"in {elapsed:.1f} seconds",
                    flush=True,
                )
                checkpoint = retained + rows
                _write_csv(raw_path, checkpoint)
                _write_csv(summary_path, _summaries(checkpoint))
    combined = retained + rows
    _write_csv(raw_path, combined)
    _write_csv(summary_path, _summaries(combined))
    print(f"Wrote replication results to {raw_path}")
    print(f"Wrote RMSE summaries to {summary_path}")


if __name__ == "__main__":
    main()
