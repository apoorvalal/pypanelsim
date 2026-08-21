"""Baker DGP: biased vanilla TWFE and saturated event-study recovery."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

import pypanelsim as pps


def baker_frame(
    panel: pps.PanelDataset,
    config: pps.BakerPanelConfig,
) -> pd.DataFrame:
    """Convert a Baker panel to PyFixest's long-form cohort convention."""

    n_units, n_periods = panel.shape
    adoption_period = panel.adoption_times
    frame = pd.DataFrame(dict(panel.as_long_dict()))
    frame["state"] = frame["state"].astype(np.int64)
    frame["cohort_period"] = np.repeat(adoption_period, n_periods)
    frame["cohort_year"] = frame["cohort_period"] + config.start_year
    frame["year"] = frame["time"] + config.start_year
    frame["rel_time"] = np.tile(np.arange(n_periods), n_units) - np.repeat(
        adoption_period, n_periods
    )
    frame["tau"] = frame["treatment_effect"]
    return frame


def _truth_by_event_time(
    frame: pd.DataFrame,
    *,
    window: int,
    eligible_cohorts: tuple[int, ...] | None = None,
) -> pd.Series:
    sample = frame.loc[frame["rel_time"].between(-window, window)]
    if eligible_cohorts is not None:
        sample = sample.loc[sample["cohort_period"].isin(eligible_cohorts)]
    return sample.groupby("rel_time", observed=True)["tau"].mean()


def _twfe_results(
    fit,
    truth: pd.Series,
    *,
    window: int,
) -> pd.DataFrame:
    tidy = fit.tidy().reset_index()
    coefficient = tidy.columns[0]
    tidy["event_time"] = (
        tidy[coefficient].astype(str).str.extract(r"::(-?\d+(?:\.\d+)?)$")[0]
    ).astype(float)
    tidy = tidy.loc[tidy["event_time"].between(-window, window)].copy()
    tidy = tidy.rename(
        columns={
            "Estimate": "estimate",
            "Std. Error": "std_error",
            "2.5%": "lower",
            "97.5%": "upper",
        }
    )
    reference = pd.DataFrame(
        {
            "event_time": [-1.0],
            "estimate": [0.0],
            "std_error": [0.0],
            "lower": [0.0],
            "upper": [0.0],
        }
    )
    result = pd.concat(
        [
            tidy[["event_time", "estimate", "std_error", "lower", "upper"]],
            reference,
        ],
        ignore_index=True,
    ).sort_values("event_time")
    result["truth"] = result["event_time"].map(truth)
    return result.reset_index(drop=True)


def _saturated_results(
    fit,
    truth: pd.Series,
    *,
    window: int,
) -> pd.DataFrame:
    aggregate = fit.aggregate(agg="period", weighting="shares").reset_index()
    aggregate = aggregate.rename(
        columns={
            "period": "event_time",
            "Estimate": "estimate",
            "Std. Error": "std_error",
            "2.5%": "lower",
            "97.5%": "upper",
        }
    )
    aggregate = aggregate.loc[
        aggregate["event_time"].between(-window, window),
        ["event_time", "estimate", "std_error", "lower", "upper"],
    ].copy()
    reference = pd.DataFrame(
        {
            "event_time": [-1.0],
            "estimate": [0.0],
            "std_error": [0.0],
            "lower": [0.0],
            "upper": [0.0],
        }
    )
    result = pd.concat([aggregate, reference], ignore_index=True)
    numeric = ["event_time", "estimate", "std_error", "lower", "upper"]
    result[numeric] = result[numeric].astype(float)
    result = result.sort_values("event_time")
    result["truth"] = result["event_time"].map(truth)
    return result.reset_index(drop=True)


def fit_baker_event_studies(
    panel: pps.PanelDataset,
    config: pps.BakerPanelConfig,
    *,
    window: int = 5,
) -> dict[str, object]:
    """Fit vanilla TWFE and a saturated Sun-Abraham-style event study."""

    frame = baker_frame(panel, config)
    frame["event_time_binned"] = frame["rel_time"].clip(
        lower=-(window + 1), upper=window + 1
    )
    twfe_fit = pf.feols(
        "outcome ~ i(event_time_binned, ref=-1) | unit + time",
        frame,
        vcov={"CRV1": "state"},
    )
    twfe_truth = _truth_by_event_time(frame, window=window)
    twfe = _twfe_results(twfe_fit, twfe_truth, window=window)

    last_cohort = config.adoption_periods[-1]
    saturated_frame = frame.loc[
        (frame["time"] < last_cohort)
        & ((frame["treatment"] == 0.0) | (frame["rel_time"] <= window))
    ].copy()
    saturated_frame["cohort_for_estimator"] = saturated_frame["cohort_period"].mask(
        saturated_frame["cohort_period"] == last_cohort, 0
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The SaturatedEventStudyClass is currently in beta.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"(?s).*variables dropped due to multicollinearity.*",
        )
        saturated_fit = pf.event_study(
            saturated_frame,
            yname="outcome",
            idname="unit",
            tname="time",
            gname="cohort_for_estimator",
            estimator="saturated",
            att=False,
            cluster="state",
        )
    saturated_truth = _truth_by_event_time(
        frame,
        window=window,
        eligible_cohorts=config.adoption_periods[:-1],
    )
    saturated = _saturated_results(
        saturated_fit,
        saturated_truth,
        window=window,
    )
    return {
        "frame": frame,
        "twfe_fit": twfe_fit,
        "saturated_fit": saturated_fit,
        "twfe": twfe,
        "saturated": saturated,
    }


def _post_rmse(results: pd.DataFrame) -> float:
    post = results.loc[results["event_time"] >= 0.0]
    return float(np.sqrt(np.mean((post["estimate"] - post["truth"]) ** 2)))


def plot_baker_event_studies(
    estimates: dict[str, object],
) -> tuple[plt.Figure, np.ndarray]:
    """Plot vanilla and saturated estimates against their estimands."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    specifications = (
        ("Vanilla relative-time TWFE", estimates["twfe"]),
        ("Saturated cohort by event time", estimates["saturated"]),
    )
    for axis, (title, raw_results) in zip(axes, specifications, strict=True):
        results = raw_results
        axis.errorbar(
            results["event_time"],
            results["estimate"],
            yerr=np.vstack(
                (
                    results["estimate"] - results["lower"],
                    results["upper"] - results["estimate"],
                )
            ),
            fmt="o-",
            color="#3D5A80",
            capsize=3,
            label="PyFixest estimate (95% CI)",
        )
        axis.plot(
            results["event_time"],
            results["truth"],
            "--",
            color="#C44536",
            linewidth=2,
            label="True cohort-weighted effect",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(-0.5, color="grey", linestyle=":", linewidth=1.0)
        axis.set_xticks(range(-5, 6))
        axis.set_xlabel("Event time")
        axis.set_title(f"{title}\npost-period RMSE = {_post_rmse(results):.3f}")
    axes[0].set_ylabel("Treatment effect")
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Baker staggered-adoption DGP: TWFE contamination and saturated recovery",
        fontsize=14,
    )
    return figure, axes


def render_baker_event_study(
    output: Path | None = None,
    *,
    seed: int = 28101695,
) -> Path:
    """Render PyFixest event-study estimates against known DGP truth."""

    if output is None:
        output = Path(__file__).with_name("baker_twfe_event_study.png")
    config = pps.BakerPanelConfig()
    panel = pps.baker(config=config, seed=seed)
    estimates = fit_baker_event_studies(panel, config)

    figure, _ = plot_baker_event_studies(estimates)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print(f"Vanilla TWFE post-period RMSE: {_post_rmse(estimates['twfe']):.3f}")
    print(
        "Saturated event-study post-period RMSE: "
        f"{_post_rmse(estimates['saturated']):.3f}"
    )
    return output


if __name__ == "__main__":
    print(render_baker_event_study())
