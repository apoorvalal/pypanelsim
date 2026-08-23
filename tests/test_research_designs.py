import numpy as np
import pytest

import pypanelsim as pps


@pytest.mark.parametrize("name", pps.available_lepskii_designs())
def test_every_lepskii_design_simulates_and_exposes_supported_truth(name: str) -> None:
    config = pps.LepskiiPanelConfig(n_units=200)
    panel = pps.lepskii_design(name, config=config).simulate(seed=9)
    truth = panel.truth.event_study(event_times=range(-5, 13))

    assert panel.shape == (200, 36)
    assert len(np.unique(panel.adoption_times[panel.ever_treated])) == 9
    assert truth["effect"].shape == (18,)
    assert np.all(truth["supported_cohorts"] > 0)


def test_lepskii_profiles_reproduce_key_draft_shapes() -> None:
    config = pps.LepskiiPanelConfig(n_units=200)
    constant = pps.lepskii_profiles("complete_homogeneity", config=config)
    signed = pps.lepskii_profiles("no_pool_signed_paths", config=config)

    np.testing.assert_allclose(constant[8], 1.4)
    assert signed[8][-1] > 1.0
    assert signed[10][-1] < -1.0


@pytest.mark.parametrize("name", pps.available_att_dml_designs())
def test_every_att_dml_design_simulates(name: str) -> None:
    if name.startswith("conditional"):
        config = pps.ATTDMConditionalConfig(
            n_units=120,
            n_features=10,
            n_active=3,
            trend_strength=2.0,
        )
    elif name.startswith("latent"):
        config = pps.ATTDMLatentConfig(n_units=120, n_periods=7)
    else:
        config = pps.ATTDMClusteredConfig(
            n_units=40,
            n_periods=20,
            n_treated=8,
            adoption_period=12,
            n_clusters=5,
        )
    panel = pps.att_dml_design(name, config=config).simulate(seed=10)

    assert np.all(np.isfinite(panel.outcome))
    assert panel.is_absorbing
    if name.startswith("conditional"):
        assert panel.unit_covariates.shape == (120, 10)
        assert panel.true_att == pytest.approx(3.0, abs=0.1)
        if name.endswith("_nonlinear"):
            required = {10, 11, 12}
            assert required.issubset(
                set(panel.metadata["assignment"]["active_features"])
            )
            assert required.issubset(set(panel.metadata["outcome"]["trend_active"]))


def test_att_dml_baseline_discrepancy_is_an_explicit_switch() -> None:
    code_law = pps.ATTDMConditionalConfig(
        n_units=200,
        n_features=10,
        n_active=3,
        treated_baseline_shift=False,
    )
    manuscript_law = pps.ATTDMConditionalConfig(
        n_units=200,
        n_features=10,
        n_active=3,
        treated_baseline_shift=True,
    )
    code_panel = pps.att_dml_design(
        "conditional_parallel_trends", config=code_law
    ).simulate(streams=pps.SimulationSeeds.from_seed(21))
    manuscript_panel = pps.att_dml_design(
        "conditional_parallel_trends", config=manuscript_law
    ).simulate(streams=pps.SimulationSeeds.from_seed(21))

    assert code_panel.metadata["outcome"]["treated_baseline_shift"] is False
    assert manuscript_panel.metadata["outcome"]["treated_baseline_shift"] is True
    assert not np.array_equal(
        code_panel.untreated_outcome[:, 0],
        manuscript_panel.untreated_outcome[:, 0],
    )


@pytest.mark.parametrize("name", pps.available_anscombe_designs())
def test_every_regression_compression_anscombe_design_simulates(name: str) -> None:
    panel = pps.anscombe_design(
        name, config=pps.AnscombePanelConfig(n_units=80)
    ).simulate(seed=12)

    assert panel.shape == (80, 20)
    if name in {"zero", "time", "cohort_time"}:
        assert abs(panel.true_att) < 1e-12


def test_regression_compression_main_design_and_gsynth_composite() -> None:
    compression = pps.regression_compression_design(
        config=pps.RegressionCompressionConfig(
            n_units=80, n_periods=20, adoption_period=10, n_treated=30
        )
    ).simulate(seed=13)
    composite = pps.gsynth_composite_design(
        config=pps.GSynthCompositeConfig(n_control=40, n_treated=10, n_pre=10, n_post=5)
    ).simulate(seed=14)

    assert compression.metadata["outcome"]["kind"] == "sum_outcome"
    assert len(compression.metadata["outcome"]["components"]) == 3
    assert composite.metadata["outcome"]["kind"] == "sum_outcome"
    weights = [
        component["weight"] for component in composite.metadata["outcome"]["components"]
    ]
    np.testing.assert_allclose(weights, [0.25] * 4)


def test_one_draw_research_convenience_functions() -> None:
    assert pps.lepskii(
        "complete_homogeneity",
        config=pps.LepskiiPanelConfig(n_units=200),
        seed=1,
    ).shape == (200, 36)
    assert pps.att_dml(
        "latent_parallel",
        config=pps.ATTDMLatentConfig(n_units=50),
        seed=1,
    ).shape == (50, 7)
    assert pps.regression_compression(
        config=pps.RegressionCompressionConfig(
            n_units=50, n_periods=10, adoption_period=5, n_treated=20
        ),
        seed=1,
    ).shape == (50, 10)
    assert pps.anscombe(
        "zero", config=pps.AnscombePanelConfig(n_units=40), seed=1
    ).shape == (40, 20)
    assert pps.gsynth_composite(
        config=pps.GSynthCompositeConfig(n_control=20, n_treated=5, n_pre=8, n_post=3),
        seed=1,
    ).shape == (25, 11)
