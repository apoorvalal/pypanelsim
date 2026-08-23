"""Named design suites adapted from downstream research projects."""

from .att_dml import (
    ATTDMClusteredConfig,
    ATTDMConditionalConfig,
    ATTDMLatentConfig,
    att_dml,
    att_dml_design,
    available_att_dml_designs,
)
from .gsynth import GSynthCompositeConfig, gsynth_composite, gsynth_composite_design
from .lepskii import (
    LepskiiPanelConfig,
    available_lepskii_designs,
    lepskii,
    lepskii_design,
    lepskii_profiles,
)
from .regression_compression import (
    AnscombePanelConfig,
    RegressionCompressionConfig,
    anscombe,
    anscombe_design,
    available_anscombe_designs,
    regression_compression,
    regression_compression_design,
)

__all__ = [
    "ATTDMClusteredConfig",
    "ATTDMConditionalConfig",
    "ATTDMLatentConfig",
    "AnscombePanelConfig",
    "GSynthCompositeConfig",
    "LepskiiPanelConfig",
    "RegressionCompressionConfig",
    "anscombe",
    "anscombe_design",
    "att_dml",
    "att_dml_design",
    "available_anscombe_designs",
    "available_att_dml_designs",
    "available_lepskii_designs",
    "gsynth_composite",
    "gsynth_composite_design",
    "lepskii",
    "lepskii_design",
    "lepskii_profiles",
    "regression_compression",
    "regression_compression_design",
]
