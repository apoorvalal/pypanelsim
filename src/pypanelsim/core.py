"""Public contracts for composing and inspecting panel simulations.

Use this module for the stable data and orchestration layer. Probability laws
live in :mod:`pypanelsim.primitives`, while configured benchmark families live
in :mod:`pypanelsim.designs`.
"""

from .components import (
    AssignmentContext,
    AssignmentModel,
    ComponentDraw,
    EffectModel,
    OutcomeModel,
    PanelDimensions,
    SimulationContext,
    TimeFeatureDraw,
    TimeFeatureModel,
    UnitFeatureDraw,
    UnitFeatureModel,
)
from .data import PanelDataset
from .registry import DGPRegistry
from .simulator import PanelSimulator, SimulationSeeds, resolve_rng
from .truth import PanelTruth

__all__ = [
    "AssignmentContext",
    "AssignmentModel",
    "ComponentDraw",
    "DGPRegistry",
    "EffectModel",
    "OutcomeModel",
    "PanelDataset",
    "PanelDimensions",
    "PanelSimulator",
    "PanelTruth",
    "SimulationContext",
    "SimulationSeeds",
    "TimeFeatureDraw",
    "TimeFeatureModel",
    "UnitFeatureDraw",
    "UnitFeatureModel",
    "resolve_rng",
]
