"""Dependency-free causal-truth summaries for simulated panels."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .data import PanelDataset


def _frozen_columns(**columns: Any) -> MappingProxyType[str, NDArray[Any]]:
    frozen: dict[str, NDArray[Any]] = {}
    for name, values in columns.items():
        array = np.asarray(values)
        array = np.array(array, copy=True)
        array.setflags(write=False)
        frozen[name] = array
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class PanelTruth:
    """Causal truth derived from one immutable :class:`PanelDataset`.

    The summaries use the realized adoption schedule. They do not impose an
    estimator or extrapolate effects beyond cohort-period cells supported by
    the simulated panel.
    """

    panel: PanelDataset

    def _cohorts(self) -> NDArray[np.int64]:
        if not self.panel.is_absorbing:
            raise ValueError("cohort truth requires absorbing treatment")
        adoption = self.panel.adoption_times
        cohorts = np.unique(adoption[adoption < self.panel.n_periods])
        if cohorts.size == 0:
            raise ValueError("cohort truth requires at least one treated cohort")
        cohorts = cohorts.astype(np.int64, copy=False)
        cohorts.setflags(write=False)
        return cohorts

    def cohort_event(
        self,
        event_times: Any | None = None,
    ) -> MappingProxyType[str, NDArray[Any]]:
        """Return supported cohort-by-event-time truth in long columns."""

        cohorts = self._cohorts()
        if event_times is None:
            lower = -int(cohorts.max())
            upper = self.panel.n_periods - int(cohorts.min())
            events = np.arange(lower, upper, dtype=np.int64)
        else:
            events = np.asarray(tuple(event_times), dtype=np.int64)
            if events.ndim != 1 or events.size == 0:
                raise ValueError("event_times must be a nonempty one-dimensional grid")
            if np.unique(events).size != events.size:
                raise ValueError("event_times must contain unique values")

        adoption = self.panel.adoption_times
        cohort_column: list[int] = []
        event_column: list[int] = []
        calendar_column: list[int] = []
        size_column: list[int] = []
        support_column: list[bool] = []
        effect_column: list[float] = []

        for cohort in cohorts:
            units = np.flatnonzero(adoption == cohort)
            for event_time in events:
                calendar_time = int(cohort + event_time)
                supported = 0 <= calendar_time < self.panel.n_periods
                if supported and event_time >= 0:
                    effect = float(
                        self.panel.treatment_effect[units, calendar_time].mean()
                    )
                elif supported:
                    effect = 0.0
                else:
                    effect = np.nan
                cohort_column.append(int(cohort))
                event_column.append(int(event_time))
                calendar_column.append(calendar_time)
                size_column.append(int(units.size))
                support_column.append(supported)
                effect_column.append(effect)

        return _frozen_columns(
            cohort=np.asarray(cohort_column, dtype=np.int64),
            event_time=np.asarray(event_column, dtype=np.int64),
            calendar_time=np.asarray(calendar_column, dtype=np.int64),
            cohort_size=np.asarray(size_column, dtype=np.int64),
            supported=np.asarray(support_column, dtype=bool),
            effect=np.asarray(effect_column, dtype=float),
        )

    def event_study(
        self,
        event_times: Any | None = None,
        *,
        weighting: str = "cohort_size",
    ) -> MappingProxyType[str, NDArray[Any]]:
        """Aggregate cohort truth on its supported event-time population.

        ``weighting="cohort_size"`` targets treated units at each event time.
        ``weighting="equal_cohort"`` gives each supported cohort equal weight.
        """

        if weighting not in {"cohort_size", "equal_cohort"}:
            raise ValueError("weighting must be 'cohort_size' or 'equal_cohort'")
        cells = self.cohort_event(event_times)
        events = np.unique(cells["event_time"])
        effects: list[float] = []
        supported_counts: list[int] = []
        target_counts: list[int] = []
        for event_time in events:
            selected = (cells["event_time"] == event_time) & cells["supported"]
            values = cells["effect"][selected]
            sizes = cells["cohort_size"][selected]
            if values.size == 0:
                effect = np.nan
            elif weighting == "cohort_size":
                effect = float(np.average(values, weights=sizes))
            else:
                effect = float(values.mean())
            effects.append(effect)
            supported_counts.append(int(selected.sum()))
            target_counts.append(int(sizes.sum()))
        return _frozen_columns(
            event_time=events.astype(np.int64, copy=False),
            effect=np.asarray(effects, dtype=float),
            supported_cohorts=np.asarray(supported_counts, dtype=np.int64),
            target_unit_count=np.asarray(target_counts, dtype=np.int64),
        )

    def att_by_cohort(self) -> MappingProxyType[str, NDArray[Any]]:
        """Return the realized ATT and treated-cell count for each cohort."""

        cohorts = self._cohorts()
        adoption = self.panel.adoption_times
        effects: list[float] = []
        unit_counts: list[int] = []
        cell_counts: list[int] = []
        for cohort in cohorts:
            units = np.flatnonzero(adoption == cohort)
            treated = self.panel.treatment[units] == 1.0
            values = self.panel.treatment_effect[units][treated]
            effects.append(float(values.mean()))
            unit_counts.append(int(units.size))
            cell_counts.append(int(values.size))
        return _frozen_columns(
            cohort=cohorts,
            att=np.asarray(effects, dtype=float),
            unit_count=np.asarray(unit_counts, dtype=np.int64),
            treated_cell_count=np.asarray(cell_counts, dtype=np.int64),
        )
