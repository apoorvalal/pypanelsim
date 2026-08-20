"""A small registry for named panel-simulator factories."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

from .simulator import PanelSimulator

SimulatorFactory = Callable[..., PanelSimulator]
_VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class DGPRegistry:
    """Map stable design names to factories that create simulators."""

    def __init__(self) -> None:
        self._factories: dict[str, SimulatorFactory] = {}

    def register(
        self,
        name: str,
        factory: SimulatorFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register one factory under a normalized public name."""

        if not _VALID_NAME.fullmatch(name):
            raise ValueError(
                "registry names must start with a letter and contain only "
                "lowercase letters, digits, and underscores"
            )
        if not callable(factory):
            raise TypeError("factory must be callable")
        if name in self._factories and not replace:
            raise ValueError(f"a simulator factory is already registered as {name!r}")
        self._factories[name] = factory

    def create(self, name: str, /, **kwargs: object) -> PanelSimulator:
        """Create a simulator from a registered factory."""

        try:
            factory = self._factories[name]
        except KeyError as error:
            available = ", ".join(self.names()) or "none"
            raise KeyError(
                f"unknown panel design {name!r}; available designs: {available}"
            ) from error
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered names in sorted order."""

        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._factories)
