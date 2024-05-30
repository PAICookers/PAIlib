from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

from .hw_defs import HwParams
from .reg_types import CoreMode

__all__ = ["NeuronSegment", "AxonCoord", "AxonSegment", "HwCore"]


class NeuronSegment(NamedTuple):
    """Segment of neuron.

    Example:
        index = (0, 100, 1)
        addr_offset = 10
        interval = 2
        addr_ram = (10, 11, ..., 210)

        for addr in addr_ram:
            write in addr with tick_relative and addr_axon at 'addr/interval'.
    """

    index: slice
    """The original index of this segment of neurons."""
    addr_offset: int
    """The offset of the RAM address."""
    interval: int = 1
    """The interval of address when mapping neuron attributes on the RAM."""

    @property
    def n_neuron(self) -> int:
        return self.index.stop - self.index.start

    @property
    def n_addr(self) -> int:
        return self.interval * self.n_neuron

    @property
    def addr_max(self) -> int:
        if (
            _addr_max := self.addr_offset
            + self.interval * (self.index.stop - self.index.start)
        ) > HwParams.ADDR_RAM_MAX:
            raise ValueError(
                f"RAM address out of range {HwParams.ADDR_RAM_MAX} ({_addr_max})."
            )

        return _addr_max

    @property
    def addr_ram(self) -> list[int]:
        """Convert index of neuron into RAM address."""
        return list(range(self.addr_offset, self.addr_max, 1))

    @property
    def addr_slice(self) -> slice:
        """Display the RAM address in slice format."""
        return slice(
            self.addr_offset,
            self.addr_max,
            self.interval,
        )


class AxonCoord(NamedTuple):
    tick_relative: int
    addr_axon: int


class AxonSegment(NamedTuple):
    n_axon: int
    """#N of axons."""
    addr_width: int
    """The range of axon address is [addr_offset, addr_offset + addr_width)."""
    addr_offset: int
    """The offset of the assigned address."""


class HwCore(ABC):
    """Hardware core abstraction."""

    RUNTIME_MODE: ClassVar[CoreMode]

    @property
    @abstractmethod
    def n_core_required(self) -> int:
        """#N of cores required to accommodate neurons inside self."""
        ...

    @classmethod
    @abstractmethod
    def build(cls): ...
