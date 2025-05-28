from dataclasses import dataclass, field
from typing import ClassVar, Union

import numpy as np

from ..coordinate import ChipCoord, Coord
from ..coordinate import ReplicationId as RId
from .frame_defs import FrameFormat as FF
from .frame_defs import FrameHeader as FH
from .frame_defs import FrameType as FT
from .frame_defs import FramePackageType
from .frame_defs import OfflineNeuronRAMFormat as Off_NRAMF
from .types import FRAME_DTYPE, FrameArrayType
from .utils import header2type


@dataclass
class FramePackagePayload:
    """The payload of a frame package."""

    neu_start_addr: int
    """Neuron start address."""
    package_type: FramePackageType
    """Type of the package, config/test-in or test-out."""
    n_package: int
    """#N of packages on the SRAM to read."""

    def __post_init__(self) -> None:
        if (
            self.neu_start_addr > Off_NRAMF.GENERAL_PACKAGE_NEU_START_ADDR_MASK
            or self.neu_start_addr < 0
        ):
            raise ValueError(
                f"neuron base address out of range, {self.neu_start_addr}."
            )

        if self.n_package > Off_NRAMF.GENERAL_PACKAGE_NUM_MASK or self.n_package < 0:
            raise ValueError(
                f"the number of data packages out of range, {self.n_package}."
            )

    @property
    def value(self) -> FRAME_DTYPE:
        return FRAME_DTYPE(
            (
                (self.neu_start_addr & FF.GENERAL_PACKAGE_NEU_START_ADDR_MASK)
                << FF.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET
            )
            | (
                (self.package_type & FF.GENERAL_PACKAGE_TYPE_MASK)
                << FF.GENERAL_PACKAGE_TYPE_OFFSET
            )
            | (
                (self.n_package & FF.GENERAL_PACKAGE_NUM_MASK)
                << FF.GENERAL_PACKAGE_NUM_OFFSET
            )
        )


@dataclass
class _FrameBase:
    header: FH
    chip_coord: ChipCoord
    core_coord: Coord
    rid: RId

    @property
    def frame_type(self) -> FT:
        return header2type(self.header)

    @property
    def chip_addr(self) -> int:
        return self.chip_coord.address

    @property
    def core_addr(self) -> int:
        return self.core_coord.address

    @property
    def rid_addr(self) -> int:
        return self.rid.address

    @property
    def _frame_common(self) -> int:
        header = self.header.value & FF.GENERAL_HEADER_MASK
        chip_addr = self.chip_addr & FF.GENERAL_CHIP_ADDR_MASK
        core_addr = self.core_addr & FF.GENERAL_CORE_ADDR_MASK
        rid_addr = self.rid_addr & FF.GENERAL_CORE_EX_ADDR_MASK

        return (
            (header << FF.GENERAL_HEADER_OFFSET)
            + (chip_addr << FF.GENERAL_CHIP_ADDR_OFFSET)
            + (core_addr << FF.GENERAL_CORE_ADDR_OFFSET)
            + (rid_addr << FF.GENERAL_CORE_EX_ADDR_OFFSET)
        )


@dataclass
class Frame(_FrameBase):
    """Frames which contains information.

    Single frame:
        [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
        4 bits         10 bits             10 bits             10 bits         30 bits

    Frames group = N * single frame:
        [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
        4 bits         10 bits             10 bits             10 bits         30 bits
    """

    repr_split_intv: ClassVar[list[int]] = []
    payload: Union[FRAME_DTYPE, FrameArrayType] = field(default=FRAME_DTYPE(0))

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the single frame."""
        if isinstance(self.payload, (int, np.integer)):
            pl = np.atleast_1d(self.payload).astype(FRAME_DTYPE)
        else:
            pl = self.payload

        value = self._frame_common + (pl & FF.GENERAL_PAYLOAD_MASK)
        value = np.asarray(value, dtype=FRAME_DTYPE)
        value.setflags(write=False)

        return value

    def __len__(self) -> int:
        return 1 if isinstance(self.payload, int) else self.payload.size

    def __str__(self) -> str:
        return (
            f"Frame info:\n"
            f"Head:             {self.header}\n"
            f"Chip coord:       {self.chip_coord}\n"
            f"Core coord:       {self.core_coord}\n"
            f"Replication id:   {self.rid}\n"
            f"Payload:          {self.payload}\n"
        )

    def _make_same_dest(self, payload: Union[FRAME_DTYPE, FrameArrayType]) -> "Frame":
        """Make a new frame with the same destination as the current frame."""
        return type(self)(
            self.header, self.chip_coord, self.core_coord, self.rid, payload
        )


@dataclass
class FramePackage(_FrameBase):
    """Frame package for a length of `N` contents:

    1. [Header(sub type)] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
            4 bits               10 bits            10 bits            10 bits          30 bits
    2. [contents[0]], 64 bits.
    ...
    N+1. [contents[N-1]], 64 bits.

    NOTE: N can be 0. At this case, the frame package is empty.
    """

    PACKAGE_DATA_SHOW_MAX_LINE: ClassVar[int] = 99

    payload: FramePackagePayload
    packages: FrameArrayType = field(
        default_factory=lambda: np.zeros(0, dtype=FRAME_DTYPE)
    )

    @property
    def n_package(self) -> int:
        return self.packages.size

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the group."""
        value = np.zeros((len(self),), dtype=FRAME_DTYPE)

        value[0] = self._frame_common + (
            int(self.payload.value) & FF.GENERAL_PAYLOAD_MASK
        )
        if self.n_package > 0:
            value[1:] = self.packages.copy()
        value.setflags(write=False)

        return value

    def __len__(self) -> int:
        return 1 + self.n_package

    def __str__(self) -> str:
        info = (
            f"Frame package info:\n"
            f"Header:               {self.header}\n"
            f"Chip coord:           {self.chip_coord}\n"
            f"Core coord:           {self.core_coord}\n"
            f"Replication id:       {self.rid}\n"
            f"Payload:\n"
            f"    Neuron start addr:{self.payload.neu_start_addr}\n"
            f"    Package type:     {self.payload.package_type.name}\n"
            f"    N package:        {self.payload.n_package}\n"
        )

        if self.n_package > 0:
            info += f"Package data:\n"

            for i in range(self.n_package):
                info += f"#{i:2d}: {self.packages[i]}\n"
                if i >= self.PACKAGE_DATA_SHOW_MAX_LINE:
                    info += (
                        f"... (showing {self.PACKAGE_DATA_SHOW_MAX_LINE} lines only)\n"
                    )
                    break

        return info

    def _make_same_dest(
        self, payload: FramePackagePayload, packages: FrameArrayType
    ) -> "FramePackage":
        """Make a new frame with the same destination as the current frame package."""
        return type(self)(
            self.header, self.chip_coord, self.core_coord, self.rid, payload, packages
        )
