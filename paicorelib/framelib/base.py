import sys
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Union

import numpy as np

from ..coordinate import ChipCoord, Coord
from ..coordinate import ReplicationId as RId
from .frame_defs import FrameFormat as FF
from .frame_defs import FrameHeader as FH
from .frame_defs import FramePackageType
from .frame_defs import FrameType as FT
from .frame_defs import OfflineNeuRAMFormat as Off_NRAMF
from .frame_defs import Online_WF1F_SubType
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
    """#N of packages on the SRAM."""

    def __post_init__(self) -> None:
        if (
            self.neu_start_addr > Off_NRAMF.GENERAL_PACKAGE_NEU_START_ADDR_MASK
            or self.neu_start_addr < 0
        ):
            raise ValueError(
                f"neuron base address out of range [0, {Off_NRAMF.GENERAL_PACKAGE_NEU_START_ADDR_MASK}], "
                f"got {self.neu_start_addr}."
            )

        if self.n_package > Off_NRAMF.GENERAL_PACKAGE_NUM_MASK or self.n_package < 0:
            raise ValueError(
                f"the number of data packages out of range [0, {Off_NRAMF.GENERAL_PACKAGE_NUM_MASK}], "
                f"got {self.n_package}."
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
    """Frame common part:
    [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
     4 bits         10 bits             10 bits             10 bits        30 bits
    """

    header: FH
    chip_coord: ChipCoord
    core_coord: Coord
    rid: RId

    if sys.version_info >= (3, 10):
        subtype: Optional[Online_WF1F_SubType] = field(default=None, kw_only=True)

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

    1. Single frame:
        [Common part] + [payload]
           30 bits       30 bits

    2. Frames group = N * single frame:
        [Common part] + [payload[0]]
        [Common part] + [payload[1]]
        [Common part] + [payload[2]]
                    ...
           30 bits        30 bits
    """

    _payload: Union[np.unsignedinteger, FrameArrayType] = field(default=FRAME_DTYPE(0))

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the frame."""
        value = self._frame_common + (self.payload & FF.GENERAL_PAYLOAD_MASK)
        value = np.asarray(value, dtype=FRAME_DTYPE)
        value.setflags(write=False)
        return value

    @property
    def payload(self) -> FrameArrayType:
        """Get the payload of the frame."""
        return np.atleast_1d(self._payload).view(FRAME_DTYPE)

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

    _payload: FramePackagePayload
    packages: FrameArrayType = field(
        default_factory=lambda: np.zeros(0, dtype=FRAME_DTYPE)
    )

    @property
    def payload(self) -> FramePackagePayload:
        """Get the payload of the frame package."""
        return self._payload

    @property
    def n_package(self) -> int:
        return self.packages.size

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the group."""
        value = np.zeros((len(self),), dtype=FRAME_DTYPE)

        value[0] = self._frame_common + (self.payload.value & FF.GENERAL_PAYLOAD_MASK)
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
            info += "Package data:\n"

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
