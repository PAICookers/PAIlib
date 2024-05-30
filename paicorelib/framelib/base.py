import copy
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from paicorelib import Coord
from paicorelib import ReplicationId as RId

from .frame_defs import FrameFormat as FF
from .frame_defs import FrameHeader as FH
from .frame_defs import FrameType as FT
from .types import FRAME_DTYPE, FrameArrayType
from .utils import header2type


@dataclass
class Frame:
    """frames which contains information.

    single frame:
        [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
        4 bits         10 bits             10 bits             10 bits         30 bits

    frames group:
        [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
        4 bits         10 bits             10 bits             10 bits         30 bits
    """

    header: FH
    chip_coord: Coord
    core_coord: Coord
    rid: RId
    payload: Union[FRAME_DTYPE, FrameArrayType] = field(
        default_factory=lambda: np.empty(0, dtype=FRAME_DTYPE)
    )

    @classmethod
    def _decode(
        cls,
        header: FH,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        payload: Union[FRAME_DTYPE, FrameArrayType],
    ):
        return cls(header, chip_coord, core_coord, rid, payload)

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
    def value(self) -> FrameArrayType:
        """Get the full frames of the single frame."""
        if isinstance(self.payload, np.integer):
            pl = np.asarray([self.payload], dtype=FRAME_DTYPE)
        else:
            pl = self.payload

        value = self._frame_common + (pl & FF.GENERAL_PAYLOAD_MASK)
        value = np.asarray(value, dtype=FRAME_DTYPE)
        value.setflags(write=False)

        return value

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

    def __len__(self) -> int:
        return 1 if isinstance(self.payload, int) else self.payload.size

    def __str__(self) -> str:
        return (
            f"Frame info:\n"
            f"Head:                 {self.header}\n"
            f"Chip address:         {self.chip_coord}\n"
            f"Core address:         {self.core_coord}\n"
            f"Replication address:  {self.rid}\n"
            f"Payload:              {self.payload}\n"
        )

    def __deepcopy__(self) -> "Frame":
        """Deep copy the frame and return a new `Frame`."""
        return Frame(
            self.header,
            self.chip_coord,
            self.core_coord,
            self.rid,
            copy.deepcopy(self.payload),
        )


@dataclass
class FramePackage(Frame):
    """Frame package for a length of `N` contents:

    1. [Header(sub type)] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
            4 bits               10 bits            10 bits            10 bits          30 bits
    2. [contents[0]], 64 bits.
    N+1. [contents[N-1]], 64 bits.

    """

    payload: FRAME_DTYPE = FRAME_DTYPE(0)
    packages: FrameArrayType = field(
        default_factory=lambda: np.empty(0, dtype=FRAME_DTYPE)
    )

    @classmethod
    def _decode(
        cls,
        header: FH,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        payload: FRAME_DTYPE,
        packages: FrameArrayType,
    ):
        assert payload.ndim == 1
        return cls(header, chip_coord, core_coord, rid, payload, packages)

    @property
    def n_package(self) -> int:
        return self.packages.size

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the group."""
        value = np.zeros((len(self),), dtype=FRAME_DTYPE)

        value[0] = self._frame_common + (int(self.payload) & FF.GENERAL_PAYLOAD_MASK)
        value[1:] = self.packages.copy()
        value.setflags(write=False)

        return value

    def __len__(self) -> int:
        return 1 + self.n_package

    def __str__(self) -> str:
        _present = (
            f"FramePackage info:\n"
            f"Header:               {self.header}\n"
            f"Chip address:         {self.chip_coord}\n"
            f"Core address:         {self.core_coord}\n"
            f"Replication address:  {self.rid}\n"
            f"Payload:              {self.payload}\n"
            f"Data:\n"
        )

        for i in range(self.n_package):
            _present += f"#{i}: {self.packages[i]}\n"

        return _present

    def __deepcopy__(self) -> "FramePackage":
        """Deep copy the frame package and return a new `FramePackage`."""
        return FramePackage(
            self.header,
            self.chip_coord,
            self.core_coord,
            self.rid,
            self.payload.copy(),
            copy.deepcopy(self.packages),
        )
