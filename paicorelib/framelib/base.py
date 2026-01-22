from abc import ABCMeta
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..coordinate import ChipCoord, Coord, CoordZXYOffset
from ..coordinate import ReplicationId as RId
from ..routing_hexa import AERPacketZXYCopy
from .frame_defs import FF, FFV2, FT, FramePackageType, Online_WF1F_SubType
from .frame_defs import FrameHeader as FH
from .frame_defs import OfflineNeuRAMFormat as Off_NRAMF
from .types import FRAME_DTYPE, FrameArrayType
from .utils import header2type

__all__ = [
    "Frame",
    "FramePackage",
    "FramePackagePayload",
    "FramePackagePayloadV2",
    "FrameV2",
    "FramePackageHeaderV2",
    "get_frame_dest",
    "get_frame_destV2",
]


@dataclass
class FramePackagePayload:
    """The payload of a frame package."""

    ram_start_addr: int
    """Neuron start address."""
    package_type: FramePackageType
    """Type of the package, config/test-in or test-out."""
    n_package: int = 0
    """#N of packages on the SRAM."""

    def __post_init__(self) -> None:
        if (
            self.ram_start_addr > Off_NRAMF.GENERAL_PACKAGE_NEU_START_ADDR_MASK
            or self.ram_start_addr < 0
        ):
            raise ValueError(
                f"neuron base address out of range [0, {Off_NRAMF.GENERAL_PACKAGE_NEU_START_ADDR_MASK}], "
                f"got {self.ram_start_addr}."
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
                (self.ram_start_addr & FF.GENERAL_PACKAGE_NEU_START_ADDR_MASK)
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

    def __str__(self) -> str:
        return f"start_addr={self.ram_start_addr}, type={self.package_type.name}, n={self.n_package}"


def get_frame_dest(
    header: FH, chip_coord: ChipCoord, core_coord: Coord, rid: RId
) -> int:
    h = header.value & FF.GENERAL_HEADER_MASK
    chip_addr = chip_coord.address & FF.GENERAL_CHIP_ADDR_MASK
    core_addr = core_coord.address & FF.GENERAL_CORE_ADDR_MASK
    rid_addr = rid.address & FF.GENERAL_CORE_EX_ADDR_MASK

    return (
        (h << FF.GENERAL_HEADER_OFFSET)
        + (chip_addr << FF.GENERAL_CHIP_ADDR_OFFSET)
        + (core_addr << FF.GENERAL_CORE_ADDR_OFFSET)
        + (rid_addr << FF.GENERAL_CORE_EX_ADDR_OFFSET)
    )


class FrameMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        if cls.__name__ in ("Frame", "FramePackage", "_FrameBase"):
            header_override = kwargs.pop("header", None)
            instance = super().__call__(*args, **kwargs)

            if header_override is not None:
                instance.header = header_override
            return instance
        else:
            return super().__call__(*args, **kwargs)


@dataclass
class _FrameBase(metaclass=FrameMeta):
    """Frame common part:
    [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
     4 bits         10 bits             10 bits             10 bits        30 bits
    """

    header: ClassVar[FH]
    chip_coord: ChipCoord
    core_coord: Coord
    rid: RId
    subtype: Online_WF1F_SubType | None = field(default=None, kw_only=True)

    @property
    def frame_type(self) -> FT:
        return header2type(self.header)

    @property
    def frame_dest(self) -> int:
        return get_frame_dest(self.header, self.chip_coord, self.core_coord, self.rid)


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

    _payload: np.unsignedinteger | FrameArrayType = field(default=FRAME_DTYPE(0))

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the frame."""
        value = self.frame_dest + (self.payload & FF.GENERAL_PAYLOAD_MASK)
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
            f"Frame text:\n"
            f"Head:             {self.header}\n"
            f"Chip coord:       {self.chip_coord}\n"
            f"Core coord:       {self.core_coord}\n"
            f"Replication id:   {self.rid}\n"
            f"Payload:          {self.payload}\n"
        )

    def _make_same_dest(self, payload: FRAME_DTYPE | FrameArrayType):
        """Make a new frame with the same destination as the current frame."""
        return type(self)(self.chip_coord, self.core_coord, self.rid, payload)


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

        value[0] = self.frame_dest + (
            self.payload.value & FRAME_DTYPE(FF.GENERAL_PAYLOAD_MASK)
        )
        if self.n_package > 0:
            value[1:] = self.packages.copy()

        value.setflags(write=False)
        return value

    def __len__(self) -> int:
        return 1 + self.n_package

    def __str__(self) -> str:
        text = "Frame package info:"
        text += f"\nheader:         {self.header}"
        text += f"\nchip coord:      {self.chip_coord}"
        text += f"\ncore coord:      {self.core_coord}"
        text += f"\nreplication id:  {self.rid}"
        text += f"\npayload:         {self.payload}"

        if self.n_package > 0:
            text += "\nPackage data:"

            for i in range(self.n_package):
                text += f"#{i:2d}: {self.packages[i]}\n"
                if i >= self.PACKAGE_DATA_SHOW_MAX_LINE:
                    text += (
                        f"... (showing {self.PACKAGE_DATA_SHOW_MAX_LINE} lines only)\n"
                    )
                    break

        return text

    def _make_same_dest(
        self, payload: FramePackagePayload, packages: FrameArrayType
    ) -> "FramePackage":
        """Make a new frame with the same destination as the current frame package."""
        return type(self)(self.chip_coord, self.core_coord, self.rid, payload, packages)


def get_frame_destV2(
    header: FH, pkt_offset: CoordZXYOffset, pkt_ncopy: AERPacketZXYCopy
) -> int:
    oz, ox, oy = pkt_offset.to_sign_magnitude()
    nz, nx, ny = pkt_ncopy.to_sign_magnitude()
    pkt_addr = (
        (
            ((oz & FFV2.GENERAL_CORE_XY_ADDR_MASK) << FFV2.GENERAL_CORE_XY_ADDR_OFFSET)
            | ((ox & FFV2.GENERAL_CORE_X_ADDR_MASK) << FFV2.GENERAL_CORE_X_ADDR_OFFSET)
            | ((oy & FFV2.GENERAL_CORE_Y_ADDR_MASK) << FFV2.GENERAL_CORE_Y_ADDR_OFFSET)
        )
        | ((nz & FFV2.GENERAL_COPY_XY_ADDR_MASK) << FFV2.GENERAL_COPY_XY_ADDR_OFFSET)
        | ((nx & FFV2.GENERAL_COPY_X_ADDR_MASK) << FFV2.GENERAL_COPY_X_ADDR_OFFSET)
        | ((ny & FFV2.GENERAL_COPY_Y_ADDR_MASK) << FFV2.GENERAL_COPY_Y_ADDR_OFFSET)
    )
    return (
        (header.value & FFV2.GENERAL_HEADER_MASK) << FFV2.GENERAL_HEADER_OFFSET
    ) | pkt_addr


@dataclass
class _FrameBaseV2:
    header: FH
    pkt_offset: CoordZXYOffset
    pkt_ncopy: AERPacketZXYCopy

    def _pkt_addr_in_sign_magnitude(self) -> int:
        oz, ox, oy = self.pkt_offset.to_sign_magnitude()
        nz, nx, ny = self.pkt_ncopy.to_sign_magnitude()
        return (
            (
                (
                    (oz & FFV2.GENERAL_CORE_XY_ADDR_MASK)
                    << FFV2.GENERAL_CORE_XY_ADDR_OFFSET
                )
                | (
                    (ox & FFV2.GENERAL_CORE_X_ADDR_MASK)
                    << FFV2.GENERAL_CORE_X_ADDR_OFFSET
                )
                | (
                    (oy & FFV2.GENERAL_CORE_Y_ADDR_MASK)
                    << FFV2.GENERAL_CORE_Y_ADDR_OFFSET
                )
            )
            | (
                (nz & FFV2.GENERAL_COPY_XY_ADDR_MASK)
                << FFV2.GENERAL_COPY_XY_ADDR_OFFSET
            )
            | ((nx & FFV2.GENERAL_COPY_X_ADDR_MASK) << FFV2.GENERAL_COPY_X_ADDR_OFFSET)
            | ((ny & FFV2.GENERAL_COPY_Y_ADDR_MASK) << FFV2.GENERAL_COPY_Y_ADDR_OFFSET)
        )

    @property
    def frame_dest(self) -> int:
        return get_frame_destV2(self.header, self.pkt_offset, self.pkt_ncopy)

    def __str__(self) -> str:
        text = "Frame info:"
        text += f"\nheader:     {self.header}"
        text += f"\npkg offset: {self.pkt_offset}"
        text += f"\npkg ncopy:  {self.pkt_ncopy}"
        return text


class FramePackagePayloadV2(FramePackagePayload):
    def __post_init__(self) -> None:
        if (
            self.ram_start_addr > FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK
            or self.ram_start_addr < 0
        ):
            raise ValueError(
                f"neuron base address out of range [0, {FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK}], "
                f"got {self.ram_start_addr}."
            )

        if self.n_package > FFV2.GENERAL_PACKAGE_NUM_MASK or self.n_package < 0:
            raise ValueError(
                f"the number of data packages out of range [0, {FFV2.GENERAL_PACKAGE_NUM_MASK}], "
                f"got {self.n_package}."
            )

    @property
    def value(self) -> FRAME_DTYPE:
        return FRAME_DTYPE(
            (
                (self.ram_start_addr & FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK)
                << FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET
            )
            | (
                (self.package_type & FFV2.GENERAL_PACKAGE_TYPE_MASK)
                << FFV2.GENERAL_PACKAGE_TYPE_OFFSET
            )
            | (
                (self.n_package & FFV2.GENERAL_PACKAGE_NUM_MASK)
                << FFV2.GENERAL_PACKAGE_NUM_OFFSET
            )
        )


@dataclass
class FrameV2(_FrameBaseV2):
    payload: int

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the frame."""
        frame = self.frame_dest | (
            (int(self.payload) & FFV2.GENERAL_PAYLOAD_MASK)
            << FFV2.GENERAL_PAYLOAD_OFFSET
        )
        return np.array([frame], dtype=FRAME_DTYPE)


@dataclass
class FramePackageHeaderV2(_FrameBaseV2):
    payload: FramePackagePayloadV2

    @classmethod
    def make_pkg_header(
        cls,
        header: FH,
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        start_addr: int,
        pkg_type: FramePackageType,
        n_package: int = 0,
    ):
        return cls(
            header,
            pkt_offset,
            pkt_ncopy,
            FramePackagePayloadV2(start_addr, pkg_type, n_package),
        )

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the frame package header."""
        frame = self.frame_dest | (
            (int(self.payload.value) & FFV2.GENERAL_PAYLOAD_MASK)
            << FFV2.GENERAL_PAYLOAD_OFFSET
        )
        return np.array([frame], dtype=FRAME_DTYPE)

    def __str__(self) -> str:
        return super().__str__() + f"\npayload:    {self.payload}"
