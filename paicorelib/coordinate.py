import math
import sys
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Sequence, Union, final, overload

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from pydantic import Field
from pydantic.dataclasses import dataclass

from .hw_defs import HwParams
from .reg_defs import CoreType

__all__ = [
    # Classes
    "Coord",
    "OfflineCoord",
    "OnlineCoord",
    "CoordOffset",
    "ReplicationId",
    # Aliases
    "ChipCoord",
    "CoordAddr",
    "CoordTuple",
    # Types
    "CoordLike",
    "RIdLike",
    # Functions
    "to_coord",
    "to_coords",
    "to_coordoffset",
    "to_rid",
]

CoordTuple: TypeAlias = tuple[int, int]
CoordAddr: TypeAlias = int

_cx_max = HwParams.CORE_X_MAX
_cx_min = HwParams.CORE_X_MIN
_cy_max = HwParams.CORE_Y_MAX
_cy_min = HwParams.CORE_Y_MIN
_cx_range = _cx_max - _cx_min + 1
_cy_range = _cy_max - _cy_min + 1


def _xy_parser(other: Union[CoordTuple, "CoordOffset"]) -> CoordTuple:
    """Parse the coordinate in tuple format."""
    if not isinstance(other, (tuple, CoordOffset)):
        raise TypeError(f"unsupported type: {type(other)}.")

    if isinstance(other, tuple):
        if len(other) != 2:
            raise ValueError(f"expected a tuple of 2 elements, but got {len(other)}.")

        return CoordOffset(*other).to_tuple()  # check the range of coordoffset
    else:
        return other.to_tuple()


class _CoordIdentifier(ABC):
    """Identifier to descripe coordinate of hardware unit. \
        The subclass of identifier must implement `__eq__` & `__ne__`."""

    @abstractmethod
    def __eq__(self, __other) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __ne__(self, __other) -> bool:
        raise NotImplementedError


@dataclass
class Coord(_CoordIdentifier):
    """Coordinate of the cores. Set coordinate (x, y) for every core.

    Left to right, +X, up to down, +Y.
    """

    x: int = Field(default=_cx_min, ge=_cx_min, le=_cx_max)
    y: int = Field(default=_cy_min, ge=_cy_min, le=_cy_max)

    @classmethod
    def from_addr(cls, addr: CoordAddr) -> "Coord":
        if HwParams.COORD_Y_PRIORITY:
            return cls(addr >> HwParams.N_BIT_COORD_ADDR, addr & _cy_max)
        else:
            return cls(addr >> HwParams.N_BIT_COORD_ADDR, addr & _cx_max)

    def __add__(self, __other: "CoordOffset") -> "Coord":
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c2 = c1 + CoordOffset(1, 1)
        >>> c1
        >>> Coord(2, 2)

        NOTE: `Coord` + `Coord` is meaningless.
        """
        if not isinstance(__other, CoordOffset):
            raise TypeError(f"unsupported type: {type(__other)}.")

        sum_x, sum_y = _sum_carry(self.x + __other.delta_x, self.y + __other.delta_y)

        return Coord(sum_x, sum_y)

    def __iadd__(self, __other: Union[CoordTuple, "CoordOffset"]) -> "Coord":
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c1 += CoordOffset(1, 1)
        >>> c1
        >>> Coord(2, 2)
        """
        bx, by = _xy_parser(__other)
        self.x, self.y = _sum_carry(self.x + bx, self.y + by)

        return self

    @overload
    def __sub__(self, __other: "Coord") -> "CoordOffset": ...

    @overload
    def __sub__(self, __other: "CoordOffset") -> "Coord": ...

    def __sub__(
        self, __other: Union["Coord", "CoordOffset"]
    ) -> Union["Coord", "CoordOffset"]:
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c2 = Coord(2, 2) - c1
        >>> c2
        >>> CoordOffset(1, 1)
        """
        if isinstance(__other, Coord):
            return CoordOffset(self.x - __other.x, self.y - __other.y)

        elif isinstance(__other, CoordOffset):
            diff_x, diff_y = _sum_carry(
                self.x - __other.delta_x, self.y - __other.delta_y
            )
            return Coord(diff_x, diff_y)
        else:
            raise TypeError(f"unsupported type: {type(__other)}.")

    def __isub__(self, __other: Union[CoordTuple, "CoordOffset"]) -> "Coord":
        """
        Example:
        >>> c1 = Coord(2, 2)
        >>> c1 -= CoordOffset(1, 1)
        >>> c1
        >>> Coord(1, 1)
        """
        bx, by = _xy_parser(__other)
        self.x, self.y = _sum_carry(self.x - bx, self.y - by)

        return self

    """Operations below are used only when comparing with a Cooord."""

    def __eq__(self, __other) -> bool:
        """
        Example:
        >>> Coord(4, 5) == Coord(4, 6)
        >>> False
        """
        if isinstance(__other, tuple):
            return self.to_tuple() == __other
        elif isinstance(__other, Coord):
            return self.x == __other.x and self.y == __other.y
        else:
            raise TypeError(f"unsupported type: {type(__other)}.")

    def __ne__(self, __other) -> bool:
        return not self.__eq__(__other)

    def __and__(self, __other: "Coord") -> "Coord":
        return Coord(self.x & __other.x, self.y & __other.y)

    def __xor__(self, __other: "Coord") -> "ReplicationId":
        return ReplicationId(self.x ^ __other.x, self.y ^ __other.y)

    def __hash__(self) -> int:
        return hash(self.address)

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    def __repr__(self) -> str:
        return f"Coord{self.__str__()}"

    @staticmethod
    def _to_bin(n: int, keep_bits: int) -> str:
        """Convert an integer to a binary string with a fixed number of bits, removing the prefix '0b'."""
        assert 0 <= n < (1 << keep_bits)
        return bin(n)[2:].zfill(keep_bits)

    def to_bin_str(self) -> str:
        """Convert to binary string"""
        return f"({self._to_bin(self.x, HwParams.N_BIT_COORD_ADDR)},{self._to_bin(self.y, HwParams.N_BIT_COORD_ADDR)})"

    def to_tuple(self) -> CoordTuple:
        """Convert to tuple"""
        return (self.x, self.y)

    def is_type_online(self) -> bool:
        """Check if the core is of online type.

        NOTE: The online core is located in the square area at the bottom right corner(+X,+Y) of the chip.
        """
        return (
            HwParams.CORE_X_ONLINE_MIN <= self.x <= HwParams.CORE_X_ONLINE_MAX
            and HwParams.CORE_Y_ONLINE_MIN <= self.y <= HwParams.CORE_Y_ONLINE_MAX
        )

    @property
    def address(self) -> CoordAddr:
        """Convert to address, 10 bits"""
        if HwParams.COORD_Y_PRIORITY:
            return (self.x << HwParams.N_BIT_COORD_ADDR) | self.y
        else:
            return (self.y << HwParams.N_BIT_COORD_ADDR) | self.x

    @property
    def core_type(self) -> CoreType:
        return CoreType.ONLINE if self.is_type_online() else CoreType.OFFLINE


_offcx_min = HwParams.CORE_X_OFFLINE_MIN
_offcx_max = HwParams.CORE_X_OFFLINE_MAX
_offcy_min = HwParams.CORE_Y_OFFLINE_MIN
_offcy_max = HwParams.CORE_Y_OFFLINE_MAX


@final
class OfflineCoord(Coord):
    """Offline core coordinate"""

    x: int = Field(default=_offcx_min, ge=_offcx_min, le=_offcx_max)
    y: int = Field(default=_offcy_min, ge=_offcy_min, le=_offcy_max)


_oncx_min = HwParams.CORE_X_ONLINE_MIN
_oncx_max = HwParams.CORE_X_ONLINE_MAX
_oncy_min = HwParams.CORE_Y_ONLINE_MIN
_oncy_max = HwParams.CORE_Y_ONLINE_MAX


@final
class OnlineCoord(Coord):
    """Online core coordinate"""

    x: int = Field(default=_oncx_min, ge=_oncx_min, le=_oncx_max)
    y: int = Field(default=_oncy_min, ge=_oncy_min, le=_oncy_max)


@final
class ReplicationId(Coord):
    @classmethod
    def from_addr(cls, addr: CoordAddr) -> "ReplicationId":
        if HwParams.COORD_Y_PRIORITY:
            return cls(addr >> HwParams.N_BIT_COORD_ADDR, addr & _cy_max)
        else:
            return cls(addr >> HwParams.N_BIT_COORD_ADDR, addr & _cx_max)

    def __and__(self, __other: Union[Coord, "ReplicationId"]) -> "ReplicationId":
        return ReplicationId(self.x & __other.x, self.y & __other.y)

    def __or__(self, __other: Union[Coord, "ReplicationId"]) -> "ReplicationId":
        return ReplicationId(self.x | __other.x, self.y | __other.y)

    def __invert__(self) -> "ReplicationId":
        return ReplicationId(_cx_max & (~self.x), _cy_max & (~self.y))

    def __xor__(self, __other: Union[Coord, "ReplicationId"]) -> "ReplicationId":
        return ReplicationId(self.x ^ __other.x, self.y ^ __other.y)

    def __str__(self) -> str:
        return f"({self.x},{self.y})*"

    def __repr__(self) -> str:
        return f"RId{self.__str__()}"

    def is_type_online(self):
        raise NotImplementedError

    @property
    def core_type(self):
        raise NotImplementedError


class DistanceType(Enum):
    DISTANCE_EUCLIDEAN = auto()
    DISTANCE_MANHATTAN = auto()
    DISTANCE_CHEBYSHEV = auto()


@dataclass
class CoordOffset:
    """Offset of coordinate"""

    delta_x: int = Field(default=_cx_min, ge=-_cx_max, le=_cx_max)
    delta_y: int = Field(default=_cy_min, ge=-_cy_max, le=_cy_max)

    @classmethod
    def from_offset(cls, offset: int) -> "CoordOffset":
        if HwParams.COORD_Y_PRIORITY:
            return cls(offset >> HwParams.N_BIT_COORD_ADDR, offset & _cy_max)
        else:
            return cls(offset & _cy_max, offset >> HwParams.N_BIT_COORD_ADDR)

    @overload
    def __add__(self, __other: Coord) -> Coord: ...

    @overload
    def __add__(self, __other: "CoordOffset") -> "CoordOffset": ...

    def __add__(
        self, __other: Union["CoordOffset", Coord]
    ) -> Union["CoordOffset", Coord]:
        """
        Examples:
        >>> delta_c1 = CoordOffset(1, 1)
        >>> delta_c2 = delta_c1 + CoordOffset(1, 1)
        >>> delta_c2
        >>> CoordOffset(2, 2)

        Coord = CoordOffset + Coord
        >>> delta_c = CoordOffset(1, 1)
        >>> c1 = Coord(2, 3)
        >>> c2 = delta_c + c1
        >>> c2
        >>> Coord(3, 4)
        """
        if isinstance(__other, CoordOffset):
            # Do not carry.
            return CoordOffset(
                self.delta_x + __other.delta_x, self.delta_y + __other.delta_y
            )
        elif isinstance(__other, Coord):
            sum_x, sum_y = _sum_carry(
                self.delta_x + __other.x, self.delta_y + __other.y
            )
            return Coord(sum_x, sum_y)
        else:
            raise TypeError(f"unsupported type: {type(__other)}.")

    def __iadd__(self, __other: Union[CoordTuple, "CoordOffset"]) -> "CoordOffset":
        """
        Example:
        >>> delta_c = CoordOffset(1, 1)
        >>> delta_c += CoordOffset(1, 1)
        >>> delta_c
        >>> CoordOffset(2, 2)
        """
        bx, by = _xy_parser(__other)
        self.delta_x += bx
        self.delta_y += by

        self._check()
        return self

    def __sub__(self, __other: "CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c1 = CoordOffset(1, 1)
        >>> delta_c2 = CoordOffset(2, 2)
        >>> delta_c = delta_c1 - delta_c2
        >>> delta_c
        >>> CoordOffset(-1, -1)
        """
        if not isinstance(__other, CoordOffset):
            raise TypeError(f"unsupported type: {type(__other)}.")

        return CoordOffset(
            self.delta_x - __other.delta_x, self.delta_y - __other.delta_y
        )

    def __isub__(self, __other: Union[CoordTuple, "CoordOffset"]) -> "CoordOffset":
        """
        Example:
        >>> delta_c = CoordOffset(1, 1)
        >>> delta_c -= CoordOffset(1, 1)
        >>> delta_c
        >>> CoordOffset(0, 0)
        """
        bx, by = _xy_parser(__other)
        self.delta_x -= bx
        self.delta_y -= by

        self._check()
        return self

    def __eq__(self, __other) -> bool:
        """
        Example:
        >>> CoordOffset(4, 5) == CoordOffset(4, 6)
        >>> False
        """
        if isinstance(__other, tuple):
            return self.to_tuple() == __other
        elif isinstance(__other, CoordOffset):
            return self.delta_x == __other.delta_x and self.delta_y == __other.delta_y
        else:
            raise TypeError(f"unsupported type: {type(__other)}.")

    def __ne__(self, __other) -> bool:
        return not self.__eq__(__other)

    def __str__(self) -> str:
        return f"({self.delta_x},{self.delta_y})"

    def __repr__(self) -> str:
        return f"CoordOffset{self.__str__()}"

    def to_tuple(self) -> CoordTuple:
        """Convert to tuple"""
        return (self.delta_x, self.delta_y)

    def to_distance(
        self, distance_type: DistanceType = DistanceType.DISTANCE_EUCLIDEAN
    ) -> Union[float, int]:
        """Distance between two coordinates."""
        if distance_type is DistanceType.DISTANCE_EUCLIDEAN:
            return self._euclidean_distance()
        elif distance_type is DistanceType.DISTANCE_MANHATTAN:
            return self._manhattan_distance()
        else:
            return self._chebyshev_distance()

    def _euclidean_distance(self) -> float:
        """Euclidean distance"""
        return math.sqrt(self.delta_x**2 + self.delta_y**2)

    def _manhattan_distance(self) -> int:
        """Manhattan distance"""
        return abs(self.delta_x) + abs(self.delta_y)

    def _chebyshev_distance(self) -> int:
        """Chebyshev distance"""
        return max(abs(self.delta_x), abs(self.delta_y))

    def _check(self) -> None:
        if (not -_cx_max <= self.delta_x <= _cx_max) or (
            not -_cy_max <= self.delta_y <= _cy_max
        ):
            raise ValueError(
                f"offset of coordinate is out of range ({self.delta_x}, {self.delta_y})."
            )


_EXCEED_UPPER_TEXT = "coordinate of {0} exceeds the upper limit {1} ({2})."
_EXCEED_LOWER_TEXT = "coordinate of {0} exceeds the lower limit {1} ({2})."
_EXCEED_UPPER_CARRY_TEXT = (
    "coordinate of {0} exceeds the upper limit {1} ({2}) when {3} carries."
)
_EXCEED_LOWER_CARRY_TEXT = (
    "coordinate of {0} exceeds the lower limit {1} ({2}) when {3} carries."
)
_EXCEED_UPPER_BORROW_TEXT = (
    "coordinate of {0} exceeds the upper limit {1} ({2}) when {3} borrows."
)
_EXCEED_LOWER_BORROW_TEXT = (
    "coordinate of {0} exceeds the lower limit {1} ({2}) when {3} borrows."
)


def _sum_carry(cx: int, cy: int) -> CoordTuple:
    if HwParams.COORD_Y_PRIORITY:
        if cy > _cy_max:
            if cx + 1 > _cx_max:
                raise ValueError(
                    _EXCEED_UPPER_CARRY_TEXT.format("x", _cx_max, cx + 1, "y")
                )
            elif cx + 1 < _cx_min:
                raise ValueError(
                    _EXCEED_LOWER_CARRY_TEXT.format("x", _cx_min, cx + 1, "y")
                )
            else:
                cx += 1
                cy -= _cy_range
        elif _cy_min <= cy <= _cy_max:
            if cx > _cx_max:
                raise ValueError(_EXCEED_UPPER_TEXT.format("x", _cx_max, cx))
            elif cx < _cx_min:
                raise ValueError(_EXCEED_LOWER_TEXT.format("x", _cx_min, cx))
        else:  # cy < _cy_min
            if cx - 1 > _cx_max:
                raise ValueError(
                    _EXCEED_UPPER_BORROW_TEXT.format("x", _cx_max, cx - 1, "y")
                )
            elif cx - 1 < _cx_min:
                raise ValueError(
                    _EXCEED_LOWER_BORROW_TEXT.format("x", _cx_min, cx - 1, "y")
                )
            else:
                cx -= 1
                cy += _cy_range
    else:
        if cx > _cx_max:
            if cy + 1 > _cy_max:
                raise ValueError(
                    _EXCEED_UPPER_CARRY_TEXT.format("y", _cy_max, cy + 1, "x")
                )
            elif cy + 1 < _cy_min:
                raise ValueError(
                    _EXCEED_LOWER_CARRY_TEXT.format("y", _cy_min, cy + 1, "x")
                )
            else:
                cy += 1
                cx -= _cy_range
        elif _cx_min <= cx <= _cx_max:
            if cy > _cy_max:
                raise ValueError(_EXCEED_UPPER_TEXT.format("y", _cy_max, cy))
            elif cy < _cy_min:
                raise ValueError(_EXCEED_LOWER_TEXT.format("y", _cy_min, cy))
        else:  # cx < _cx_min
            if cy - 1 > _cy_max:
                raise ValueError(
                    _EXCEED_UPPER_BORROW_TEXT.format("y", _cy_max, cy - 1, "x")
                )
            elif cy - 1 < _cy_min:
                raise ValueError(
                    _EXCEED_LOWER_BORROW_TEXT.format("y", _cy_min, cy - 1, "x")
                )
            else:
                cy -= 1
                cx += _cx_range

    return cx, cy


ChipCoord: TypeAlias = Coord
CoordLike: TypeAlias = Union[Coord, CoordAddr, CoordTuple]
RIdLike: TypeAlias = Union[ReplicationId, CoordAddr, CoordTuple]


def to_coord(coordlike: CoordLike) -> Coord:
    if isinstance(coordlike, CoordAddr):
        return Coord.from_addr(coordlike)
    elif isinstance(coordlike, tuple):
        return Coord(*coordlike)
    else:
        return coordlike


def to_coords(coordlikes: Sequence[CoordLike]) -> list[Coord]:
    return [to_coord(coordlike) for coordlike in coordlikes]


def to_coordoffset(offset: int) -> CoordOffset:
    return CoordOffset.from_offset(offset)


def to_rid(ridlike: RIdLike) -> ReplicationId:
    if isinstance(ridlike, CoordAddr):
        return ReplicationId.from_addr(ridlike)
    elif isinstance(ridlike, tuple):
        return ReplicationId(*ridlike)
    else:
        return ridlike
