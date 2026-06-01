from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum, unique
from typing import Literal, final, overload

from pydantic import Field
from pydantic.dataclasses import dataclass

from .core_defs import CoreType
from .hw_defs import HwParams, HwParamsV2
from .utils import _mask

__all__ = [
    # Classes
    "Coord",
    "OfflineCoord",
    "OnlineCoord",
    "CoordOffset",
    "ReplicationId",
    # Classes for chip v2.5
    "CoordXY",
    "CoordXYOffset",
    "CoordZXYOffset",
    "CoordXYUnitVec",
    # Aliases
    "ChipCoord",
    "CoordAddr",
    "CoordTuple2d",
    "CoordTuple3d",
    # Types
    "CoordLike",
    "RIdLike",
    "CoordXYLike",
    "CoordXYOffsetLike",
    "CoordZXYOffsetLike",
    # Functions
    "to_coord",
    "to_coords",
    "to_coordoffset",
    "to_rid",
    "to_coordxy",
    "to_coordxys",
    "to_coordxyoffset",
    "to_coordzxyoffset",
    "coordzxy_to_sign_magnitude",
]

CoordAddr = int
CoordTuple2d = tuple[int, int]
CoordTuple3d = tuple[int, int, int]

_cx_max = HwParams.CORE_X_MAX
_cx_min = HwParams.CORE_X_MIN
_cy_max = HwParams.CORE_Y_MAX
_cy_min = HwParams.CORE_Y_MIN
_cx_range = _cx_max - _cx_min + 1
_cy_range = _cy_max - _cy_min + 1


def _xy_parser(other: "CoordTuple2d | CoordOffset") -> CoordTuple2d:
    """Parse the coordinate in tuple format."""
    if not isinstance(other, (tuple, CoordOffset)):
        raise TypeError(f"unsupported type: {type(other).__name__}.")

    if isinstance(other, tuple):
        if len(other) != 2:
            raise ValueError(f"expected a tuple of 2 elements, but got {len(other)}.")

        return CoordOffset(*other).to_tuple()  # check the range of coordoffset
    else:
        return other.to_tuple()


class CoordFormat(ABC):
    """
    Identifier to descripe coordinate of hardware unit.
    """

    @abstractmethod
    def __eq__(self, other) -> bool: ...

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    @abstractmethod
    def __hash__(self): ...


class CoordVecFormat(ABC):
    """
    Identifier to descripe vector of coordinate of hardware unit.
    """

    @abstractmethod
    def __eq__(self, other) -> bool: ...

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.__str__()}"


UNSUPPORTED_OPERAND_TYPE_MSG = "unsupported operand type for {0}: '{1}' and '{2}'."


@dataclass
class Coord(CoordFormat):
    """Coordinate of the cores. Set coordinate (x, y) for every core.

    Left to right, +X, up to down, +Y.
    """

    x: int = Field(default=_cx_min, ge=_cx_min, le=_cx_max, description="X coordinate")
    y: int = Field(default=_cy_min, ge=_cy_min, le=_cy_max, description="Y coordinate")

    @classmethod
    def from_addr(cls, addr: CoordAddr) -> "Coord":
        if HwParams.COORD_Y_PRIORITY:
            return cls(addr >> HwParams.N_BIT_COORD_ADDR, addr & _cy_max)
        else:
            return cls(addr >> HwParams.N_BIT_COORD_ADDR, addr & _cx_max)

    def __add__(self, other: "CoordOffset") -> "Coord":
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c2 = c1 + CoordOffset(1, 1)
        >>> c1
        >>> Coord(2, 2)

        NOTE: `Coord` + `Coord` is meaningless.
        """
        if not isinstance(other, CoordOffset):
            raise TypeError(
                UNSUPPORTED_OPERAND_TYPE_MSG.format(
                    "+", type(self).__name__, type(other).__name__
                )
            )

        sum_x, sum_y = _sum_carry(self.x + other.x, self.y + other.y)

        return Coord(sum_x, sum_y)

    def __iadd__(self, other: "CoordTuple2d | CoordOffset") -> "Coord":
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c1 += CoordOffset(1, 1)
        >>> c1
        >>> Coord(2, 2)
        """
        bx, by = _xy_parser(other)
        self.x, self.y = _sum_carry(self.x + bx, self.y + by)

        return self

    @overload
    def __sub__(self, other: "Coord") -> "CoordOffset": ...

    @overload
    def __sub__(self, other: "CoordOffset") -> "Coord": ...

    def __sub__(self, other: "Coord | CoordOffset") -> "Coord | CoordOffset":
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c2 = Coord(2, 2) - c1
        >>> c2
        >>> CoordOffset(1, 1)
        """
        if not isinstance(other, (Coord, CoordOffset)):
            raise TypeError(
                UNSUPPORTED_OPERAND_TYPE_MSG.format(
                    "-", type(self).__name__, type(other).__name__
                )
            )

        if isinstance(other, Coord):
            return CoordOffset(self.x - other.x, self.y - other.y)
        else:
            diff_x, diff_y = _sum_carry(self.x - other.x, self.y - other.y)
            return Coord(diff_x, diff_y)

    def __isub__(self, other: "CoordTuple2d | CoordOffset") -> "Coord":
        """
        Example:
        >>> c1 = Coord(2, 2)
        >>> c1 -= CoordOffset(1, 1)
        >>> c1
        >>> Coord(1, 1)
        """
        bx, by = _xy_parser(other)
        self.x, self.y = _sum_carry(self.x - bx, self.y - by)

        return self

    """Operations below are used only when comparing with a Cooord."""

    def __eq__(self, other: "CoordTuple2d | Coord") -> bool:
        """
        Example:
        >>> Coord(4, 5) == Coord(4, 6)
        >>> False
        """
        if not isinstance(other, (tuple, Coord)):
            raise TypeError(
                UNSUPPORTED_OPERAND_TYPE_MSG.format(
                    "==", type(self).__name__, type(other).__name__
                )
            )

        if isinstance(other, tuple):
            return self.to_tuple() == other
        else:
            return self.x == other.x and self.y == other.y

    def __and__(self, other: "Coord") -> "Coord":
        return Coord(self.x & other.x, self.y & other.y)

    def __xor__(self, other: "Coord") -> "ReplicationId":
        return ReplicationId(self.x ^ other.x, self.y ^ other.y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    @staticmethod
    def _to_bin(n: int, keep_bits: int) -> str:
        """Convert an integer to a binary string with a fixed number of bits, removing the prefix '0b'."""
        assert 0 <= n < (1 << keep_bits)
        return bin(n)[2:].zfill(keep_bits)

    def to_bin_str(self) -> str:
        """Convert to binary string"""
        return f"({self._to_bin(self.x, HwParams.N_BIT_COORD_ADDR)},{self._to_bin(self.y, HwParams.N_BIT_COORD_ADDR)})"

    def to_tuple(self) -> CoordTuple2d:
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


ChipCoord = Coord
CoordLike = Coord | CoordAddr | CoordTuple2d


@final
class OfflineCoord(Coord):
    """Offline core coordinate"""

    x: int = Field(
        ge=HwParams.CORE_X_OFFLINE_MIN,
        le=HwParams.CORE_X_OFFLINE_MAX,
        description="X coordinate of offline core",
    )
    y: int = Field(
        ge=HwParams.CORE_Y_OFFLINE_MIN,
        le=HwParams.CORE_Y_OFFLINE_MAX,
        description="Y coordinate of offline core",
    )


@final
class OnlineCoord(Coord):
    """Online core coordinate"""

    x: int = Field(
        ge=HwParams.CORE_X_ONLINE_MIN,
        le=HwParams.CORE_X_ONLINE_MAX,
        description="X coordinate of online core",
    )
    y: int = Field(
        ge=HwParams.CORE_Y_ONLINE_MIN,
        le=HwParams.CORE_Y_ONLINE_MAX,
        description="Y coordinate of online core",
    )


@final
class ReplicationId(Coord):
    @classmethod
    def from_addr(cls, addr: CoordAddr) -> "ReplicationId":
        if HwParams.COORD_Y_PRIORITY:
            return cls(addr >> HwParams.N_BIT_COORD_ADDR, addr & _cy_max)
        else:
            return cls(addr >> HwParams.N_BIT_COORD_ADDR, addr & _cx_max)

    def __and__(self, other: "Coord | ReplicationId") -> "ReplicationId":
        return ReplicationId(self.x & other.x, self.y & other.y)

    def __or__(self, other: "Coord | ReplicationId") -> "ReplicationId":
        return ReplicationId(self.x | other.x, self.y | other.y)

    def __invert__(self) -> "ReplicationId":
        return ReplicationId(_cx_max & (~self.x), _cy_max & (~self.y))

    def __xor__(self, other: "Coord | ReplicationId") -> "ReplicationId":
        return ReplicationId(self.x ^ other.x, self.y ^ other.y)

    def __repr__(self) -> str:
        return f"RId{self.__str__()}"

    def is_type_online(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support this method"
        )

    @property
    def core_type(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support this property"
        )


RIdLike = ReplicationId | CoordAddr | CoordTuple2d


@dataclass
class CoordOffset(CoordVecFormat):
    """Offset of coordinate"""

    x: int = Field(default=_cx_min, ge=-_cx_max, le=_cx_max, description="X offset")
    y: int = Field(default=_cy_min, ge=-_cy_max, le=_cy_max, description="Y offset")

    @classmethod
    def from_offset(cls, offset: int) -> "CoordOffset":
        if HwParams.COORD_Y_PRIORITY:
            return cls(offset >> HwParams.N_BIT_COORD_ADDR, offset & _cy_max)
        else:
            return cls(offset & _cy_max, offset >> HwParams.N_BIT_COORD_ADDR)

    @overload
    def __add__(self, other: Coord) -> Coord: ...

    @overload
    def __add__(self, other: "CoordOffset") -> "CoordOffset": ...

    def __add__(self, other: "Coord | CoordOffset") -> "Coord | CoordOffset":
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
        if isinstance(other, CoordOffset):
            # Do not carry.
            return CoordOffset(self.x + other.x, self.y + other.y)
        else:
            sum_x, sum_y = _sum_carry(self.x + other.x, self.y + other.y)
            return Coord(sum_x, sum_y)

    def __iadd__(self, other: "CoordTuple2d | CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c = CoordOffset(1, 1)
        >>> delta_c += CoordOffset(1, 1)
        >>> delta_c
        >>> CoordOffset(2, 2)
        """
        bx, by = _xy_parser(other)
        self.x += bx
        self.y += by

        self._check()
        return self

    def __sub__(self, other: "CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c1 = CoordOffset(1, 1)
        >>> delta_c2 = CoordOffset(2, 2)
        >>> delta_c = delta_c1 - delta_c2
        >>> delta_c
        >>> CoordOffset(-1, -1)
        """
        return CoordOffset(self.x - other.x, self.y - other.y)

    def __isub__(self, other: "CoordTuple2d | CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c = CoordOffset(1, 1)
        >>> delta_c -= CoordOffset(1, 1)
        >>> delta_c
        >>> CoordOffset(0, 0)
        """
        bx, by = _xy_parser(other)
        self.x -= bx
        self.y -= by

        self._check()
        return self

    def __eq__(self, other: "CoordTuple2d | CoordOffset") -> bool:
        if isinstance(other, tuple):
            return self.to_tuple() == other
        else:
            return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    def to_tuple(self) -> CoordTuple2d:
        """Convert to tuple"""
        return (self.x, self.y)

    def _check(self) -> None:
        if (not -_cx_max <= self.x <= _cx_max) or (not -_cy_max <= self.y <= _cy_max):
            raise ValueError(
                f"offset of coordinate is out of range ({self.x}, {self.y})."
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


def _sum_carry(cx: int, cy: int) -> CoordTuple2d:
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


# Implemented for chip v2.5


@dataclass
class CoordXY(CoordFormat):
    """
    Coordinate in XY format. This describes an actual coordinate in 2d plane.

    TODO considering to deal with wraparound maps. Mirror?
    """

    x: int = Field(
        default=0,
        ge=HwParamsV2.CORE_X_MIN,
        le=HwParamsV2.CORE_X_MAX,
        description="X coordinate",
    )
    y: int = Field(
        default=0,
        ge=HwParamsV2.CORE_Y_MIN,
        le=HwParamsV2.CORE_Y_MAX,
        description="Y coordinate",
    )

    def __eq__(self, other: "CoordXY") -> bool:
        return self.x == other.x and self.y == other.y

    def __add__(self, other: "CoordXYOffset") -> "CoordXY":
        if not isinstance(other, CoordXYOffset):
            raise TypeError(
                UNSUPPORTED_OPERAND_TYPE_MSG.format(
                    "+", type(self).__name__, type(other).__name__
                )
            )

        return CoordXY(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: "CoordXYOffset"):
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other: "CoordXY") -> "CoordXYOffset":
        if not isinstance(other, CoordXY):
            raise TypeError(
                UNSUPPORTED_OPERAND_TYPE_MSG.format(
                    "-", type(self).__name__, type(other).__name__
                )
            )

        return CoordXYOffset(self.x - other.x, self.y - other.y)

    def __isub__(self, other: "CoordXYOffset"):
        self.x -= other.x
        self.y -= other.y
        return self

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def copy(self):
        return type(self)(self.x, self.y)

    @overload
    def neighbor(
        self, vec: "CoordXYUnitVec", inplace: Literal[True] = True
    ) -> None: ...

    @overload
    def neighbor(
        self, vec: "CoordXYUnitVec", inplace: Literal[False] = False
    ) -> "CoordXY": ...

    def neighbor(
        self, vec: "CoordXYUnitVec", inplace: bool = False
    ) -> "CoordXY | None":
        if inplace:
            self.x += vec.value.x
            self.y += vec.value.y
        else:
            return self.__add__(vec.value)


@dataclass
class CoordXYOffset(CoordVecFormat):
    """
    Offset in X, Y axis of X-Y coordinate.
    """

    x: int = Field(
        default=0,
        ge=HwParamsV2.CORE_X_MIN,
        le=HwParamsV2.CORE_X_MAX,
        description="offset in X axis",
    )
    y: int = Field(
        default=0,
        ge=HwParamsV2.CORE_Y_MIN,
        le=HwParamsV2.CORE_Y_MAX,
        description="offset in Y axis",
    )

    def __eq__(self, other: "CoordXYOffset") -> bool:
        return self.x == other.x and self.y == other.y

    def __neg__(self) -> "CoordXYOffset":
        return CoordXYOffset(-self.x, -self.y)

    def __add__(self, other: "CoordXYOffset") -> "CoordXYOffset":
        return CoordXYOffset(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: "CoordXYOffset"):
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other: "CoordXYOffset") -> "CoordXYOffset":
        return CoordXYOffset(self.x - other.x, self.y - other.y)

    def __isub__(self, other: "CoordXYOffset"):
        self.x -= other.x
        self.y -= other.y
        return self

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def copy(self):
        return type(self)(self.x, self.y)


@dataclass
class CoordZXYOffset(CoordVecFormat):
    """
    Offset in Z, X, Y axis of X-Y coordinate, where Z is the direction of vector (X,Y)=(1,1).
    This format is used to represent the fields 'CORE_XY', 'CORE_X' & 'CORE_Y' of the AER package.
    """

    z: int = Field(
        default=0,
        ge=HwParamsV2.CORE_Z_MIN,
        le=HwParamsV2.CORE_Z_MAX,
        description="offset in Z axis",
    )
    x: int = Field(
        default=0,
        ge=HwParamsV2.CORE_X_MIN,
        le=HwParamsV2.CORE_X_MAX,
        description="offset in X axis",
    )
    y: int = Field(
        default=0,
        ge=HwParamsV2.CORE_Y_MIN,
        le=HwParamsV2.CORE_Y_MAX,
        description="offset in Y axis",
    )

    def __eq__(self, other: "CoordZXYOffset") -> bool:
        return self.z == other.z and self.x == other.x and self.y == other.y

    def __neg__(self) -> "CoordZXYOffset":
        return CoordZXYOffset(-self.z, -self.x, -self.y)

    def __add__(self, other: "CoordZXYOffset") -> "CoordZXYOffset":
        return CoordZXYOffset(self.z + other.z, self.x + other.x, self.y + other.y)

    def __iadd__(self, other: "CoordZXYOffset"):
        self.z += other.z
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other: "CoordZXYOffset") -> "CoordZXYOffset":
        return CoordZXYOffset(self.z - other.z, self.x - other.x, self.y - other.y)

    def __isub__(self, other: "CoordZXYOffset"):
        self.z -= other.z
        self.x -= other.x
        self.y -= other.y
        return self

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.z, self.x, self.y)

    def to_xy(self) -> CoordXYOffset:
        """Convert to X-Y offset. This means no movement in Z axis, and Z-axis offset is added to X & Y axis offset."""
        return CoordXYOffset(self.z + self.x, self.z + self.y)

    def __str__(self) -> str:
        return f"({self.z},{self.x},{self.y})"

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def copy(self):
        return type(self)(self.z, self.x, self.y)

    def to_sign_magnitude(self) -> tuple[int, int, int]:
        """Convert the ZXY coordinate to sign-magnitude format."""
        return coordzxy_to_sign_magnitude(self)


@unique
class CoordXYUnitVec(Enum):
    """
    The unit vectors of X-Y coordinate.
    NOTE: +Z = (1,1).
    """

    LOCAL = CoordXYOffset(0, 0)
    X_POS = CoordXYOffset(1, 0)
    X_NEG = -X_POS
    Y_POS = CoordXYOffset(0, 1)
    Y_NEG = -Y_POS
    Z_POS = CoordXYOffset(1, 1)
    Z_NEG = -Z_POS


CoordXYLike = CoordXY | CoordTuple2d
CoordXYOffsetLike = CoordXYOffset | CoordTuple2d
CoordZXYOffsetLike = CoordZXYOffset | CoordTuple3d


def to_coordxy(c: CoordXYLike) -> CoordXY:
    if isinstance(c, tuple):
        return CoordXY(*c)
    else:
        return c


def to_coordxys(c: Sequence[CoordXYLike]) -> list[CoordXY]:
    return [to_coordxy(c) for c in c]


def to_coordxyoffset(c: CoordXYOffsetLike) -> CoordXYOffset:
    if isinstance(c, tuple):
        return CoordXYOffset(*c)
    else:
        return c


def to_coordzxyoffset(c: CoordZXYOffsetLike) -> CoordZXYOffset:
    if isinstance(c, tuple):
        return CoordZXYOffset(*c)
    else:
        return c


def coordzxy_to_sign_magnitude(
    coordzxylike: CoordZXYOffsetLike,
) -> tuple[int, int, int]:
    """Convert the coordinate in ZXY format to sign-magnitude format."""
    if isinstance(coordzxylike, CoordZXYOffset):
        z, x, y = coordzxylike.to_tuple()
    else:
        z, x, y = coordzxylike

    def inner(n: int) -> int:
        nbits = HwParamsV2.N_BIT_COORD_ADDR
        signbit = 0 if n >= 0 else 1
        magnitude = abs(n) & _mask(nbits - 1)
        return (signbit << (nbits - 1)) | magnitude

    return (inner(z), inner(x), inner(y))
