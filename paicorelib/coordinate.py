import math
import sys
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Sequence, Tuple, TypeVar, Union, final, overload

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from pydantic import Field
from pydantic.dataclasses import dataclass

from .hw_defs import HwParams

__all__ = [
    "Coord",
    "CoordOffset",
    "ReplicationId",
    "CoordLike",
    "RIdLike",
    "to_coord",
    "to_coordoffset",
    "to_rid",
]

CoordTuple: TypeAlias = Tuple[int, int]


def _xy_parser(other: Union[CoordTuple, "CoordOffset"]) -> CoordTuple:
    """Parse the coordinate in tuple format."""
    if not isinstance(other, (tuple, CoordOffset)):
        raise TypeError(f"Unsupported type: {type(other)}")

    if isinstance(other, tuple):
        if len(other) != 2:
            raise ValueError(f"Expect a tuple of 2 elements, but got {len(other)}.")

        return CoordOffset.from_tuple(other).to_tuple()
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

    x: int = Field(
        default=HwParams.CORE_X_MIN,
        ge=HwParams.CORE_X_MIN,
        le=HwParams.CORE_X_MAX,
    )
    y: int = Field(
        default=HwParams.CORE_Y_MIN,
        ge=HwParams.CORE_Y_MIN,
        le=HwParams.CORE_Y_MAX,
    )

    @classmethod
    def from_tuple(cls, pos: CoordTuple) -> "Coord":
        return cls(*pos)

    @classmethod
    def from_addr(cls, addr: int) -> "Coord":
        return cls(addr >> HwParams.N_BIT_CORE_Y, addr & HwParams.CORE_Y_MAX)

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
            raise TypeError(f"Unsupported type: {type(__other)}")

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
    def __sub__(self, __other: "Coord") -> "CoordOffset":
        ...

    @overload
    def __sub__(self, __other: "CoordOffset") -> "Coord":
        ...

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

        if isinstance(__other, CoordOffset):
            diff_x, diff_y = _sum_carry(
                self.x - __other.delta_x, self.y - __other.delta_y
            )
            return Coord(diff_x, diff_y)

        raise TypeError(f"Unsupported type: {type(__other)}")

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

    def __eq__(self, __other: Union[CoordTuple, "Coord"]) -> bool:
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
            raise TypeError(f"Unsupported type: {type(__other)}")

    def __ne__(self, __other: Union[CoordTuple, "Coord"]) -> bool:
        return not self.__eq__(__other)

    # def __lt__(self, __other: "Coord") -> bool:
    #     """Whether the coord is on the left OR below of __other.

    #     Examples:
    #     >>> Coord(4, 5) < Coord(4, 6)
    #     True

    #     >>> Coord(4, 5) < Coord(5, 5)
    #     True

    #     >>> Coord(4, 5) < Coord(5, 3)
    #     True
    #     """
    #     if not isinstance(__other, Coord):
    #         raise TypeError(f"Unsupported type: {type(__other)}")

    #     return self.x < __other.x or self.y < __other.y

    # def __gt__(self, __other: "Coord") -> bool:
    #     """Whether the coord is on the right AND above of __other.

    #     Examples:
    #     >>> Coord(5, 5) > Coord(4, 5)
    #     True

    #     >>> Coord(4, 6) > Coord(4, 5)
    #     True

    #     >>> Coord(5, 4) > Coord(4, 5)
    #     False
    #     """
    #     if not isinstance(__other, Coord):
    #         raise TypeError(f"Unsupported type: {type(__other)}")

    #     # Except the `__eq__`
    #     return (
    #         (self.x > __other.x and self.y > __other.y)
    #         or (self.x == __other.x and self.y > __other.y)
    #         or (self.x > __other.x and self.y == __other.y)
    #     )

    # def __le__(self, __other: "Coord") -> bool:
    #     return self.__lt__(__other) or self.__eq__(__other)

    # def __ge__(self, __other: "Coord") -> bool:
    #     return self.__gt__(__other) or self.__eq__(__other)

    def __xor__(self, __other: "Coord") -> "ReplicationId":
        return ReplicationId(self.x ^ __other.x, self.y ^ __other.y)

    def __hash__(self) -> int:
        return hash(self.address)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Coord({self.x}, {self.y})"

    def to_tuple(self) -> CoordTuple:
        """Convert to tuple"""
        return (self.x, self.y)

    @property
    def address(self) -> int:
        """Convert to address, 10 bits"""
        return (self.x << HwParams.N_BIT_CORE_Y) | self.y


@final
class ReplicationId(Coord):
    def __and__(self, __other: Union[Coord, "ReplicationId"]) -> "ReplicationId":
        return ReplicationId(self.x & __other.x, self.y & __other.y)

    def __or__(self, __other: Union[Coord, "ReplicationId"]) -> "ReplicationId":
        return ReplicationId(self.x | __other.x, self.y | __other.y)

    def __xor__(self, __other: Union[Coord, "ReplicationId"]) -> "ReplicationId":
        return ReplicationId(self.x ^ __other.x, self.y ^ __other.y)

    # def __lshift__(self, __bit: int) -> int:
    #     return self.address << __bit

    # def __rshift__(self, __bit: int) -> int:
    #     return self.address >> __bit


class DistanceType(Enum):
    DISTANCE_ENCLIDEAN = auto()
    DISTANCE_MANHATTAN = auto()
    DISTANCE_CHEBYSHEV = auto()


@dataclass
class CoordOffset:
    """Offset of coordinate"""

    delta_x: int = Field(
        default=HwParams.CORE_X_MIN, ge=-HwParams.CORE_X_MAX, le=HwParams.CORE_X_MAX
    )
    delta_y: int = Field(
        default=HwParams.CORE_Y_MIN, ge=-HwParams.CORE_Y_MAX, le=HwParams.CORE_Y_MAX
    )

    @classmethod
    def from_tuple(cls, pos: CoordTuple) -> "CoordOffset":
        return cls(*pos)

    @overload
    def __add__(self, __other: Coord) -> Coord:
        ...

    @overload
    def __add__(self, __other: "CoordOffset") -> "CoordOffset":
        ...

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
            raise TypeError(f"Unsupported type: {type(__other)}")

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
            raise TypeError(f"Unsupported type: {type(__other)}")

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

    def __eq__(self, __other: Union[CoordTuple, "CoordOffset"]) -> bool:
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
            raise TypeError(f"Unsupported type: {type(__other)}")

    def __ne__(self, __other: "CoordOffset") -> bool:
        return not self.__eq__(__other)

    def to_tuple(self) -> CoordTuple:
        """Convert to tuple"""
        return (self.delta_x, self.delta_y)

    def to_distance(
        self, distance_type: DistanceType = DistanceType.DISTANCE_ENCLIDEAN
    ) -> Union[float, int]:
        """Distance between two coordinates."""
        if distance_type is DistanceType.DISTANCE_ENCLIDEAN:
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
        if (not -HwParams.CORE_X_MAX <= self.delta_x <= HwParams.CORE_X_MAX) or (
            not -HwParams.CORE_Y_MAX <= self.delta_y <= HwParams.CORE_Y_MAX
        ):
            raise ValueError(
                f"Offset of coordinate is out of range: ({self.delta_x}, {self.delta_y})."
            )


_x_crange = HwParams.CORE_X_MAX - HwParams.CORE_X_MIN + 1
_y_crange = HwParams.CORE_Y_MAX - HwParams.CORE_Y_MIN + 1


def _sum_carry(cx: int, cy: int) -> CoordTuple:
    if HwParams.COORD_Y_PRIORITY:
        if cx > HwParams.CORE_X_MAX:
            if cy < HwParams.CORE_Y_MAX:
                cx -= _x_crange
                cy += 1
            else:
                raise ValueError(
                    f"Coordinate of Y out of high limit: {HwParams.CORE_Y_MAX-1}({cy})."
                )
        elif cx < HwParams.CORE_X_MIN:
            if cy > HwParams.CORE_Y_MIN:
                cx += _x_crange
                cy -= 1
            else:
                raise ValueError(
                    f"Coordinate of Y out of low limit: {HwParams.CORE_Y_MIN}."
                )
    else:
        if cy > HwParams.CORE_Y_MAX:
            if cx < HwParams.CORE_X_MAX:
                cx += 1
                cy -= _y_crange
            else:
                raise ValueError(
                    f"Coordinate of X out of high limit: {HwParams.CORE_X_MAX-1}({cx})."
                )
        elif cy < HwParams.CORE_Y_MIN:
            if cx > HwParams.CORE_X_MIN:
                cx -= 1
                cy += _y_crange
            else:
                raise ValueError(
                    f"Coordinate of X out of low limit: {HwParams.CORE_X_MIN}."
                )

    return cx, cy


CoordLike = TypeVar("CoordLike", Coord, int, List[int], CoordTuple)
RIdLike = TypeVar("RIdLike", ReplicationId, int, List[int], CoordTuple)


def to_coord(coordlike: CoordLike) -> Coord:
    if isinstance(coordlike, int):
        return Coord.from_addr(coordlike)

    if isinstance(coordlike, (list, tuple)):
        if len(coordlike) != 2:
            raise TypeError(
                f"Expect a tuple or list of 2 elements, but got {len(coordlike)}."
            )

        return Coord(*coordlike)

    return coordlike


def to_coords(coordlikes: Sequence[CoordLike]) -> List[Coord]:
    return [to_coord(coordlike) for coordlike in coordlikes]


def to_coordoffset(offset: int) -> CoordOffset:
    return CoordOffset(
        offset % (HwParams.CORE_X_MAX + 1), offset // (HwParams.CORE_Y_MAX + 1)
    )


def to_rid(ridlike: RIdLike) -> ReplicationId:
    if isinstance(ridlike, int):
        return ReplicationId.from_addr(ridlike)

    if isinstance(ridlike, (list, tuple)):
        if len(ridlike) != 2:
            raise ValueError(
                f"Expect a tuple or list of 2 elements, but got {len(ridlike)}."
            )

        return ReplicationId(*ridlike)

    return ridlike
