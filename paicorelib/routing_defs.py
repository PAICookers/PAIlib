from enum import Enum, IntEnum, unique
from typing import NamedTuple, Sequence

from .coordinate import Coord
from .coordinate import ReplicationId as RId
from .hw_defs import HwParams

__all__ = [
    "RoutingLevel",
    "RoutingDirection",
    "RoutingStatus",
    "RoutingCost",
    "ROUTING_DIRECTIONS_IDX",
    "RoutingCoord",
    "get_routing_consumption",
    "get_multicast_cores",
    "get_replication_id",
]

N_ROUTING_LEVEL = 6  # L0~L5
MAX_ROUTING_PATH_LENGTH = N_ROUTING_LEVEL - 1


@unique
class RoutingLevel(IntEnum):
    """The level of routing node.

    L0-level nodes are leaves of tree to store data. A L0-cluster is a physical core.
    """

    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


@unique
class RoutingDirection(Enum):
    """Indicate the direction of the four children in the cluster."""

    X0Y0 = (0, 0)
    X0Y1 = (0, 1)
    X1Y0 = (1, 0)
    X1Y1 = (1, 1)
    ANY = (-1, -1)

    def to_index(self) -> int:
        """Convert the direction to index in children list."""
        if self is RoutingDirection.ANY:
            raise TypeError("The direction of routing is not specified.")

        x, y = self.value

        if HwParams.COORD_Y_PRIORITY:
            return (x << 1) + y
        else:
            return (y << 1) + x


@unique
class RoutingStatus(IntEnum):
    """Indicate the status of L0-level cluster. Not used."""

    AVAILABLE = 0
    """Available for item to attach."""

    USED = 1
    """An item is attached to this cluster."""

    OCCUPIED = 2
    """Wasted. It will be an optimization goal."""

    ALL_EMPTY = 3
    """Not used."""


class RoutingCost(NamedTuple):
    n_L0: int
    n_L1: int
    n_L2: int
    n_L3: int
    n_L4: int
    n_L5: int = 1

    def get_routing_level(self) -> RoutingLevel:
        """Return the routing cluster level. If the #N of Lx-level > 1, then we need a  \
            cluster with level Lx+1. And we need the #N of routing sub-level clusters.

        XXX: At present, if #N of L5 > 1, raise exception.
        """
        if self.n_L5 > 1:
            raise ValueError(f"#N of L5-level node out of range, got {self.n_L5}.")

        for i in reversed(range(N_ROUTING_LEVEL)):
            if self[i] > 1:
                return RoutingLevel(i + 1)

        return RoutingLevel.L1


ROUTING_DIRECTIONS_IDX = (
    (
        RoutingDirection.X0Y0,
        RoutingDirection.X0Y1,
        RoutingDirection.X1Y0,
        RoutingDirection.X1Y1,
    )
    if HwParams.COORD_Y_PRIORITY
    else (
        RoutingDirection.X0Y0,
        RoutingDirection.X1Y0,
        RoutingDirection.X0Y1,
        RoutingDirection.X1Y1,
    )
)


class RoutingCoord(NamedTuple):
    """Use router directions to represent the coordinate of a cluster."""

    L4: RoutingDirection = RoutingDirection.ANY
    L3: RoutingDirection = RoutingDirection.ANY
    L2: RoutingDirection = RoutingDirection.ANY
    L1: RoutingDirection = RoutingDirection.ANY
    L0: RoutingDirection = RoutingDirection.ANY

    def _coord_specify_check(self) -> None:
        if RoutingDirection.ANY in self:
            raise ValueError(
                f"The direction of routing is not specified completely, got {self}."
            )

    def _L0_property(self) -> None:
        if self.level > RoutingLevel.L0:
            raise AttributeError(
                f"This property is only for L0-level cluster, but self is {self.level}."
            )

    @property
    def level(self) -> RoutingLevel:
        for i in range(len(self)):
            if self[i] is RoutingDirection.ANY:
                return RoutingLevel(MAX_ROUTING_PATH_LENGTH - i)

        return RoutingLevel.L0

    def to_coord(self) -> Coord:
        self._L0_property()
        self._coord_specify_check()

        x = (
            (self.L4.value[0] << 4)
            + (self.L3.value[0] << 3)
            + (self.L2.value[0] << 2)
            + (self.L1.value[0] << 1)
            + self.L0.value[0]
        )

        y = (
            (self.L4.value[1] << 4)
            + (self.L3.value[1] << 3)
            + (self.L2.value[1] << 2)
            + (self.L1.value[1] << 1)
            + self.L0.value[1]
        )

        return Coord(x, y)


def get_routing_consumption(n_core: int) -> RoutingCost:
    """Get the consumption of clusters at different levels by given the `n_core`."""
    n_sub_node = HwParams.N_SUB_ROUTING_NODE

    # Find the nearest #N(=2^X) to accommodate `n_core` L0-level clusters.
    n_Lx = [0] * N_ROUTING_LEVEL
    n_Lx[0] = 1 << (n_core - 1).bit_length()

    for i in range(N_ROUTING_LEVEL - 1):
        n_Lx[1 + i] = 1 if n_Lx[i] < n_sub_node else (n_Lx[i] // n_sub_node)

    return RoutingCost(*n_Lx)


def get_replication_id(coords: Sequence[Coord]) -> RId:
    """Get the replication ID by given the coordinates.

    Args:
        - coords: sequence of coordinates.
    """
    if len(coords) < 1:
        raise ValueError("the length of coordinates must be at least 1.")

    base_coord = coords[0]
    rid = RId(0, 0)

    for coord in coords[1:]:
        rid |= base_coord ^ coord

    return rid


def get_multicast_cores(base_coord: Coord, rid: RId) -> set[Coord]:
    cores = set()
    corex = set()
    corey = set()
    temp = set()

    corex.add(base_coord.x)
    corey.add(base_coord.y)

    for lx in range(5):
        if (rid.x >> lx) & 1:
            for x in corex:
                temp.add(x ^ (1 << lx))

            corex = corex.union(temp)
            temp.clear()

        if (rid.y >> lx) & 1:
            for y in corey:
                temp.add(y ^ (1 << lx))

            corey = corey.union(temp)
            temp.clear()

    for x in corex:
        for y in corey:
            cores.add(Coord(x, y))

    return cores
