from collections import UserList
from enum import Enum, IntEnum, unique
from typing import NamedTuple, Sequence

from .coordinate import Coord
from .coordinate import ReplicationId as RId
from .hw_defs import HwParams

__all__ = [
    "RoutingCoord",
    "RoutingCost",
    "RoutingDirection",
    "RoutingLevel",
    "RoutingPath",
    "RoutingStatus",
    "ROUTING_DIRECTIONS_IDX",
    "get_routing_consumption",
    "get_multicast_cores",
    "get_replication_id",
]


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

        for i in reversed(range(len(self))):
            if self[i] > 1:
                return RoutingLevel(i + 1)

        return RoutingLevel.L1


N_ROUTING_LEVEL = 6  # will be deprecated
MAX_ROUTING_PATH_LENGTH = HwParams.N_ROUTING_PATH_LENGTH_MAX


ROUTING_DIRECTIONS_IDX = (
    [
        RoutingDirection.X0Y0,
        RoutingDirection.X0Y1,
        RoutingDirection.X1Y0,
        RoutingDirection.X1Y1,
    ]
    if HwParams.COORD_Y_PRIORITY
    else [
        RoutingDirection.X0Y0,
        RoutingDirection.X1Y0,
        RoutingDirection.X0Y1,
        RoutingDirection.X1Y1,
    ]
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
                f"the direction of routing is not specified completely, got {self}."
            )

    def _L0_property(self) -> None:
        if self.level > RoutingLevel.L0:
            raise AttributeError(
                f"this property is only for L0-level cluster, but self is {self.level}."
            )

    @property
    def level(self) -> RoutingLevel:
        for i in range(MAX_ROUTING_PATH_LENGTH):
            if self[i] is RoutingDirection.ANY:
                return RoutingLevel(MAX_ROUTING_PATH_LENGTH - i)

        return RoutingLevel.L0

    def to_coord(self) -> Coord:
        self._L0_property()
        self._coord_specify_check()

        x = sum(self[i].value[0] << (4 - i) for i in range(MAX_ROUTING_PATH_LENGTH))
        y = sum(self[i].value[1] << (4 - i) for i in range(MAX_ROUTING_PATH_LENGTH))

        return Coord(x, y)

    def __lt__(self, other: "RoutingCoord") -> bool:
        for i in range(MAX_ROUTING_PATH_LENGTH):
            if self[i] is RoutingDirection.ANY:
                if other[i] is RoutingDirection.ANY:
                    continue
                else:
                    return False
            elif other[i] is RoutingDirection.ANY:
                return True
            elif self[i].to_index() == other[i].to_index():
                continue
            elif self[i].to_index() < other[i].to_index():
                return True
            else:
                return False

        return False


class RoutingPath(UserList[RoutingDirection]):
    def __init__(self, *path: RoutingDirection, reverse: bool = False) -> None:
        if reverse:
            _total_rp = [RoutingDirection.ANY] * (
                MAX_ROUTING_PATH_LENGTH - len(path)
            ) + list(reversed(path))
        else:
            _total_rp = path

        super().__init__(_total_rp)

    @property
    def routing_coord(self) -> RoutingCoord:
        return RoutingCoord(*self)

    def to_coord(self) -> Coord:
        return self.routing_coord.to_coord()

    @classmethod
    def n_core2routing_path(cls, n_core: int):
        """Return the routing path by given the #N of cores in reverse order (L4 to L0)."""
        routing_path = []

        # From L0 to L4
        for _ in range(MAX_ROUTING_PATH_LENGTH):
            n_core, re = divmod(n_core, HwParams.N_SUB_ROUTING_NODE)
            routing_path.append(ROUTING_DIRECTIONS_IDX[re])

        return cls(*routing_path, reverse=True)


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
