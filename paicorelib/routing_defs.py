from collections import UserList
from enum import Enum, IntEnum, unique
from typing import NamedTuple, Sequence

from .coordinate import Coord
from .coordinate import ReplicationId as RId
from .hw_defs import HwParams

__all__ = [
    # Classes
    "RoutingCoord",
    "RoutingDirection",
    "RoutingLevel",
    "RoutingPath",
    "RoutingStatus",
    # Functions
    "get_replication_id",
    "get_multicast_cores",
    # Constants
    "ONLINE_CORES_BASE_COORD",
    "ROUTING_DIRECTIONS_IDX",
]

# The base coordinate of online cores
ONLINE_CORES_BASE_COORD = HwParams.N_CORE_OFFLINE


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

    def __str__(self) -> str:
        if self is RoutingDirection.ANY:
            return "ANY"
        else:
            return f"X{self.value[0]}Y{self.value[1]}"


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

    def __str__(self) -> str:
        return f"(L4: {self.L4}, L3: {self.L3}, L2: {self.L2}, L1: {self.L1}, L0: {self.L0})"


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


def get_replication_id(coords: Sequence[Coord]) -> tuple[Coord, RId]:
    """Get the replication ID by given the coordinates.

    Args:
        - coords: sequence of coordinates.

    Returns:
        return a tuple of base coordinate & replication ID.
    """
    if len(coords) < 1:
        raise ValueError("the length of coordinates must be at least 1.")

    base_coord = coords[0]
    rid = RId(0, 0)

    for coord in coords[1:]:
        rid |= base_coord ^ coord

    base_coord_of_mcast = base_coord & (~rid)
    return base_coord_of_mcast, rid


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
