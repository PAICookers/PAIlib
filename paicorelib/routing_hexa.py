from collections import deque
from collections.abc import Generator
from enum import Flag, auto
from uuid import uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass

from .coordinate import CoordXY, CoordXYUnitVec, CoordZXYOffset
from .hw_defs import HwParamsV2

__all__ = [
    "AERPacketZXYCopy",
    "AERPacket",
    "aer_packet_walk",
    "aer_packet_area",
    "find_coordxy_shortest_path",
]


@dataclass
class AERPacketZXYCopy(CoordZXYOffset):
    z: int = Field(
        default=0,
        ge=HwParamsV2.COPY_Z_MIN,
        le=HwParamsV2.COPY_Z_MAX,
        description="number of copies in Z axis",
    )
    x: int = Field(
        default=0,
        ge=HwParamsV2.COPY_X_MIN,
        le=HwParamsV2.COPY_X_MAX,
        description="number of copies in X axis",
    )
    y: int = Field(
        default=0,
        ge=HwParamsV2.COPY_Y_MIN,
        le=HwParamsV2.COPY_Y_MAX,
        description="number of copies in Y axis",
    )

    def __repr__(self) -> str:
        return f"#copy{self.__str__()}"


class PacketNextMove(Flag):
    TO_LOCAL = auto()
    TO_NEIGHBOR = auto()
    MULTICAST = TO_LOCAL | TO_NEIGHBOR


@dataclass
class AERPacket:
    """
    An AER packet consists of:
        1. the offset in (Z, X, Y) format.
        2. a number of copies in Z, X & Y axis.

    NOTE: The foothold is the coordinate where the packet is on when routing.
    """

    foothold: CoordXY = Field(
        default_factory=CoordXY, description="the coordinate where the packet is on"
    )
    offset: CoordZXYOffset = Field(
        default_factory=CoordZXYOffset,
        description="the offset of the packet in each axis",
    )
    ncopy: AERPacketZXYCopy = Field(
        default_factory=AERPacketZXYCopy,
        description="the number of copies of the packet in each axis",
    )
    id: str = Field(default_factory=lambda: uuid4().hex[:8], description="packet id")

    def __iter__(
        self,
    ) -> "Generator[tuple[PacketNextMove, AERPacket | None], None, None]":
        while True:
            if self.offset.z > 0:
                self.offset.z -= 1
                dirc = CoordXYUnitVec.Z_POS
                move = PacketNextMove.TO_NEIGHBOR
            elif self.offset.z < 0:
                self.offset.z += 1
                dirc = CoordXYUnitVec.Z_NEG
                move = PacketNextMove.TO_NEIGHBOR
            elif self.ncopy.z > 0:
                self.ncopy.z -= 1
                dirc = CoordXYUnitVec.Z_POS
                move = PacketNextMove.MULTICAST
            elif self.ncopy.z < 0:
                self.ncopy.z += 1
                dirc = CoordXYUnitVec.Z_NEG
                move = PacketNextMove.MULTICAST
            elif self.offset.x > 0:
                self.offset.x -= 1
                dirc = CoordXYUnitVec.X_POS
                move = PacketNextMove.TO_NEIGHBOR
            elif self.offset.x < 0:
                self.offset.x += 1
                dirc = CoordXYUnitVec.X_NEG
                move = PacketNextMove.TO_NEIGHBOR
            elif self.ncopy.x > 0:
                self.ncopy.x -= 1
                dirc = CoordXYUnitVec.X_POS
                move = PacketNextMove.MULTICAST
            elif self.ncopy.x < 0:
                self.ncopy.x += 1
                dirc = CoordXYUnitVec.X_NEG
                move = PacketNextMove.MULTICAST
            elif self.offset.y > 0:
                self.offset.y -= 1
                dirc = CoordXYUnitVec.Y_POS
                move = PacketNextMove.TO_NEIGHBOR
            elif self.offset.y < 0:
                self.offset.y += 1
                dirc = CoordXYUnitVec.Y_NEG
                move = PacketNextMove.TO_NEIGHBOR
            elif self.ncopy.y > 0:
                self.ncopy.y -= 1
                dirc = CoordXYUnitVec.Y_POS
                move = PacketNextMove.MULTICAST
            elif self.ncopy.y < 0:
                self.ncopy.y += 1
                dirc = CoordXYUnitVec.Y_NEG
                move = PacketNextMove.MULTICAST
            else:
                move = PacketNextMove.TO_LOCAL

            if move == PacketNextMove.MULTICAST:
                # The core will copy the packet, so change the foothold in place.
                yield move, self.neighbor(dirc)
            elif move == PacketNextMove.TO_NEIGHBOR:
                # The core will modify the foothold of the packet, so change the foothold in place.
                self.foothold.neighbor(dirc, inplace=True)
                yield move, None
            else:  # TO_LOCAL
                # The packet is arrived & stop here
                yield move, None
                break

    def arrived(self) -> bool:
        """Check if the packet is arrived."""
        return self.offset == CoordZXYOffset(
            0, 0, 0
        ) and self.ncopy == AERPacketZXYCopy(0, 0, 0)

    def neighbor(self, direction: CoordXYUnitVec):
        """Move the packet to its neighbor at the given direction."""
        next_foothold = self.foothold.neighbor(direction, inplace=False)
        return type(self)(next_foothold, self.offset.copy(), self.ncopy.copy())

    def copy(self):
        return type(self)(self.foothold.copy(), self.offset.copy(), self.ncopy.copy())


def aer_packet_walk(packet: AERPacket) -> list[CoordXY]:
    """
    Simulate the AER packet routing behavior of chip v2.5.

    Returns:
        A list of coordinates that the packet passed through.
    """
    queue = deque([packet])
    cover_coords = list()  # ordered
    visited = set()

    while queue:
        p = queue.popleft()
        for move, new_pkg in p:
            cur_foot = p.foothold.copy()
            if PacketNextMove.TO_LOCAL in move:
                # TO_LOCAL or MULTICAST
                if cur_foot not in visited:  # walk to a new coordinate
                    visited.add(cur_foot)
                    cover_coords.append(cur_foot)

            if move == PacketNextMove.MULTICAST:
                assert new_pkg is not None
                queue.append(new_pkg)

    return cover_coords


def aer_packet_area(ncopy: AERPacketZXYCopy | tuple[int, int, int]) -> int:
    """
    Get the area of the AER packet. The area is only dependent on the 'ncopy'.
    """
    if isinstance(ncopy, tuple):
        z, x, y = map(abs, ncopy)
    else:
        z, x, y = abs(ncopy.z), abs(ncopy.x), abs(ncopy.y)

    return 1 + z + (1 + z) * x + (1 + z + x) * y


def find_coordxy_shortest_path(
    target: CoordXY, start: CoordXY = CoordXY(0, 0)
) -> tuple[CoordZXYOffset, int]:
    """
    Find the shortest path(in ZXY offset format) from `start` to `target`, and return the length.

    vec = (dx, dy) = (x2 - x1, y2 - y1)
    Assuming that z(1, 1) + x(1, 0) + y(0, 1) = vec & min(|z| + |x| + |y|)
    --> z + x = dx
        z + y = dy
    --> x = dx - z, y = dy - z
    --> min(|z| + |x| + |y|) = min(|z| + |dx - z| + |dy - z|)
    --> o1 = dx, o2 = dy
    """
    vec = target - start
    dx, dy = vec.x, vec.y
    zeros = [0, dx, dy]

    best_z = sorted(zeros)[1]
    best_x = dx - best_z
    best_y = dy - best_z
    min_cost = abs(best_z) + abs(best_x) + abs(best_y)

    return CoordZXYOffset(best_z, best_x, best_y), min_cost
