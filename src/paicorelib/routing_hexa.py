from collections.abc import Generator
from enum import Flag, auto
from uuid import uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass

from .coordinate import (
    CoordTuple2d,
    CoordTuple3d,
    CoordXY,
    CoordXYUnitVec,
    CoordZXYOffset,
    CoordZXYOffsetLike,
)
from .hw_defs import HwParamsV2

__all__ = [
    "AERPacketZXYCopy",
    "AERPacket",
    "aer_packet_copy_offsets",
    "route_coord_path",
    "aer_packet_walk",
    "aer_packet_area",
    "find_coordxy_shortest_path",
]


@dataclass
class AERPacketZXYCopy(CoordZXYOffset):
    """Mutable AER copy counts. Use ``to_tuple()`` for stable dict/set keys."""

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
                yield move, self.neighbor(dirc)
            elif move == PacketNextMove.TO_NEIGHBOR:
                self.foothold.neighbor(dirc, inplace=True)
                yield move, None
            else:
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


def aer_packet_copy_offsets(
    ncopy: AERPacketZXYCopy | CoordTuple3d,
) -> tuple[CoordXY, ...]:
    """Return local coordinates covered by one Z/X/Y copy tuple."""
    return tuple(_aer_packet_walk_xy(0, 0, 0, 0, 0, *_zxy_tuple(ncopy)))


def route_coord_path(
    start_coord: CoordXY, offset: CoordZXYOffsetLike
) -> tuple[CoordXY, ...]:
    """Return coordinates visited by one Z, then X, then Y route."""
    x, y = start_coord.x, start_coord.y
    z_offset, x_offset, y_offset = _zxy_tuple(offset)
    path = [start_coord]

    def walk(step_count: int, dx: int, dy: int) -> None:
        nonlocal x, y
        for _ in range(step_count):
            x += dx
            y += dy
            path.append(CoordXY(x, y))

    if z_offset:
        step = 1 if z_offset > 0 else -1
        walk(abs(z_offset), step, step)
    if x_offset:
        walk(abs(x_offset), 1 if x_offset > 0 else -1, 0)
    if y_offset:
        walk(abs(y_offset), 0, 1 if y_offset > 0 else -1)

    return tuple(path)


def aer_packet_walk(packet: AERPacket) -> list[CoordXY]:
    """
    Simulate the AER packet routing behavior of chip v2.5.

    Returns:
        A list of coordinates that the packet passed through.
    """
    return _aer_packet_walk_xy(
        packet.foothold.x,
        packet.foothold.y,
        packet.offset.z,
        packet.offset.x,
        packet.offset.y,
        packet.ncopy.z,
        packet.ncopy.x,
        packet.ncopy.y,
    )


def _zxy_tuple(zxy: CoordZXYOffsetLike) -> CoordTuple3d:
    if isinstance(zxy, tuple):
        return zxy
    return zxy.to_tuple()


def _append_coord(
    coords: list[CoordXY], visited: set[CoordTuple2d], x: int, y: int
) -> None:
    key = (x, y)
    if key in visited:
        return
    visited.add(key)
    coords.append(CoordXY(x, y))


def _aer_packet_walk_xy(
    start_x: int,
    start_y: int,
    offset_z: int,
    offset_x: int,
    offset_y: int,
    copy_z: int,
    copy_x: int,
    copy_y: int,
) -> list[CoordXY]:
    queue = [(start_x, start_y, offset_z, offset_x, offset_y, copy_z, copy_x, copy_y)]
    coords: list[CoordXY] = []
    visited: set[CoordTuple2d] = set()
    queue_index = 0

    while queue_index < len(queue):
        x, y, z_offset, x_offset, y_offset, z_copy, x_copy, y_copy = queue[queue_index]
        queue_index += 1

        while True:
            if z_offset > 0:
                z_offset -= 1
                x += 1
                y += 1
            elif z_offset < 0:
                z_offset += 1
                x -= 1
                y -= 1
            elif z_copy > 0:
                z_copy -= 1
                _append_coord(coords, visited, x, y)
                queue.append(
                    (
                        x + 1,
                        y + 1,
                        z_offset,
                        x_offset,
                        y_offset,
                        z_copy,
                        x_copy,
                        y_copy,
                    )
                )
            elif z_copy < 0:
                z_copy += 1
                _append_coord(coords, visited, x, y)
                queue.append(
                    (
                        x - 1,
                        y - 1,
                        z_offset,
                        x_offset,
                        y_offset,
                        z_copy,
                        x_copy,
                        y_copy,
                    )
                )
            elif x_offset > 0:
                x_offset -= 1
                x += 1
            elif x_offset < 0:
                x_offset += 1
                x -= 1
            elif x_copy > 0:
                x_copy -= 1
                _append_coord(coords, visited, x, y)
                queue.append(
                    (
                        x + 1,
                        y,
                        z_offset,
                        x_offset,
                        y_offset,
                        z_copy,
                        x_copy,
                        y_copy,
                    )
                )
            elif x_copy < 0:
                x_copy += 1
                _append_coord(coords, visited, x, y)
                queue.append(
                    (
                        x - 1,
                        y,
                        z_offset,
                        x_offset,
                        y_offset,
                        z_copy,
                        x_copy,
                        y_copy,
                    )
                )
            elif y_offset > 0:
                y_offset -= 1
                y += 1
            elif y_offset < 0:
                y_offset += 1
                y -= 1
            elif y_copy > 0:
                y_copy -= 1
                _append_coord(coords, visited, x, y)
                queue.append(
                    (
                        x,
                        y + 1,
                        z_offset,
                        x_offset,
                        y_offset,
                        z_copy,
                        x_copy,
                        y_copy,
                    )
                )
            elif y_copy < 0:
                y_copy += 1
                _append_coord(coords, visited, x, y)
                queue.append(
                    (
                        x,
                        y - 1,
                        z_offset,
                        x_offset,
                        y_offset,
                        z_copy,
                        x_copy,
                        y_copy,
                    )
                )
            else:
                _append_coord(coords, visited, x, y)
                break

    return coords


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
