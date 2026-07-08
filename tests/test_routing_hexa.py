import itertools

import pytest

from paicorelib.coordinate import CoordXY, CoordXYUnitVec, CoordZXYOffset
from paicorelib.routing_hexa import (
    AERPacket,
    AERPacketZXYCopy,
    PacketNextMove,
    aer_packet_area,
    aer_packet_copy_offsets,
    aer_packet_walk,
    find_coordxy_shortest_path,
    route_coord_path,
)


@pytest.mark.parametrize(
    "packet, n_core",
    [
        (
            AERPacket(
                CoordXY(0, 0), CoordZXYOffset(1, 0, 0), AERPacketZXYCopy(1, 0, 0)
            ),
            2,
        ),
        (
            AERPacket(
                CoordXY(0, 0), CoordZXYOffset(-1, 0, 0), AERPacketZXYCopy(-1, -1, 0)
            ),
            4,
        ),
        (
            AERPacket(
                CoordXY(0, 0), CoordZXYOffset(0, 0, 0), AERPacketZXYCopy(1, 1, 1)
            ),
            7,
        ),
        (
            AERPacket(
                CoordXY(0, 0), CoordZXYOffset(0, 0, 0), AERPacketZXYCopy(2, 3, 2)
            ),
            24,
        ),
    ],
)
def test_aer_package_walk(packet, n_core):
    area = aer_packet_area(packet.ncopy)
    covered = aer_packet_walk(packet)
    assert len(covered) == n_core

    assert area == n_core


def test_aer_package_area():
    for z, x, y in itertools.product(range(0, 5), range(0, 5), range(0, 5)):
        n = aer_packet_area((z, x, y))
        packet = AERPacket(ncopy=AERPacketZXYCopy(z, x, y))
        covered = aer_packet_walk(packet)
        n_covered = len(covered)

        assert n == n_covered


@pytest.mark.parametrize(
    "packet, expected",
    [
        (
            AERPacket(
                CoordXY(2, 3), CoordZXYOffset(1, -1, 0), AERPacketZXYCopy(1, 1, 0)
            ),
            [CoordXY(3, 4), CoordXY(2, 4), CoordXY(3, 5), CoordXY(4, 5)],
        ),
        (
            AERPacket(
                CoordXY(4, 4), CoordZXYOffset(-1, 2, -1), AERPacketZXYCopy(-1, 0, 1)
            ),
            [CoordXY(3, 3), CoordXY(5, 2), CoordXY(4, 1), CoordXY(5, 3), CoordXY(4, 2)],
        ),
    ],
)
def test_aer_packet_walk_preserves_zxy_order(packet, expected):
    assert aer_packet_walk(packet) == expected


def test_aer_packet_walk_does_not_mutate_packet():
    packet = AERPacket(
        CoordXY(2, 3), CoordZXYOffset(1, -1, 0), AERPacketZXYCopy(1, 1, 0)
    )

    aer_packet_walk(packet)

    assert packet.foothold == CoordXY(2, 3)
    assert packet.offset == CoordZXYOffset(1, -1, 0)
    assert packet.ncopy == AERPacketZXYCopy(1, 1, 0)


def test_aer_packet_neighbor_preserves_payload_and_moves_copy():
    packet = AERPacket(
        CoordXY(2, 3), CoordZXYOffset(1, -1, 0), AERPacketZXYCopy(1, 1, 0)
    )

    neighbor = packet.neighbor(CoordXYUnitVec.Z_POS)

    assert neighbor.foothold == CoordXY(3, 4)
    assert neighbor.offset == packet.offset
    assert neighbor.ncopy == packet.ncopy
    assert packet.foothold == CoordXY(2, 3)


def test_aer_packet_iter_keeps_single_packet_debug_steps():
    packet = AERPacket(ncopy=AERPacketZXYCopy(1, 0, 0))

    first_move, new_packet = next(iter(packet))

    assert first_move == PacketNextMove.MULTICAST
    assert new_packet is not None
    assert new_packet.foothold == CoordXY(1, 1)


def test_aer_packet_copy_offsets_matches_zero_offset_walk():
    ncopy = AERPacketZXYCopy(2, -1, 1)

    assert list(aer_packet_copy_offsets(ncopy)) == aer_packet_walk(
        AERPacket(ncopy=ncopy)
    )


def test_route_coord_path_uses_zxy_order():
    assert route_coord_path(CoordXY(2, 3), CoordZXYOffset(1, -2, 1)) == (
        CoordXY(2, 3),
        CoordXY(3, 4),
        CoordXY(2, 4),
        CoordXY(1, 4),
        CoordXY(1, 5),
    )


@pytest.mark.parametrize(
    "target, start, min_cost",
    [
        (CoordXY(-2, -1), CoordXY(3, 1), 5),
        (CoordXY(3, 3), CoordXY(0, -3), 6),
        (CoordXY(1, 1), CoordXY(3, -2), 5),
        (CoordXY(2, -2), CoordXY(-3, 4), 11),
        (CoordXY(-1, 2), CoordXY(0, 0), 3),
    ],
)
def test_find_shortest_path(target, start, min_cost):
    path, cost = find_coordxy_shortest_path(target, start)
    assert cost == min_cost
