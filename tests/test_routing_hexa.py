import itertools

import pytest

from paicorelib.coordinate import CoordXY, CoordZXYOffset
from paicorelib.routing_hexa import (
    AERPacket,
    AERPacketZXYCopy,
    aer_packet_area,
    aer_packet_walk,
    find_coordxy_shortest_path,
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
