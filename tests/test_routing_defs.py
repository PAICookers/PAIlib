import pytest

from paicorelib import Coord
from paicorelib import ReplicationId as RId
from paicorelib import (
    RoutingCoord,
    RoutingCost,
    RoutingDirection,
    RoutingLevel,
    get_multicast_cores,
    get_replication_id,
    get_routing_consumption,
)


@pytest.mark.parametrize(
    "coord, rid, num",
    [
        (Coord(0b00110, 0b01000), RId(0b11100, 0b00000), 8),
        (Coord(0b00001, 0b00000), RId(0b00011, 0b00001), 8),
        (Coord(0b11111, 0b00000), RId(0b01001, 0b00011), 16),
        (Coord(0b00000, 0b00000), RId(0b00001, 0b00010), 4),
        (Coord(0b00010, 0b00111), RId(0b00000, 0b00000), 1),
        (Coord(0b11111, 0b00111), RId(0b00001, 0b00000), 2),
        (Coord(0b10010, 0b10011), RId(0b11111, 0b11111), 1024),
        (Coord(0b11111, 0b11111), RId(0b00011, 0b11100), 32),
    ],
)
def test_get_multicast_cores_length(coord, rid, num):
    cores = get_multicast_cores(coord, rid)

    assert len(cores) == num


@pytest.mark.parametrize(
    "coord, rid, expected",
    [
        (
            Coord(0b00000, 0b00000),
            RId(0b00001, 0b00010),
            {
                Coord(0b00000, 0b00000),
                Coord(0b00001, 0b00000),
                Coord(0b00000, 0b00010),
                Coord(0b00001, 0b00010),
            },
        ),
        (Coord(0b00010, 0b00111), RId(0b00000, 0b00000), {Coord(0b00010, 0b00111)}),
        (
            Coord(0b11111, 0b00000),
            RId(0b10000, 0b00000),
            {Coord(0b11111, 0b00000), Coord(0b01111, 0b00000)},
        ),
    ],
)
def test_get_multicast_cores(coord, rid, expected):
    import time

    t1 = time.time()
    cores = get_multicast_cores(coord, rid)

    print(time.time() - t1)

    assert cores == expected


@pytest.mark.parametrize(
    "coords, expected",
    [
        (
            [
                Coord(0b00000, 0b00000),
                Coord(0b00001, 0b00000),
                Coord(0b00001, 0b00001),
            ],
            RId(0b00001, 0b000001),
        ),
        (
            [
                Coord(0b11111, 0b11111),
                Coord(0b00000, 0b00000),
            ],
            RId(0b11111, 0b11111),
        ),
        (
            [
                Coord(0b10000, 0b10000),
                Coord(0b00001, 0b10000),
                Coord(0b00001, 0b10000),
            ],
            RId(0b10001, 0b000000),
        ),
    ],
)
def test_get_replication_id(coords, expected):
    rid = get_replication_id(coords)

    assert rid == expected


@pytest.mark.parametrize(
    "n_core, expected_cost",
    [
        (1, RoutingCost(1, 1, 1, 1, 1)),
        (2, RoutingCost(2, 1, 1, 1, 1)),
        (3, RoutingCost(4, 1, 1, 1, 1)),
        (4, RoutingCost(4, 1, 1, 1, 1)),
        (7, RoutingCost(8, 2, 1, 1, 1)),
        (12, RoutingCost(16, 4, 1, 1, 1)),
        (20, RoutingCost(32, 8, 2, 1, 1)),
        (32, RoutingCost(32, 8, 2, 1, 1)),
        (33, RoutingCost(64, 16, 4, 1, 1)),
        (63, RoutingCost(64, 16, 4, 1, 1)),
        (64, RoutingCost(64, 16, 4, 1, 1)),
        (65, RoutingCost(128, 32, 8, 2, 1)),
        (127, RoutingCost(128, 32, 8, 2, 1)),
        (128, RoutingCost(128, 32, 8, 2, 1)),
        (1023, RoutingCost(1024, 256, 64, 16, 4)),
        (1024, RoutingCost(1024, 256, 64, 16, 4)),
    ],
)
def test_get_routing_consumption(n_core, expected_cost):
    cost = get_routing_consumption(n_core)

    assert cost == expected_cost


def test_routing_node_coord():
    path = []
    for i in range(5):
        path.append(RoutingDirection.X0Y0)

    coord = RoutingCoord(*path)

    assert coord.level == RoutingLevel.L0
    assert coord.coordinate == Coord(0, 0)

    path.clear()
    for i in range(6):
        path.append(RoutingDirection.X0Y0)

    with pytest.raises(TypeError):
        coord = RoutingCoord(*path)

    path.clear()
    path = [
        RoutingDirection.X0Y1,
        RoutingDirection.X1Y1,
        RoutingDirection.X0Y0,
        RoutingDirection.X0Y1,
        RoutingDirection.X0Y1,
    ]

    coord = RoutingCoord(*path)
    assert coord.level == RoutingLevel.L0
    assert coord.coordinate == Coord(0b01000, 0b11011)

    path.clear()
    path = [
        RoutingDirection.X0Y0,
        RoutingDirection.X1Y1,
        RoutingDirection.X0Y0,
        RoutingDirection.ANY,
        RoutingDirection.X0Y1,
    ]

    coord = RoutingCoord(*path)
    assert coord.level == RoutingLevel.L2

    with pytest.raises(AttributeError):
        coord.coordinate

    path.clear()
    path = [
        RoutingDirection.ANY,
        RoutingDirection.X1Y1,
        RoutingDirection.X0Y0,
        RoutingDirection.ANY,
        RoutingDirection.X0Y1,
    ]

    coord = RoutingCoord(*path)
    assert coord.level == RoutingLevel.L5

    with pytest.raises(AttributeError):
        coord.coordinate
