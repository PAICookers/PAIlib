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
from paicorelib.routing_defs import MAX_ROUTING_PATH_LENGTH


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
    "n_core, expected_cost, expected_lx",
    [
        (1, RoutingCost(1, 1, 1, 1, 1), RoutingLevel.L1),
        (2, RoutingCost(2, 1, 1, 1, 1), RoutingLevel.L1),
        (3, RoutingCost(4, 1, 1, 1, 1), RoutingLevel.L1),
        (7, RoutingCost(8, 2, 1, 1, 1), RoutingLevel.L2),
        (12, RoutingCost(16, 4, 1, 1, 1), RoutingLevel.L2),
        (20, RoutingCost(32, 8, 2, 1, 1), RoutingLevel.L3),
        (32, RoutingCost(32, 8, 2, 1, 1), RoutingLevel.L3),
        (63, RoutingCost(64, 16, 4, 1, 1), RoutingLevel.L3),
        (65, RoutingCost(128, 32, 8, 2, 1), RoutingLevel.L4),
        (127, RoutingCost(128, 32, 8, 2, 1), RoutingLevel.L4),
        (128, RoutingCost(128, 32, 8, 2, 1), RoutingLevel.L4),
        (500, RoutingCost(512, 128, 32, 8, 2), RoutingLevel.L5),
        (1024, RoutingCost(1024, 256, 64, 16, 4), RoutingLevel.L5),
    ],
)
def test_get_routing_consumption(n_core, expected_cost, expected_lx):
    cost = get_routing_consumption(n_core)

    assert cost == expected_cost
    assert cost.get_routing_level() == expected_lx
    assert cost[expected_lx.value]


def test_get_routing_consumption_outrange():
    n_core, expected_cost = 1200, RoutingCost(2048, 512, 128, 32, 8, 2)
    cost = get_routing_consumption(n_core)
    assert cost == expected_cost

    with pytest.raises(ValueError):
        cost.get_routing_level()


def test_RoutingCoord():
    path = []
    for _ in range(MAX_ROUTING_PATH_LENGTH):
        path.append(RoutingDirection.X0Y0)

    coord = RoutingCoord(*path)
    assert coord.level == RoutingLevel.L0
    assert coord.to_coord() == Coord(0, 0)

    path.clear()
    for _ in range(MAX_ROUTING_PATH_LENGTH + 1):  # Out of length
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
    assert coord.to_coord() == Coord(0b01000, 0b11011)

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
        coord.to_coord()

    path.clear()
    path = [
        RoutingDirection.X1Y1,
        RoutingDirection.X0Y0,
        RoutingDirection.ANY,
        RoutingDirection.ANY,
        RoutingDirection.X0Y1,
    ]

    coord = RoutingCoord(*path)
    assert coord.level == RoutingLevel.L3

    with pytest.raises(AttributeError):
        coord.to_coord()
