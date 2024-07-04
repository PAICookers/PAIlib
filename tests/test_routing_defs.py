import pytest

from paicorelib import Coord
from paicorelib import ReplicationId as RId
from paicorelib import RoutingCoord, RoutingCost, RoutingDirection
from paicorelib import RoutingLevel as Level
from paicorelib import (
    RoutingPath,
    get_multicast_cores,
    get_replication_id,
    get_routing_consumption,
)
from paicorelib.hw_defs import HwParams
from paicorelib.routing_defs import MAX_ROUTING_PATH_LENGTH

X0Y0 = RoutingDirection.X0Y0
X1Y0 = RoutingDirection.X1Y0
X0Y1 = RoutingDirection.X0Y1
X1Y1 = RoutingDirection.X1Y1
ANY = RoutingDirection.ANY


@pytest.mark.parametrize("y_priority", [True, False])
def test_RoutingDirection_to_index(y_priority, monkeypatch):
    monkeypatch.setattr(HwParams, "COORD_Y_PRIORITY", y_priority)

    direc_idx = (
        (X0Y0, X0Y1, X1Y0, X1Y1)
        if HwParams.COORD_Y_PRIORITY
        else (X0Y0, X1Y0, X0Y1, X1Y1)
    )

    for idx, d in enumerate(direc_idx):
        assert d.to_index() == idx


def test_RoutingCoord_lt():
    rc1 = RoutingCoord(X0Y1, X0Y0)
    rc2 = RoutingCoord(X0Y0, X1Y1)

    assert rc2 < rc1
    assert not rc1 < rc2


def test_RoutingPath_instance():
    rd_lst = [X1Y0, X1Y1, X0Y0]
    rp1 = RoutingPath(*rd_lst, reverse=True)
    rc1 = rp1.routing_coord
    rp2 = RoutingPath(*rd_lst, reverse=False)
    rc2 = rp2.routing_coord

    assert rc1 == RoutingCoord(ANY, ANY, X0Y0, X1Y1, X1Y0)
    assert rc2 == RoutingCoord(X1Y0, X1Y1, X0Y0)


@pytest.mark.parametrize(
    "n_core, expected",
    [
        (0, RoutingPath(*(X0Y0, X0Y0, X0Y0, X0Y0, X0Y0))),
        (1, RoutingPath(*(X0Y0, X0Y0, X0Y0, X0Y0, X0Y1))),
        (10, RoutingPath(*(X0Y0, X0Y0, X0Y0, X1Y0, X1Y0))),
        (1023, RoutingPath(*(X1Y1, X1Y1, X1Y1, X1Y1, X1Y1))),
    ],
)
def test_n_core2routing_path(n_core, expected):
    rp = RoutingPath.n_core2routing_path(n_core)
    assert rp == expected


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
    cores = get_multicast_cores(coord, rid)
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
        (1, RoutingCost(1, 1, 1, 1, 1), Level.L1),
        (2, RoutingCost(2, 1, 1, 1, 1), Level.L1),
        (3, RoutingCost(4, 1, 1, 1, 1), Level.L1),
        (7, RoutingCost(8, 2, 1, 1, 1), Level.L2),
        (12, RoutingCost(16, 4, 1, 1, 1), Level.L2),
        (20, RoutingCost(32, 8, 2, 1, 1), Level.L3),
        (32, RoutingCost(32, 8, 2, 1, 1), Level.L3),
        (63, RoutingCost(64, 16, 4, 1, 1), Level.L3),
        (65, RoutingCost(128, 32, 8, 2, 1), Level.L4),
        (127, RoutingCost(128, 32, 8, 2, 1), Level.L4),
        (128, RoutingCost(128, 32, 8, 2, 1), Level.L4),
        (500, RoutingCost(512, 128, 32, 8, 2), Level.L5),
        (1024, RoutingCost(1024, 256, 64, 16, 4), Level.L5),
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
        path.append(X0Y0)

    coord = RoutingCoord(*path)
    assert coord.level == Level.L0
    assert coord.to_coord() == Coord(0, 0)

    path.clear()
    for _ in range(MAX_ROUTING_PATH_LENGTH + 1):  # Out of length
        path.append(X0Y0)

    with pytest.raises(TypeError):
        coord = RoutingCoord(*path)

    path.clear()
    path = [X0Y1, X1Y1, X0Y0, X0Y1, X0Y1]

    coord = RoutingCoord(*path)
    assert coord.level == Level.L0
    assert coord.to_coord() == Coord(0b01000, 0b11011)

    path.clear()
    path = [X0Y0, X1Y1, X0Y0, ANY, X0Y1]

    coord = RoutingCoord(*path)
    assert coord.level == Level.L2

    with pytest.raises(AttributeError):
        coord.to_coord()

    path.clear()
    path = [X1Y1, X0Y0, ANY, ANY, X0Y1]

    coord = RoutingCoord(*path)
    assert coord.level == Level.L3

    with pytest.raises(AttributeError):
        coord.to_coord()
