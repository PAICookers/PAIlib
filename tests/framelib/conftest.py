import pytest

from paicorelib.coordinate import CoordZXYOffset
from paicorelib.routing_hexa import AERPacketZXYCopy
from tests.utils import build_v2_packet_route


@pytest.fixture
def v2_packet_route() -> tuple[CoordZXYOffset, AERPacketZXYCopy]:
    return build_v2_packet_route()
