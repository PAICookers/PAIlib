import numpy as np
import pytest

from paicorelib import Coord
from paicorelib import ReplicationId as RId
from paicorelib.coordinate import CoordZXYOffset
from paicorelib.framelib.base import (
    Frame,
    FramePackage,
    FramePackageHeaderV2,
    FramePackagePayload,
    FramePackagePayloadV2,
    FrameV2,
    get_frame_dest_v2,
)
from paicorelib.framelib.frame_defs import FFV2, FramePackageType
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import FramePackageType as FPType
from paicorelib.framelib.types import FRAME_DTYPE
from paicorelib.routing_defs import _rid_unset
from paicorelib.routing_hexa import AERPacketZXYCopy

from ..utils import bit_field


class TestFrameInstance:
    @pytest.mark.parametrize(
        "header, chip_coord, core_coord, rid, payload",
        [
            (
                FH.CONFIG_TYPE1,
                Coord(1, 2),
                Coord(3, 4),
                _rid_unset(),
                np.array(list(range(5)), dtype=FRAME_DTYPE),
            ),
            (FH.CONFIG_TYPE2, Coord(1, 2), Coord(3, 4), RId(5, 5), 123),
            (
                FH.TEST_TYPE2,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                "random_payload",
            ),
            (FH.WORK_TYPE2, Coord(3, 4), Coord(1, 2), RId(4, 4), FRAME_DTYPE(0)),
        ],
    )
    def test_frame_instance(
        self, fixed_rng, header, chip_coord, core_coord, rid, payload
    ):
        if isinstance(payload, str) and payload == "random_payload":
            payload = fixed_rng.integers(0, 128, size=(8,), dtype=FRAME_DTYPE)

        f = Frame(chip_coord, core_coord, rid, payload, header=header)  # type: ignore
        assert len(f) == 1 if isinstance(payload, int) else payload.size
        assert f.value.dtype == FRAME_DTYPE
        print(f)

    @pytest.mark.parametrize(
        "header, chip_coord, core_coord, rid, payload, packages",
        [
            (
                FH.CONFIG_TYPE4,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                FramePackagePayload(0, FPType.CONF_TESTOUT, 4 * 50),
                "random_packages",
            ),
            (
                FH.TEST_TYPE3,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                FramePackagePayload(200, FPType.TESTIN, 4 * 100),
                np.zeros(0),
            ),
        ],
    )
    def test_frame_package_instance(
        self, fixed_rng, header, chip_coord, core_coord, rid, payload, packages
    ):
        if isinstance(packages, str) and packages == "random_packages":
            packages = fixed_rng.integers(
                0, np.iinfo(np.uint64).max, size=(4 * 50,), dtype=FRAME_DTYPE
            )

        fp = FramePackage(chip_coord, core_coord, rid, payload, packages, header=header)  # type: ignore
        assert len(fp) == packages.size + 1
        assert fp.value.dtype == FRAME_DTYPE
        print(fp)


class TestFramePackagePayloadV2:
    def test_value(self):
        payload = FramePackagePayloadV2(10, FramePackageType.CONF_TESTOUT, 99)

        assert (
            bit_field(
                payload.value,
                FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET,
                FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK,
            )
            == 10
        )
        assert (
            bit_field(
                payload.value,
                FFV2.GENERAL_PACKAGE_TYPE_OFFSET,
                FFV2.GENERAL_PACKAGE_TYPE_MASK,
            )
            == FramePackageType.CONF_TESTOUT.value
        )
        assert (
            bit_field(
                payload.value,
                FFV2.GENERAL_PACKAGE_NUM_OFFSET,
                FFV2.GENERAL_PACKAGE_NUM_MASK,
            )
            == 99
        )

    @pytest.mark.parametrize(
        "start_addr, n_package",
        [
            (FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK + 1, 0),
            (0, FFV2.GENERAL_PACKAGE_NUM_MASK + 1),
            (-1, 0),
            (0, -1),
        ],
    )
    def test_rejects_out_of_range_values(self, start_addr, n_package):
        with pytest.raises(ValueError):
            FramePackagePayloadV2(start_addr, FramePackageType.CONF_TESTOUT, n_package)


class TestFrameV2:
    def test_value_matches_destination(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        frame = FrameV2(FH.CTRL_TYPE1, pkt_offset, pkt_ncopy, payload=123)

        assert frame.frame_dest == get_frame_dest_v2(
            FH.CTRL_TYPE1, pkt_offset, pkt_ncopy
        )
        assert frame.value.dtype == FRAME_DTYPE
        assert frame.value.shape == (1,)
        assert (
            bit_field(
                frame.value[0], FFV2.GENERAL_HEADER_OFFSET, FFV2.GENERAL_HEADER_MASK
            )
            == FH.CTRL_TYPE1.value
        )
        assert (
            bit_field(
                frame.value[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK
            )
            == 123
        )
        assert "header:" in str(frame)


class TestFramePackageHeaderV2:
    def test_make_pkg_header(self):
        header = FH.CONFIG_TYPE3
        pkt_offset = CoordZXYOffset(-1, -2, 3)
        pkt_ncopy = AERPacketZXYCopy(3, -2, 0)
        start_addr = 10
        pkg_type = FramePackageType.CONF_TESTOUT
        n_package = 99

        pkg_header = FramePackageHeaderV2.make_pkg_header(
            header, pkt_offset, pkt_ncopy, start_addr, pkg_type, n_package
        )
        frame = pkg_header.value[0]

        assert pkg_header.payload.n_package == n_package
        assert (
            bit_field(
                frame, FFV2.GENERAL_CORE_XY_ADDR_OFFSET, FFV2.GENERAL_CORE_XY_ADDR_MASK
            )
            == 33
        )
        assert (
            bit_field(
                frame, FFV2.GENERAL_CORE_X_ADDR_OFFSET, FFV2.GENERAL_CORE_X_ADDR_MASK
            )
            == 34
        )
        assert (
            bit_field(
                frame, FFV2.GENERAL_CORE_Y_ADDR_OFFSET, FFV2.GENERAL_CORE_Y_ADDR_MASK
            )
            == 3
        )
        assert (
            bit_field(
                frame, FFV2.GENERAL_COPY_XY_ADDR_OFFSET, FFV2.GENERAL_COPY_XY_ADDR_MASK
            )
            == 3
        )
        assert (
            bit_field(
                frame, FFV2.GENERAL_COPY_X_ADDR_OFFSET, FFV2.GENERAL_COPY_X_ADDR_MASK
            )
            == 34
        )
        assert (
            bit_field(
                frame, FFV2.GENERAL_COPY_Y_ADDR_OFFSET, FFV2.GENERAL_COPY_Y_ADDR_MASK
            )
            == 0
        )

        pkg_header.payload.n_package = 100
        assert pkg_header.payload.n_package == 100
        assert "payload:" in str(pkg_header)
