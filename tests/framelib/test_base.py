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
)
from paicorelib.framelib.frame_defs import FFV2, FramePackageType
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import FramePackageType as FPType
from paicorelib.framelib.types import FRAME_DTYPE
from paicorelib.routing_defs import _rid_unset
from paicorelib.routing_hexa import AERPacketZXYCopy


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
                np.random.randint(0, 128, size=(8,), dtype=FRAME_DTYPE),
            ),
            (FH.WORK_TYPE2, Coord(3, 4), Coord(1, 2), RId(4, 4), FRAME_DTYPE(0)),
        ],
    )
    def test_Frame_instance(self, header, chip_coord, core_coord, rid, payload):
        f = Frame(chip_coord, core_coord, rid, payload, header=header)  # type: ignore
        assert len(f) == 1 if isinstance(payload, int) else payload.size

        # check __str__
        print(f)

        # check .value
        _ = f.value

    @pytest.mark.parametrize(
        "header, chip_coord, core_coord, rid, payload, packages",
        [
            (
                FH.CONFIG_TYPE4,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                FramePackagePayload(0, FPType.CONF_TESTOUT, 4 * 50),
                np.random.randint(
                    0, np.iinfo(np.uint64).max, size=(4 * 50,), dtype=FRAME_DTYPE
                ),
            ),
            (
                FH.TEST_TYPE3,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                FramePackagePayload(200, FPType.TESTIN, 4 * 100),
                np.zeros(0),  # Empty package
            ),
        ],
    )
    def test_FramePackage_instance(
        self, header, chip_coord, core_coord, rid, payload, packages
    ):
        fp = FramePackage(chip_coord, core_coord, rid, payload, packages, header=header)  # type: ignore
        assert len(fp) == packages.size + 1

        # check __str__
        print(fp)

        # check .value
        _ = fp.value


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
        frames = pkg_header.value

        assert pkg_header.payload.n_package == n_package

        oz = (
            int(frames >> FFV2.GENERAL_CORE_XY_ADDR_OFFSET)
            & FFV2.GENERAL_CORE_XY_ADDR_MASK
        )
        ox = (
            int(frames >> FFV2.GENERAL_CORE_X_ADDR_OFFSET)
            & FFV2.GENERAL_CORE_X_ADDR_MASK
        )
        oy = (
            int(frames >> FFV2.GENERAL_CORE_Y_ADDR_OFFSET)
            & FFV2.GENERAL_CORE_Y_ADDR_MASK
        )
        cz = (
            int(frames >> FFV2.GENERAL_COPY_XY_ADDR_OFFSET)
            & FFV2.GENERAL_COPY_XY_ADDR_MASK
        )
        cx = (
            int(frames >> FFV2.GENERAL_COPY_X_ADDR_OFFSET)
            & FFV2.GENERAL_COPY_X_ADDR_MASK
        )
        cy = (
            int(frames >> FFV2.GENERAL_COPY_Y_ADDR_OFFSET)
            & FFV2.GENERAL_COPY_Y_ADDR_MASK
        )
        assert (oz, ox, oy) == (33, 34, 3)
        assert (cz, cx, cy) == (3, 34, 0)

        pkg_header.payload.n_package = 100
        assert pkg_header.payload.n_package == 100
