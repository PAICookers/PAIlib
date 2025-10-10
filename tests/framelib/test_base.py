import numpy as np
import pytest

from paicorelib import Coord
from paicorelib import ReplicationId as RId
from paicorelib.framelib.base import Frame, FramePackage, FramePackagePayload
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import FramePackageType as FPType
from paicorelib.framelib.types import FRAME_DTYPE
from paicorelib.routing_defs import _rid_unset


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
