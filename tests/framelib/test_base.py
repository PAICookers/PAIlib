import numpy as np
import pytest

from paicorelib import Coord
from paicorelib import ReplicationId as RId
from paicorelib.framelib.base import Frame, FramePackage
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.utils import print_frame


class TestFrameBasicObj:
    @pytest.mark.parametrize(
        "header, chip_coord, core_coord, rid, payload",
        [
            (
                FH.CONFIG_TYPE1,
                Coord(1, 2),
                Coord(3, 4),
                RId(0, 0),
                np.asarray(list(range(5)), dtype=np.uint64),
            ),
            (
                FH.CONFIG_TYPE1,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                np.asarray([3], dtype=np.uint64),
            ),
            (
                FH.CONFIG_TYPE1,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                np.random.randint(0, 128, size=(8,), dtype=np.uint64),
            ),
            (
                FH.CONFIG_TYPE1,
                Coord(3, 4),
                Coord(1, 2),
                RId(4, 4),
                np.asarray([12, 12], dtype=np.uint64),
            ),
        ],
    )
    def test_Frame_instance(self, header, chip_coord, core_coord, rid, payload):
        frame = Frame(header, chip_coord, core_coord, rid, payload)
        print(frame)
        print_frame(frame.value)

        assert len(frame) == payload.size

    @pytest.mark.parametrize(
        "header, chip_coord, core_coord, rid, payload, packages",
        [
            (
                FH.CONFIG_TYPE1,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                4,
                np.array([1, 2, 3], dtype=np.uint64),
            )
        ],
    )
    def test_FramePackage_instance(
        self, header, chip_coord, core_coord, rid, payload, packages
    ):
        framepackage = FramePackage(
            header, chip_coord, core_coord, rid, payload, packages
        )
        print(framepackage)
        print_frame(framepackage.value)

        assert len(framepackage) == packages.size + 1
