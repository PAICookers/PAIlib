import numpy as np
import pytest

from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.types import FRAME_DTYPE
from paicorelib.framelib.utils import framearray_header_check


@pytest.mark.parametrize(
    "frames",
    [
        # wrong header
        np.array(
            [
                0b0100_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01010101,
                0b0100_0000100001_00011_11100_00000_00001_000_00000000000_00000000_00000001,
                0b0100_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01000000,
                0b0100_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01000000,
                0b0101_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01000000,
                0b0110_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01000000,
            ],
            dtype=FRAME_DTYPE,
        )
    ],
)
def test_framearray_header_check(frames):
    with pytest.raises(ValueError):
        framearray_header_check(frames, FH.WORK_TYPE1, strict=True)
