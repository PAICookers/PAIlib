import numpy as np
import pytest
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.types import FRAME_DTYPE
from paicorelib.framelib.utils import framearray_header_check, header_check


def test_header_check():
    wrong_wf1 = np.array(
        # wrong header
        [0b0101_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01010101],
        dtype=FRAME_DTYPE,
    )

    with pytest.raises(ValueError):
        header_check(wrong_wf1, FH.WORK_TYPE1, strict=True)


def test_framearray_header_check():
    wrong_wf1 = np.array(
        # wrong header
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

    with pytest.raises(ValueError):
        framearray_header_check(wrong_wf1, FH.WORK_TYPE1, strict=True)
