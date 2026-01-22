import contextlib
import math

import numpy as np
import pytest

from paicorelib.coordinate import CoordZXYOffset
from paicorelib.core_defs import LCN_EX
from paicorelib.core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)
from paicorelib.core_model_v2 import OfflineCoreRegV2
from paicorelib.framelib import FRAME_DTYPE
from paicorelib.framelib.frame_defs import FFV2
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import OfflineConfigFrame2FormatV2 as Off_Cfg2_V2
from paicorelib.framelib.frame_defs import OfflineWorkFrame1FormatV2 as Off_Work1_V2
from paicorelib.framelib.frame_gen_v2 import (
    OfflineFrameGenV2,
    weight_csc_pack,
    weight_csc_unpack,
    weight_dense_pack,
    weight_dense_unpack,
)
from paicorelib.framelib.types import (
    LUT_ACTIVATION_DTYPE,
    LUT_POTENTIAL_DTYPE,
    FrameArrayType,
)
from paicorelib.framelib.utils import print_frame, single_frame_header_check
from paicorelib.neuron_defs import ResetMode
from paicorelib.neuron_defs_v2 import (
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
)
from paicorelib.neuron_model_v2 import (
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
    OfflineNeuHalfAttrsV2,
)
from paicorelib.routing_hexa import AERPacketZXYCopy
from paicorelib.utils import _mask
from tests.utils import gen_random_array


def parse_package_header(frames: FrameArrayType) -> tuple[int, int, int]:
    payload = (frames[0] >> FFV2.GENERAL_PAYLOAD_OFFSET) & FFV2.GENERAL_PAYLOAD_MASK
    pkg_type = (
        payload >> FFV2.GENERAL_PACKAGE_TYPE_OFFSET
    ) & FFV2.GENERAL_PACKAGE_TYPE_MASK
    start_addr = (
        payload >> FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET
    ) & FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK
    n_package = (
        payload >> FFV2.GENERAL_PACKAGE_NUM_OFFSET
    ) & FFV2.GENERAL_PACKAGE_NUM_MASK

    return pkg_type, start_addr, n_package


def extract_lut_from_cf2(cf2: FrameArrayType, act_dtype: LUT_ACTIVATION_DTYPE):
    F = Off_Cfg2_V2
    assert cf2.size == 257
    assert single_frame_header_check(cf2[0], FH.CONFIG_TYPE2)

    lut = cf2[1:]
    arr_pot = ((lut >> F.POTENTIAL_OFFSET) & F.POTENTIAL_MASK).astype(
        LUT_POTENTIAL_DTYPE
    )
    arr_act = ((lut >> F.ACTIVATION_OFFSET) & F.ACTIVATION_MASK).astype(act_dtype)
    return arr_pot, arr_act


class TestFrameGenV2:
    def test_make_package(self):
        pass


class TestOfflineFrameGenV2:
    # def _parse_frame(self, frame):
    #     """Helper to unpack a standard frame (Header/Addr/Payload)."""
    #     val = int(frame)
    #     header = (val >> FFV2.GENERAL_HEADER_OFFSET) & FFV2.GENERAL_HEADER_MASK
    #     core_addr = (val >> FFV2.GENERAL_CORE_ADDR_OFFSET) & FFV2.GENERAL_CORE_ADDR_MASK
    #     copy_addr = (val >> FFV2.GENERAL_COPY_ADDR_OFFSET) & FFV2.GENERAL_COPY_ADDR_MASK
    #     payload = val & FFV2.GENERAL_PAYLOAD_MASK
    #     return header, core_addr, copy_addr, payload

    # def test_gen_data_frame(self):
    #     """Test Work Frame Type 1 with dynamic widths."""
    #     core, copy = 0x10, 0x20
    #     ts, axon, data = 10, 0x100, 0x55

    #     # Define dynamic widths (must sum to 17 for time + axon)
    #     t_w, a_w, d_w = 7, 10, 8

    #     frames = OfflineFrameGenV2.gen_data_frame(
    #         core, copy, ts, axon, data, time_width=t_w, axon_width=a_w, data_width=d_w
    #     )
    #     assert frames.shape == (1,)

    #     hd, c, cp, pl = self._parse_frame(frames[0])
    #     assert hd == FH.WORK_TYPE1
    #     assert c == core
    #     assert cp == copy

    #     # Calculate expected payload manually
    #     # Logic: mixed = (ts << axon_width) | axon
    #     # Layout: MSB(1bit) at 60 | Low(16bits) at 8 | Data at 0
    #     mixed = (ts << a_w) | axon
    #     msb = (mixed >> 16) & 1
    #     low = mixed & 0xFFFF

    #     expected_payload = (msb << 60) | (low << 8) | data
    #     assert pl == expected_payload

    # def test_gen_vjt_frame(self):
    #     """Test Work Frame Type 2 with dynamic widths."""
    #     core, copy = 0x10, 0x20
    #     ts, axon, vjt = 5, 0x200, 0xAA

    #     t_w, a_w, v_w = 7, 10, 8

    #     frames = OfflineFrameGenV2.gen_vjt_frame(
    #         core, copy, ts, axon, vjt, time_width=t_w, axon_width=a_w, vjt_width=v_w
    #     )
    #     assert frames.shape == (1,)

    #     hd, c, cp, pl = self._parse_frame(frames[0])
    #     assert hd == FH.WORK_TYPE2

    #     mixed = (ts << a_w) | axon
    #     msb = (mixed >> 16) & 1
    #     low = mixed & 0xFFFF

    #     # VJT at offset 0
    #     expected_payload = (msb << 60) | (low << 8) | vjt
    #     assert pl == expected_payload

    def test_cf1(self):
        core_reg = OfflineCoreRegV2(
            name="core_reg",
            snn_ann=SNNMode.SNN,
            max_pooling=PoolingMode.AVERAGE,
            add_potential=AddPotentialMode.NORMAL,
            zero_output=ZeroOutputMode.DISABLE,
            input_sign=DataSign.SIGNED,
            input_width=DataWidth.WIDTH_8BIT,
            output_sign=DataSign.SIGNED,
            output_width=DataWidth.WIDTH_8BIT,
            weight_sign=DataSign.SIGNED,
            weight_width=DataWidth.WIDTH_8BIT,
            lcn=LCN_EX.LCN_8X,
            target_lcn=LCN_EX.LCN_8X,
            axon_skew=0,
            neuron_number=200,
            test_core_xy=5,
            test_core_x=0,
            test_core_y=0,
            global_send=0b100_0000,
            csc_accelerate=CSCAccelerateMode.ENABLE,
            global_receive=0b001_0000,
            thread_number=1,
            busy_cycle=2,
            delay_cycle=2,
            width_cycle=2,
            tick_start=1,
        )

        frames = OfflineFrameGenV2.gen_config_frame1(
            CoordZXYOffset(1, 1, 1), core_reg, AERPacketZXYCopy(0, 1, -1), 0
        )
        assert frames.dtype == FRAME_DTYPE
        assert frames.size == 1 + 3

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("act_dtype", [np.uint8, np.int8])
    def test_cf2(self, random, act_dtype):
        if random:
            arr_pot = np.arange(-128, 128, dtype=LUT_POTENTIAL_DTYPE)
            if act_dtype == np.uint8:
                arr_act = np.ones((256,), dtype=act_dtype)
            else:
                arr_act = np.full((256,), -1, dtype=act_dtype)
        else:
            arr_pot = gen_random_array((256,), dtype=LUT_POTENTIAL_DTYPE)
            arr_act = gen_random_array((256,), dtype=act_dtype)

        frames = OfflineFrameGenV2.gen_config_frame2(
            CoordZXYOffset(1, 1, 1), arr_pot, arr_act
        )
        assert frames.size == 1 + 256

        arr_pot2, arr_act2 = extract_lut_from_cf2(frames, act_dtype)
        assert np.array_equal(arr_pot, arr_pot2)
        assert np.array_equal(arr_act, arr_act2)

    def test_gen_config_frame3_pkg_header(self):
        frames = OfflineFrameGenV2.gen_config_frame3_pkg_header(
            CoordZXYOffset(1, 1, 1), 0, 0
        )
        assert frames.dtype == FRAME_DTYPE
        assert frames.size == 1

    def test_gen_pkg_half_neu(self):
        dest_info = OfflineNeuDestInfoV2(
            tick_relative=0,
            addr_axon=1,
            addr_core_xy=0,
            addr_core_x=1,
            addr_core_y=-1,
            addr_copy_xy=1,
            addr_copy_x=1,
            addr_copy_y=1,
        )
        half_attrs = OfflineNeuHalfAttrsV2(
            weight_skew=0,
            weight_address_start=0,
            weight_address_end=128,
            fold_type=FoldType.UNFOLDED,
            neuron_type=NeuronType.HALF,
            output_type=OutputType.VALUE,
        )

        pkg_half_neu = OfflineFrameGenV2._gen_pkg_half_neu(
            dest_info.model_dump(), half_attrs.model_dump()
        )
        assert pkg_half_neu.dtype == FRAME_DTYPE
        assert pkg_half_neu.size == 2

    def test_gen_pkg_full_neu(self):
        dest_info = OfflineNeuDestInfoV2(
            tick_relative=0,
            addr_axon=1,
            addr_core_xy=0,
            addr_core_x=1,
            addr_core_y=-1,
            addr_copy_xy=1,
            addr_copy_x=1,
            addr_copy_y=1,
        )
        full_attrs1 = OfflineNeuFullAttrsV2Part1(
            weight_skew=0,
            weight_address_start=0,
            weight_address_end=128,
            fold_type=FoldType.UNFOLDED,
            neuron_type=NeuronType.HALF,
            output_type=OutputType.VALUE,
        )
        full_attrs2 = OfflineNeuFullAttrsV2Part2(
            reset_mode=ResetMode.MODE_LINEAR,
            reset_v=0,
            threshold_neg_mode=ThresholdNegMode.FIRE,
            threshold_pos_mode=ThresholdPosMode.FIRE,
            threshold_neg=0,
            threshold_pos=1,
            lateral_inhibition=LateralInhibitionMode.DISABLE,
            leak_multi_sequence=LeakMultiComparisonOrder.AFTER_COMPARE,
            leak_multi_input=LeakMultiInputMode.ENABLE,
            leak_multi_mode=LeakMultiMode.ENABLE,
            leak_add_mode=LeakAddMode.FORWARD,
            leak_tau=2,
            leak_v=0,
            weight_compress=WeightCompressType.SPARSE,
        )

        pkg_full_neu = OfflineFrameGenV2._gen_pkg_full_neu(
            dest_info.model_dump(), full_attrs1.model_dump(), full_attrs2.model_dump()
        )
        assert pkg_full_neu.dtype == FRAME_DTYPE
        assert pkg_full_neu.size == 4

    def test_gen_pkg_folded_neu(self):
        attrs1 = OfflineNeuFoldedAttrsV2Part1(
            fold_range_xy=1,
            fold_range_x=1,
            fold_range_y=1,
            fold_skew_xy=0,
            fold_skew_x=1,
            fold_skew_y=1,
            fold_axon_xy=1,
            fold_axon_x=1,
            fold_axon_y=0,
            fold_number=1,
        )

        attrs2_1 = OfflineNeuFoldedAttrsV2Part2(
            fold_vjt_3=100, fold_vjt_2=200, fold_vjt_1=300, fold_vjt_0=400
        )
        attrs2_2 = OfflineNeuFoldedAttrsV2Part2(
            fold_vjt_3=0, fold_vjt_2=1, fold_vjt_1=2, fold_vjt_0=3
        )

        pkg_folded_neu = OfflineFrameGenV2._gen_pkg_folded_neu(
            attrs1.model_dump(), *[attrs2_1.model_dump(), attrs2_2.model_dump()]
        )
        assert pkg_folded_neu.dtype == FRAME_DTYPE
        assert pkg_folded_neu.size == 2 + 2 * 2

    def test_gen_config_frame3_weight_pkg(self, fixed_rng):
        weight = fixed_rng.integers(-128, 127, size=(32,))

        result = OfflineFrameGenV2.gen_config_frame3_weight_pkg(
            weight, 8, csc_compress=False
        )
        assert result.shape == (4,)

    @pytest.mark.parametrize(
        "ts, axon, data, target_lcn",
        [
            (
                np.array([0b0011_0100, 0b0011_0100]),
                np.array([0b0_0100_1011_0011, 0b100_1011_0011]),
                np.array([-16, 1], dtype=np.int8),
                LCN_EX.LCN_4X,  # 5, 12
            ),
            (
                0b1011_0100,
                0b1_1011_0011,
                np.array([10], dtype=np.uint8),
                LCN_EX.LCN_1X,  # 8, 9
            ),
        ],
    )
    def test_wf1(self, ts, axon, data, target_lcn):
        pkt_offset = CoordZXYOffset(1, 1, 1)
        pkt_ncopy = AERPacketZXYCopy(0, 1, 1)
        wf1 = OfflineFrameGenV2.gen_work_frame1(
            pkt_offset, pkt_ncopy, ts, axon, target_lcn, data
        )
        print_frame(wf1, version=2, target_lcn=target_lcn)

        TS_WIDTH, AX_WIDTH = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[target_lcn]
        if isinstance(ts, np.ndarray):
            ts = ts[0]
        if isinstance(axon, np.ndarray):
            axon = axon[0]

        assert (
            wf1[0] >> Off_Work1_V2.TIMESTEP_HIGH7_OFFSET
        ) & Off_Work1_V2.TIMESTEP_HIGH7_MASK == (
            ts >> (TS_WIDTH - 1)
        ) & Off_Work1_V2.TIMESTEP_HIGH7_MASK
        assert (wf1[0] >> Off_Work1_V2.AXON_ADDR_OFFSET) & _mask(AX_WIDTH) == axon


@pytest.mark.parametrize(
    "size, weight_width, signed",
    [
        (100, 8, True),
        (128, 8, False),
        (100, 4, True),
        (64, 4, False),
        (60, 2, True),
        (200, 1, True),
        (256, 1, False),
        (0, 8, True),
        (0, 8, False),
    ],
)
def test_weight_uncompressed_unpack(size, weight_width, signed, fixed_rng):
    if signed:
        _dtype = np.int8
        _max = _mask(weight_width - 1)
        _min = -(_max + 1)
    else:
        _dtype = np.uint8
        _min, _max = 0, _mask(weight_width)

    weight = fixed_rng.integers(_min, _max, size=size, dtype=_dtype)
    mapped = weight_dense_pack(weight, weight_width)
    assert mapped.dtype == FRAME_DTYPE

    align_size = 128 // weight_width
    expected_size = math.ceil(weight.size / align_size) * (
        128 // (FRAME_DTYPE(0).nbytes * 8)
    )
    assert mapped.shape == (expected_size,)

    unmapped = weight_dense_unpack(mapped, weight_width, signed, size)
    assert unmapped.dtype == _dtype
    assert unmapped.size == weight.size
    assert np.array_equal(weight, unmapped)


@pytest.mark.parametrize(
    "size, weight_width, signed",
    [
        (100, 8, True),
        (128, 8, False),
        (100, 4, True),
        (64, 4, False),
        (60, 2, True),
        (200, 1, True),
        (256, 1, False),
        (0, 8, True),
        (0, 8, False),
    ],
)
def test_weight_csc_unpack(size, weight_width, signed, fixed_rng):
    if signed:
        _dtype = np.int8
        _max = _mask(weight_width - 1)
        _min = -(_max + 1)
    else:
        _dtype = np.uint8
        _min, _max = 0, _mask(weight_width)

    weight = fixed_rng.integers(_min, _max, size=size, dtype=_dtype)
    # Sparse
    sparse_ratio = 0.4
    num_zeros = int(weight.size * sparse_ratio)
    if num_zeros > 0:
        flat_indices = fixed_rng.choice(weight.size, size=num_zeros, replace=False)
        weight.flat[flat_indices] = 0

    mapped = weight_csc_pack(weight, weight_width)
    assert mapped.dtype == FRAME_DTYPE

    N_NONZERO_WEIGHT_PER_ADDR = {1: 7, 2: 7, 4: 6, 8: 5}
    align_size = N_NONZERO_WEIGHT_PER_ADDR[weight_width]
    # Only store non-zero weights
    expected_size = math.ceil(np.count_nonzero(weight) / align_size) * 2
    assert mapped.shape == (expected_size,)

    unmapped = weight_csc_unpack(mapped, weight_width, signed, size)
    assert unmapped.dtype == _dtype
    assert unmapped.size == weight.size
    assert np.array_equal(weight, unmapped)


@pytest.mark.parametrize(
    "size, sparse, expectation",
    [
        (64, False, pytest.raises(ValueError)),
        (64, True, contextlib.nullcontext()),
        (65, False, contextlib.nullcontext()),
    ],
    ids=["dense_need_align", "sparse_need_align", "dense_no_align"],
)
def test_weightcsc_unpack_check_aligned(size, sparse, expectation):
    weight = np.ones(size, dtype=np.int8)
    if sparse:
        weight[5] = 0  # As long as there is a zero, we can pad

    with expectation:
        _ = weight_csc_pack(weight, 8)
