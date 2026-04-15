import contextlib
import math
from typing import Literal, cast

import numpy as np
import pytest

from paicorelib.core_defs import LCN_EX
from paicorelib.core_defs_v2 import CSCAccelerateMode, DataSign, DataWidth
from paicorelib.core_model_v2 import OfflineCoreRegV2
from paicorelib.framelib import FRAME_DTYPE
from paicorelib.framelib.frame_defs import FFV2, FramePackageType
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import (
    OfflineConfigFrame1FormatV2 as Off_Cfg1_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineConfigFrame2FormatV2 as Off_Cfg2_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineConfigFrame3FormatV2 as Off_Cfg3_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineControlFrame1FormatV2 as Off_Ctrl1_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineControlFrame3FormatV2 as Off_Ctrl3_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineWorkFrame1FormatV2 as Off_Work1_V2,
)
from paicorelib.framelib.frame_gen_v2 import (
    DataWidthLE8Like,
    FrameGenV2,
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
from paicorelib.framelib.utils import single_frame_header_check
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
)
from paicorelib.utils import _mask
from tests.utils import (
    bit_field,
    build_v2_core_reg_params,
    build_v2_dest_info_params,
    build_v2_folded_attrs_part1_params,
    build_v2_folded_attrs_part2_params,
    build_v2_full_attrs_part2_params,
    build_v2_half_attrs_params,
    build_v2_weight_array,
    gen_random_array,
)

WidthBitsLE8 = Literal[1, 2, 4, 8]


def parse_package_header(frames: FrameArrayType) -> tuple[int, int, int]:
    payload = bit_field(
        frames[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK
    )
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
    assert cf2.size == 257
    assert single_frame_header_check(cf2[0], FH.CONFIG_TYPE2)

    lut = cf2[1:]
    arr_pot = (
        (lut >> Off_Cfg2_V2.POTENTIAL_OFFSET) & Off_Cfg2_V2.POTENTIAL_MASK
    ).astype(LUT_POTENTIAL_DTYPE)
    arr_act = (
        (lut >> Off_Cfg2_V2.ACTIVATION_OFFSET) & Off_Cfg2_V2.ACTIVATION_MASK
    ).astype(act_dtype)
    return arr_pot, arr_act


def normalize_width_bits(value: DataWidthLE8Like) -> WidthBitsLE8:
    width_bits = (1 << value.value) if isinstance(value, DataWidth) else int(value)
    return cast(WidthBitsLE8, width_bits)


def validated_dump(model_cls, params):
    return model_cls.model_validate(params, strict=True).model_dump()


class TestFrameGenV2:
    def test_make_package(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        packages = np.array([11, 22, 33], dtype=FRAME_DTYPE)

        frames = FrameGenV2.make_package(
            FH.CONFIG_TYPE2, pkt_offset, pkt_ncopy, 7, packages
        )

        assert frames.dtype == FRAME_DTYPE
        assert frames.shape == (4,)
        assert single_frame_header_check(frames[0], FH.CONFIG_TYPE2)
        assert parse_package_header(frames) == (
            FramePackageType.CONF_TESTOUT.value,
            7,
            3,
        )
        assert np.array_equal(frames[1:], packages)


class TestOfflineFrameGenV2:
    def test_cf1(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        core_reg = OfflineCoreRegV2.model_validate(
            build_v2_core_reg_params(
                name="core_reg",
                input_width=DataWidth.WIDTH_8BIT,
                output_width=DataWidth.WIDTH_8BIT,
                weight_sign=DataSign.SIGNED,
                weight_width=DataWidth.WIDTH_8BIT,
                lcn=LCN_EX.LCN_8X,
                target_lcn=LCN_EX.LCN_8X,
                neuron_number=200,
                test_core_xy=5,
                global_send=0b100_0000,
                csc_accelerate=CSCAccelerateMode.ENABLE,
                global_receive=0b001_0000,
                busy_cycle=2,
                tick_start=1,
            ),
            strict=True,
        )

        frames = OfflineFrameGenV2.gen_config_frame1(pkt_offset, core_reg, pkt_ncopy, 0)

        assert frames.dtype == FRAME_DTYPE
        assert frames.size == 4
        assert parse_package_header(frames) == (
            FramePackageType.CONF_TESTOUT.value,
            0,
            3,
        )
        assert (
            bit_field(
                frames[1],
                Off_Cfg1_V2.Word1.TARGET_LCN_OFFSET,
                Off_Cfg1_V2.Word1.TARGET_LCN_MASK,
            )
            == LCN_EX.LCN_8X.value
        )
        assert (
            bit_field(
                frames[2],
                Off_Cfg1_V2.Word2.CSC_ACCELERATE_OFFSET,
                Off_Cfg1_V2.Word2.CSC_ACCELERATE_MASK,
            )
            == CSCAccelerateMode.ENABLE.value
        )
        assert (
            bit_field(
                frames[3],
                Off_Cfg1_V2.Word3.TICK_START_OFFSET,
                Off_Cfg1_V2.Word3.TICK_START_MASK,
            )
            == 1
        )

    @pytest.mark.parametrize("use_random_lut", [True, False])
    @pytest.mark.parametrize("act_dtype", [np.uint8, np.int8])
    def test_cf2(self, v2_packet_route, use_random_lut, act_dtype):
        pkt_offset, _ = v2_packet_route
        if use_random_lut:
            arr_pot = gen_random_array((256,), dtype=LUT_POTENTIAL_DTYPE)
            arr_act = gen_random_array((256,), dtype=act_dtype)
        else:
            arr_pot = np.arange(-128, 128, dtype=LUT_POTENTIAL_DTYPE)
            arr_act = (
                np.ones((256,), dtype=act_dtype)
                if act_dtype == np.uint8
                else np.full((256,), -1, dtype=act_dtype)
            )

        frames = OfflineFrameGenV2.gen_config_frame2(pkt_offset, arr_pot, arr_act)

        assert frames.size == 257

        arr_pot2, arr_act2 = extract_lut_from_cf2(frames, act_dtype)
        assert np.array_equal(arr_pot, arr_pot2)
        assert np.array_equal(arr_act, arr_act2)

    def test_cf2_rejects_mismatched_lut_size(self, v2_packet_route):
        pkt_offset, _ = v2_packet_route
        arr_pot = np.arange(255, dtype=LUT_POTENTIAL_DTYPE)
        arr_act = np.arange(256, dtype=np.uint8)

        with pytest.raises(ValueError, match="same size"):
            OfflineFrameGenV2.gen_config_frame2(pkt_offset, arr_pot, arr_act)

    @pytest.mark.parametrize("n_package", [0, 1, 100, 1000, 8000])
    def test_gen_config_frame3_pkg_header(self, v2_packet_route, n_package):
        pkt_offset, _ = v2_packet_route
        frames = OfflineFrameGenV2.gen_config_frame3_pkg_header(
            pkt_offset, 12, n_package
        )

        assert frames.dtype == FRAME_DTYPE
        assert frames.shape == (1,)
        assert single_frame_header_check(frames[0], FH.CONFIG_TYPE3)
        assert parse_package_header(frames) == (
            FramePackageType.CONF_TESTOUT.value,
            12,
            n_package,
        )

    def test_gen_pkg_half_neu_fields(self):
        dest_info = build_v2_dest_info_params(
            tick_relative=7,
            addr_axon=0x1AB,
            addr_core_xy=-1,
            addr_core_x=2,
            addr_core_y=-3,
            addr_copy_xy=4,
            addr_copy_x=5,
            addr_copy_y=-6,
        )
        half_attrs = build_v2_half_attrs_params(
            weight_skew=0x6A5,
            weight_address_start=0x123,
            weight_address_end=0x456,
            output_type=OutputType.POTENTIAL,
            fold_type=FoldType.FOLDED,
            neuron_type=NeuronType.FULL,
            vjt=0x12345678,
        )

        dest_info_dump = validated_dump(OfflineNeuDestInfoV2, dest_info)
        half_attrs_dump = validated_dump(OfflineNeuFullAttrsV2Part1, half_attrs)

        pkg_half_neu = OfflineFrameGenV2._gen_pkg_half_neu(
            dest_info_dump, half_attrs_dump
        )

        assert pkg_half_neu.dtype == FRAME_DTYPE
        assert pkg_half_neu.shape == (2,)
        assert bit_field(
            pkg_half_neu[0],
            Off_Cfg3_V2.Full.Word1.WEIGHT_SKEW_LOW5_OFFSET,
            Off_Cfg3_V2.Full.Word1.WEIGHT_SKEW_LOW5_MASK,
        ) == (half_attrs["weight_skew"] & Off_Cfg3_V2.Full.Word1.WEIGHT_SKEW_LOW5_MASK)
        assert bit_field(
            pkg_half_neu[1],
            Off_Cfg3_V2.Full.Word2.WEIGHT_SKEW_HIGH11_OFFSET,
            Off_Cfg3_V2.Full.Word2.WEIGHT_SKEW_HIGH11_MASK,
        ) == (half_attrs["weight_skew"] >> 5)
        assert (
            bit_field(
                pkg_half_neu[0],
                Off_Cfg3_V2.Full.Word1.WEIGHT_ADDRESS_START_OFFSET,
                Off_Cfg3_V2.Full.Word1.WEIGHT_ADDRESS_START_MASK,
            )
            == half_attrs["weight_address_start"]
        )
        assert (
            bit_field(
                pkg_half_neu[0],
                Off_Cfg3_V2.Full.Word1.WEIGHT_ADDRESS_END_OFFSET,
                Off_Cfg3_V2.Full.Word1.WEIGHT_ADDRESS_END_MASK,
            )
            == half_attrs["weight_address_end"]
        )
        assert bit_field(
            pkg_half_neu[1],
            Off_Cfg3_V2.Full.Word2.ADDR_CORE_Y_OFFSET,
            Off_Cfg3_V2.Full.Word2.ADDR_CORE_Y_MASK,
        ) == (dest_info["addr_core_y"] & Off_Cfg3_V2.Full.Word2.ADDR_CORE_Y_MASK)

    def test_gen_pkg_full_neu_fields(self):
        dest_info = build_v2_dest_info_params(
            tick_relative=9,
            addr_axon=17,
            addr_core_xy=1,
            addr_core_x=2,
            addr_core_y=3,
            addr_copy_xy=4,
            addr_copy_x=5,
            addr_copy_y=6,
        )
        full_attrs1 = build_v2_half_attrs_params(
            weight_skew=0x123,
            weight_address_start=10,
            weight_address_end=20,
            output_type=OutputType.POTENTIAL,
            fold_type=FoldType.UNFOLDED,
            neuron_type=NeuronType.FULL,
            vjt=0x2468ACE,
        )
        full_attrs2 = build_v2_full_attrs_part2_params(
            reset_mode=ResetMode.MODE_LINEAR,
            reset_v=-123,
            threshold_neg_mode=ThresholdNegMode.FLOOR,
            threshold_pos_mode=ThresholdPosMode.CEILING,
            threshold_neg=-0x12345,
            threshold_pos=0x12345678,
            lateral_inhibition=LateralInhibitionMode.ENABLE,
            leak_multi_sequence=LeakMultiComparisonOrder.AFTER_COMPARE,
            leak_multi_input=LeakMultiInputMode.ENABLE,
            leak_multi_mode=LeakMultiMode.ENABLE,
            leak_add_mode=LeakAddMode.BACKWARD,
            leak_tau=17,
            leak_v=-0x1234,
            weight_compress=WeightCompressType.SPARSE,
            vjt_initial=0x345,
        )

        dest_info_dump = validated_dump(OfflineNeuDestInfoV2, dest_info)
        full_attrs1_dump = validated_dump(OfflineNeuFullAttrsV2Part1, full_attrs1)
        full_attrs2_dump = validated_dump(OfflineNeuFullAttrsV2Part2, full_attrs2)

        pkg_full_neu = OfflineFrameGenV2._gen_pkg_full_neu(
            dest_info_dump, full_attrs1_dump, full_attrs2_dump
        )

        threshold_pos = (
            bit_field(
                pkg_full_neu[3],
                Off_Cfg3_V2.Full.Word4.THRESHOLD_POS_HIGH12_OFFSET,
                Off_Cfg3_V2.Full.Word4.THRESHOLD_POS_HIGH12_MASK,
            )
            << 20
        ) | bit_field(
            pkg_full_neu[2],
            Off_Cfg3_V2.Full.Word3.THRESHOLD_POS_LOW20_OFFSET,
            Off_Cfg3_V2.Full.Word3.THRESHOLD_POS_LOW20_MASK,
        )

        assert pkg_full_neu.dtype == FRAME_DTYPE
        assert pkg_full_neu.shape == (4,)
        assert threshold_pos == full_attrs2["threshold_pos"]
        assert bit_field(
            pkg_full_neu[3],
            Off_Cfg3_V2.Full.Word4.THRESHOLD_NEG_OFFSET,
            Off_Cfg3_V2.Full.Word4.THRESHOLD_NEG_MASK,
        ) == (full_attrs2["threshold_neg"] & Off_Cfg3_V2.Full.Word4.THRESHOLD_NEG_MASK)
        assert bit_field(
            pkg_full_neu[3],
            Off_Cfg3_V2.Full.Word4.RESET_V_OFFSET,
            Off_Cfg3_V2.Full.Word4.RESET_V_MASK,
        ) == (full_attrs2["reset_v"] & Off_Cfg3_V2.Full.Word4.RESET_V_MASK)
        assert bit_field(
            pkg_full_neu[2],
            Off_Cfg3_V2.Full.Word3.LEAK_V_OFFSET,
            Off_Cfg3_V2.Full.Word3.LEAK_V_MASK,
        ) == (full_attrs2["leak_v"] & Off_Cfg3_V2.Full.Word3.LEAK_V_MASK)
        assert (
            bit_field(
                pkg_full_neu[2],
                Off_Cfg3_V2.Full.Word3.WEIGHT_COMPRESS_OFFSET,
                Off_Cfg3_V2.Full.Word3.WEIGHT_COMPRESS_MASK,
            )
            == WeightCompressType.SPARSE.value
        )

    def test_gen_pkg_folded_neu_fields(self):
        attrs1 = build_v2_folded_attrs_part1_params(
            fold_range_xy=2,
            fold_range_x=3,
            fold_range_y=4,
            fold_skew_xy=5,
            fold_skew_x=6,
            fold_skew_y=0x155,
            fold_axon_xy=7,
            fold_axon_x=8,
            fold_axon_y=9,
            fold_number=24,
        )
        attrs2_1 = build_v2_folded_attrs_part2_params(
            fold_vjt_3=100,
            fold_vjt_2=200,
            fold_vjt_1=300,
            fold_vjt_0=400,
        )
        attrs2_2 = build_v2_folded_attrs_part2_params(
            fold_vjt_3=10,
            fold_vjt_2=20,
            fold_vjt_1=30,
            fold_vjt_0=40,
        )

        attrs1_dump = validated_dump(OfflineNeuFoldedAttrsV2Part1, attrs1)
        attrs2_1_dump = validated_dump(OfflineNeuFoldedAttrsV2Part2, attrs2_1)
        attrs2_2_dump = validated_dump(OfflineNeuFoldedAttrsV2Part2, attrs2_2)

        pkg_folded_neu = OfflineFrameGenV2._gen_pkg_folded_neu(
            attrs1_dump, attrs2_1_dump, attrs2_2_dump
        )

        assert pkg_folded_neu.dtype == FRAME_DTYPE
        assert pkg_folded_neu.shape == (6,)
        assert bit_field(
            pkg_folded_neu[0],
            Off_Cfg3_V2.Fold.Word1.FOLD_SKEW_Y_LOW2_OFFSET,
            Off_Cfg3_V2.Fold.Word1.FOLD_SKEW_Y_LOW2_MASK,
        ) == (attrs1["fold_skew_y"] & Off_Cfg3_V2.Fold.Word1.FOLD_SKEW_Y_LOW2_MASK)
        assert bit_field(
            pkg_folded_neu[1],
            Off_Cfg3_V2.Fold.Word2.FOLD_SKEW_Y_HIGH9_OFFSET,
            Off_Cfg3_V2.Fold.Word2.FOLD_SKEW_Y_HIGH9_MASK,
        ) == (attrs1["fold_skew_y"] >> 2)
        assert (
            bit_field(
                pkg_folded_neu[2],
                Off_Cfg3_V2.Fold.Word3.FOLD_VJT_0_OFFSET,
                Off_Cfg3_V2.Fold.Word3.FOLD_VJT_0_MASK,
            )
            == attrs2_1["fold_vjt_0"]
        )
        assert (
            bit_field(
                pkg_folded_neu[3],
                Off_Cfg3_V2.Fold.Word4.FOLD_VJT_3_OFFSET,
                Off_Cfg3_V2.Fold.Word4.FOLD_VJT_3_MASK,
            )
            == attrs2_1["fold_vjt_3"]
        )
        assert (
            bit_field(
                pkg_folded_neu[4],
                Off_Cfg3_V2.Fold.Word3.FOLD_VJT_1_OFFSET,
                Off_Cfg3_V2.Fold.Word3.FOLD_VJT_1_MASK,
            )
            == attrs2_2["fold_vjt_1"]
        )
        assert (
            bit_field(
                pkg_folded_neu[5],
                Off_Cfg3_V2.Fold.Word4.FOLD_VJT_2_OFFSET,
                Off_Cfg3_V2.Fold.Word4.FOLD_VJT_2_MASK,
            )
            == attrs2_2["fold_vjt_2"]
        )

    @pytest.mark.parametrize(
        "wrapper_name,args,expected_size",
        [
            (
                "gen_config_frame3_pkg_half",
                (
                    build_v2_dest_info_params(),
                    build_v2_half_attrs_params(),
                ),
                2,
            ),
            (
                "gen_config_frame3_pkg_full",
                (
                    build_v2_dest_info_params(),
                    build_v2_half_attrs_params(),
                    build_v2_full_attrs_part2_params(),
                ),
                4,
            ),
            (
                "gen_config_frame3_pkg_folded",
                (
                    build_v2_folded_attrs_part1_params(),
                    [build_v2_folded_attrs_part2_params()],
                ),
                4,
            ),
        ],
        ids=["half", "full", "folded"],
    )
    def test_gen_config_frame3_pkg_wrappers(self, wrapper_name, args, expected_size):
        result = getattr(OfflineFrameGenV2, wrapper_name)(*args)

        assert result.size == expected_size

    def test_gen_config_frame3_pkg_folded_wrapper_rejects_incomplete_attrs(self):
        with pytest.raises(ValueError, match="incomplete"):
            OfflineFrameGenV2.gen_config_frame3_pkg_folded(
                build_v2_folded_attrs_part1_params(), []
            )

    @pytest.mark.parametrize(
        "kwargs,expected_sizes",
        [
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": build_v2_half_attrs_params(),
                    "full_attrs2": build_v2_full_attrs_part2_params(),
                    "folded_attrs1": build_v2_folded_attrs_part1_params(),
                    "folded_attrs2_": [
                        build_v2_folded_attrs_part2_params(),
                        build_v2_folded_attrs_part2_params(fold_vjt_0=10),
                    ],
                },
                [2, 4, 6],
            ),
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": None,
                    "full_attrs2": None,
                    "folded_attrs1": build_v2_folded_attrs_part1_params(),
                    "folded_attrs2_": [build_v2_folded_attrs_part2_params()],
                },
                [0, 0, 4],
            ),
        ],
        ids=["all", "folded-only"],
    )
    def test_gen_config_frame3_pkg_neu_valid_combinations(self, kwargs, expected_sizes):
        result = OfflineFrameGenV2.gen_config_frame3_pkg_neu(**kwargs)

        assert [pkg.size for pkg in result] == expected_sizes

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": None,
                    "full_attrs2": build_v2_full_attrs_part2_params(),
                    "folded_attrs1": None,
                    "folded_attrs2_": [],
                },
                "full neuron.*missing part1",
            ),
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": build_v2_half_attrs_params(),
                    "full_attrs2": None,
                    "folded_attrs1": build_v2_folded_attrs_part1_params(),
                    "folded_attrs2_": [],
                },
                "folded neuron.*missing part2",
            ),
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": build_v2_half_attrs_params(),
                    "full_attrs2": None,
                    "folded_attrs1": None,
                    "folded_attrs2_": [build_v2_folded_attrs_part2_params()],
                },
                "folded neuron.*missing part1",
            ),
        ],
        ids=["full-missing-part1", "folded-missing-part2", "folded-missing-part1"],
    )
    def test_gen_config_frame3_pkg_neu_rejects_invalid_combinations(
        self, kwargs, error_match
    ):
        with pytest.raises(ValueError, match=error_match):
            OfflineFrameGenV2.gen_config_frame3_pkg_neu(**kwargs)

    @pytest.mark.parametrize(
        "weight_width, input_width, csc_compress",
        [
            (8, 8, False),
            (DataWidth.WIDTH_4BIT, DataWidth.WIDTH_2BIT, False),
            (DataWidth.WIDTH_8BIT, DataWidth.WIDTH_1BIT, CSCAccelerateMode.ENABLE),
        ],
    )
    def test_gen_config_frame3_weight_pkg(
        self,
        weight_width: DataWidthLE8Like,
        input_width: DataWidthLE8Like,
        csc_compress: bool | CSCAccelerateMode,
        fixed_rng: np.random.Generator,
    ):
        is_sparse = (
            csc_compress != CSCAccelerateMode.DISABLE and csc_compress is not False
        )
        width_bits = normalize_width_bits(weight_width)
        weight = build_v2_weight_array(
            32, width_bits, True, fixed_rng, sparse_ratio=0.4 if is_sparse else 0.0
        )

        result = OfflineFrameGenV2.gen_config_frame3_weight_pkg(
            weight, weight_width, input_width, csc_compress
        )

        if is_sparse:
            input_bits = normalize_width_bits(input_width)
            expected = weight_csc_pack(weight, width_bits, input_bits)
        else:
            expected = weight_dense_pack(weight, width_bits)

        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "weight_width, input_width",
        [
            (DataWidth.WIDTH_16BIT, DataWidth.WIDTH_1BIT),
            (DataWidth.WIDTH_1BIT, DataWidth.WIDTH_32BIT),
        ],
    )
    def test_gen_config_frame3_weight_pkg_rejects_unsupported_widths(
        self, weight_width, input_width
    ):
        with pytest.raises(ValueError, match="1/2/4/8-bit"):
            OfflineFrameGenV2.gen_config_frame3_weight_pkg(
                np.ones(8, dtype=np.int8), weight_width, input_width
            )

    @pytest.mark.parametrize(
        "ts, axon, data, target_lcn",
        [
            (
                np.array([0b0011_0100, 0b0011_0100]),
                np.array([0b0_0100_1011_0011, 0b100_1011_0011]),
                np.array([-16, 1], dtype=np.int8),
                LCN_EX.LCN_4X,
            ),
            (
                0b1011_0100,
                0b1_1011_0011,
                np.array([10], dtype=np.uint8),
                LCN_EX.LCN_1X,
            ),
        ],
    )
    def test_wf1(self, v2_packet_route, ts, axon, data, target_lcn):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf1 = OfflineFrameGenV2.gen_work_frame1(
            pkt_offset, pkt_ncopy, ts, axon, target_lcn, data
        )

        ts0 = ts[0] if isinstance(ts, np.ndarray) else ts
        axon0 = axon[0] if isinstance(axon, np.ndarray) else axon
        ts_width, ax_width = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[target_lcn.value]

        assert wf1.dtype == FRAME_DTYPE
        assert bit_field(
            wf1[0],
            Off_Work1_V2.TIMESTEP_HIGH7_OFFSET,
            Off_Work1_V2.TIMESTEP_HIGH7_MASK,
        ) == ((ts0 >> (ts_width - 1)) & Off_Work1_V2.TIMESTEP_HIGH7_MASK)
        assert (
            bit_field(wf1[0], Off_Work1_V2.AXON_ADDR_OFFSET, _mask(ax_width)) == axon0
        )

    def test_wf1_filters_zero_payloads(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf1 = OfflineFrameGenV2.gen_work_frame1(
            pkt_offset,
            pkt_ncopy,
            np.array([1, 2, 3, 4]),
            np.array([10, 11, 12, 13]),
            LCN_EX.LCN_1X,
            np.array([0, 5, 0, 6], dtype=np.uint8),
        )

        assert wf1.shape == (2,)
        assert bit_field(wf1[0], Off_Work1_V2.DATA_OFFSET, Off_Work1_V2.DATA_MASK) == 5
        assert bit_field(wf1[1], Off_Work1_V2.DATA_OFFSET, Off_Work1_V2.DATA_MASK) == 6

    @pytest.mark.parametrize(
        "timesteps, axons, data",
        [
            ([1, 2], [3], [4, 5]),
            ([1, 2], [3, 4], [5]),
        ],
    )
    def test_wf1_rejects_mismatched_lengths(
        self, v2_packet_route, timesteps, axons, data
    ):
        pkt_offset, pkt_ncopy = v2_packet_route

        with pytest.raises(ValueError, match="size"):
            OfflineFrameGenV2.gen_work_frame1(
                pkt_offset, pkt_ncopy, timesteps, axons, LCN_EX.LCN_1X, data
            )

    def test_control_frames(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        ctrl1 = OfflineFrameGenV2.gen_control_frame1(pkt_offset, pkt_ncopy, 123)
        ctrl2 = OfflineFrameGenV2.gen_control_frame2(pkt_offset, pkt_ncopy)
        complete = OfflineFrameGenV2.gen_complete_frame(pkt_offset, pkt_ncopy, 7)

        assert single_frame_header_check(ctrl1[0], FH.CTRL_TYPE1)
        assert single_frame_header_check(ctrl2[0], FH.CTRL_TYPE2)
        assert single_frame_header_check(complete[0], FH.CTRL_TYPE3)
        assert (
            bit_field(ctrl1[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK)
            == 123
        )
        assert (
            bit_field(
                complete[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK
            )
            == 7
        )

    def test_control_frame1_rejects_overflow(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.raises(ValueError, match="overflow"):
            OfflineFrameGenV2.gen_control_frame1(
                pkt_offset, pkt_ncopy, Off_Ctrl1_V2.NUM_TIMESTEP_MASK + 1
            )

    def test_complete_frame_rejects_overflow(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.raises(ValueError, match="thread_id"):
            OfflineFrameGenV2.gen_complete_frame(
                pkt_offset, pkt_ncopy, Off_Ctrl3_V2.THREAD_ID_MASK + 1
            )


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
    weight = build_v2_weight_array(size, weight_width, signed, fixed_rng)
    mapped = weight_dense_pack(weight, weight_width)

    align_size = 128 // weight_width
    expected_size = math.ceil(weight.size / align_size) * (
        128 // (FRAME_DTYPE(0).nbytes * 8)
    )

    assert mapped.dtype == FRAME_DTYPE
    assert mapped.shape == (expected_size,)

    unmapped = weight_dense_unpack(mapped, weight_width, signed, size)
    assert unmapped.dtype == weight.dtype
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
    weight = build_v2_weight_array(
        size, weight_width, signed, fixed_rng, sparse_ratio=0.4
    )
    mapped = weight_csc_pack(weight, weight_width, input_width=1)

    n_nonzero_per_addr = {1: 7, 2: 7, 4: 6, 8: 5}[weight_width]
    expected_size = math.ceil(np.count_nonzero(weight) / n_nonzero_per_addr) * 2

    assert mapped.dtype == FRAME_DTYPE
    assert mapped.shape == (expected_size,)

    unmapped = weight_csc_unpack(mapped, weight_width, signed, size)
    assert unmapped.dtype == weight.dtype
    assert np.array_equal(weight, unmapped)


def test_weight_csc_unpack_rejects_odd_frame_count():
    with pytest.raises(ValueError, match="even"):
        weight_csc_unpack(np.array([1], dtype=FRAME_DTYPE), 8, True, 1)


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
        weight[5] = 0

    with expectation:
        weight_csc_pack(weight, 8, 8)
