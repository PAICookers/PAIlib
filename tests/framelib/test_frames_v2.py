import numpy as np
import pytest

from paicorelib.coordinate import CoordZXYOffset
from paicorelib.framelib.frame_defs import FFV2
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import OfflineConfigFrame2FormatV2 as Off_Cfg2_V2
from paicorelib.framelib.frame_gen_v2 import OfflineFrameGenV2
from paicorelib.framelib.types import (
    LUT_ACTIVATION_DTYPE,
    LUT_POTENTIAL_DTYPE,
    FrameArrayType,
)
from paicorelib.framelib.utils import single_frame_header_check
from paicorelib.routing_hexa import AERPacketZXYCopy
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

    # def test_cf1(self):
    #     # Initialize full dict to avoid KeyError due to strict checks
    #     keys = [
    #         # Word 1
    #         "snn_ann",
    #         "max_pooling",
    #         "add_potential",
    #         "zero_output",
    #         "input_sign",
    #         "input_width",
    #         "output_sign",
    #         "output_width",
    #         "weight_sign",
    #         "weight_width",
    #         "lcn",
    #         "target_lcn",
    #         "axon_skew",
    #         "neuron_number",
    #         "test_core_xy",
    #         "test_core_x",
    #         "test_core_y",
    #         # Word 2
    #         "global_send",
    #         "csc_accelerate",
    #         "global_receive",
    #         "thread_number",
    #         "busy_cycle",
    #         "delay_cycle",
    #         "width_cycle",
    #         # Word 3
    #         "tick_start",
    #         "tick_duration",
    #         "tick_initial",
    #     ]
    #     core_reg = {k: 0 for k in keys}

    #     # Set specific test values
    #     core_reg["snn_ann"] = 1  # Word1 bit
    #     core_reg["width_cycle"] = 0xAA  # Word2 bit
    #     core_reg["tick_start"] = 0x1234  # Word3 bit
    #     core_reg["tick_initial"] = 0x5678  # Word3 bit

    #     frames = OfflineFrameGenV2.gen_config_frame1(
    #         CoordZXYOffset(1, 1, 1), AERPacketZXYCopy(0, 1, -1), 0, core_reg
    #     )

    #     assert frames.size == 4

    #     core_reg_valid = OfflineCoreRegV2.model_validate(core_reg)
    #     # cf2 = OfflineFrameGenV2.gen_config_frame2(
    #     #     CoordZXYOffset(1, 1, 1), AERPacketZXYCopy(0, 1, -1), 0, core_reg_valid
    #     # )

    @pytest.mark.parametrize("act_dtype", [np.uint8, np.int8])
    def test_cf2(self, act_dtype):
        # arr_pot = np.arange(-128, 128, dtype=LUT_POTENTIAL_DTYPE)
        # if act_dtype == np.uint8:
        #     arr_act = np.ones((256,), dtype=act_dtype)
        # else:
        #     arr_act = np.full((256,), -1, dtype=act_dtype)
        arr_pot = gen_random_array((256,), dtype=LUT_POTENTIAL_DTYPE)
        arr_act = gen_random_array((256,), dtype=act_dtype)

        frames = OfflineFrameGenV2.gen_config_frame2(
            CoordZXYOffset(1, 1, 1), AERPacketZXYCopy(0, 1, -1), arr_pot, arr_act
        )

        assert frames.size == 256 + 1

        arr_pot2, arr_act2 = extract_lut_from_cf2(frames, act_dtype)
        assert np.array_equal(arr_pot, arr_pot2)
        assert np.array_equal(arr_act, arr_act2)

    # def test_gen_neuron_config_full(self):
    #     """Test Type 3: Full Neuron."""
    #     # 1. Prepare Common Attributes
    #     common_keys = [
    #         "weight_skew_low",
    #         "weight_addr_start",
    #         "weight_addr_end",
    #         "output_type",
    #         "fold_type",
    #         "neuron_type",  # Word 2
    #         "reset_mode",
    #         "reset_v",
    #         "thres_neg_mode",
    #         "thres_pos_mode",
    #         "thres_neg",
    #         "thres_pos_hi",  # Word 3
    #         "thres_pos_low",
    #         "lateral_inhibit",
    #         "leak_multi_seq",
    #         "leak_multi_in",
    #         "leak_multi_mode",
    #         "leak_add_mode",
    #         "leak_tau",
    #         "leak_v",
    #         "weight_compress",
    #         "vjt_initial",  # Word 4
    #     ]
    #     common_attrs = {k: 0 for k in common_keys}

    #     common_attrs["reset_mode"] = 2
    #     common_attrs["vjt_initial"] = 0x123

    #     # 2. Prepare Specific Targets
    #     target_keys = [
    #         "tick_relative",
    #         "addr_axon",
    #         "addr_core_xy",
    #         "addr_core_x",
    #         "addr_core_y",
    #         "addr_copy_xy",
    #         "addr_copy_x",
    #         "addr_copy_y",
    #         "weight_skew_high",
    #         "vjt",  # vjt in Word 2
    #     ]
    #     target_1 = {k: 0 for k in target_keys}
    #     target_1["tick_relative"] = 0xAB
    #     target_1["vjt"] = 0xDEADBEEF

    #     specific_targets = [target_1]

    #     frames = OfflineFrameGenV2.gen_neuron_config(
    #         0, 0, common_attrs, specific_targets, start_addr=0, mode="full"
    #     )

    #     # Expect: 1 Header + 4 Body Frames
    #     assert frames.size == 5

    #     w1, w2, w3, w4 = frames[1], frames[2], frames[3], frames[4]

    #     # Word 1: TICK_RELATIVE
    #     assert (w1 >> 56) & 0xFF == 0xAB

    #     # Word 2: VJT (Specific)
    #     assert (w2 & 0xFFFFFFFF) == 0xDEADBEEF

    #     # Word 3: RESET_MODE (Common)
    #     assert (w3 >> 62) & 0x3 == 2

    #     # Word 4: VJT_INITIAL (Common)
    #     assert (w4 & 0xFFF) == 0x123

    # def test_gen_neuron_config_fold(self):
    #     """Test Type 3: Fold Neuron."""
    #     common_keys = [
    #         "fold_skew_y_low",
    #         "fold_number",  # Word 2
    #         "fold_vjt_3",
    #         "fold_vjt_2",  # Word 3
    #         "fold_vjt_1",
    #         "fold_vjt_0",  # Word 4
    #     ]
    #     common_attrs = {k: 0 for k in common_keys}
    #     common_attrs["fold_vjt_0"] = 0x1234

    #     target_keys = [
    #         "fold_range_xy",
    #         "fold_range_x",
    #         "fold_range_y",
    #         "fold_skew_xy",
    #         "fold_skew_x",
    #         "fold_skew_y_hi",  # Word 1
    #         "fold_axon_xy",
    #         "fold_axon_x",
    #         "fold_axon_y",  # Word 2
    #     ]
    #     target_1 = {k: 0 for k in target_keys}
    #     target_1["fold_range_xy"] = 0x1F

    #     frames = OfflineFrameGenV2.gen_neuron_config(
    #         0, 0, common_attrs, [target_1], mode="fold"
    #     )

    #     assert frames.size == 5
    #     w1, w2, w3, w4 = frames[1], frames[2], frames[3], frames[4]

    #     # Word 1: FOLD_RANGE_XY
    #     assert (w1 >> 53) & 0x7FF == 0x1F

    #     # Word 4: FOLD_VJT_0
    #     assert (w4 & 0xFFFFFFFF) == 0x1234

    # def test_gen_test_request(self):
    #     """Test Test Request Frame Generation."""
    #     frames = OfflineFrameGenV2.gen_test_request(
    #         0x10, 0x20, int(FH.TEST_TYPE1), start_addr=0x5, num_packets=10
    #     )

    #     assert frames.size == 1

    #     hd, c, cp, _ = self._parse_frame(frames[0])
    #     assert hd == FH.TEST_TYPE1

    #     pkg_type, start, num = self._parse_packet_header(frames[0])
    #     assert pkg_type == 1  # TESTIN
    #     assert start == 0x5
    #     assert num == 10
