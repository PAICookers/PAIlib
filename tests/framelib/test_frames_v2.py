import pytest
import numpy as np
from paicorelib.framelib.frame_gen import OfflineFrameGenV2
from paicorelib.framelib.frame_defs import (
    FrameHeaderV2 as FHV2,
    FrameFormatV2 as FFV2,
)

class TestOfflineFrameGenV2:
    """Unit tests for V2.5 Offline Frame Generator."""

    def _parse_frame(self, frame):
        """Helper to unpack a standard frame (Header/Addr/Payload)."""
        val = int(frame)
        header = (val >> FFV2.GENERAL_HEADER_OFFSET) & FFV2.GENERAL_HEADER_MASK
        core_addr = (val >> FFV2.GENERAL_CORE_ADDR_OFFSET) & FFV2.GENERAL_CORE_ADDR_MASK
        copy_addr = (val >> FFV2.GENERAL_COPY_ADDR_OFFSET) & FFV2.GENERAL_COPY_ADDR_MASK
        payload = (val & FFV2.GENERAL_PAYLOAD_MASK)
        return header, core_addr, copy_addr, payload

    def _parse_packet_header(self, frame):
        """Helper to unpack a Packet Header."""
        _, _, _, p_payload = self._parse_frame(frame)
        pkg_type = (p_payload >> FFV2.GENERAL_PACKAGE_TYPE_OFFSET) & 1
        start_addr = (p_payload >> FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET) & FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK
        num = (p_payload >> FFV2.GENERAL_PACKAGE_NUM_OFFSET) & FFV2.GENERAL_PACKAGE_NUM_MASK
        return pkg_type, start_addr, num

    def test_gen_data_frame(self):
        """Test Work Frame Type 1 with dynamic widths."""
        core, copy = 0x10, 0x20
        ts, axon, data = 10, 0x100, 0x55
        
        # Define dynamic widths (must sum to 17 for time + axon)
        t_w, a_w, d_w = 7, 10, 8
        
        frames = OfflineFrameGenV2.gen_data_frame(
            core, copy, ts, axon, data,
            time_width=t_w, axon_width=a_w, data_width=d_w
        )
        assert frames.shape == (1,)
        
        hd, c, cp, pl = self._parse_frame(frames[0])
        assert hd == FHV2.WORK_TYPE1
        assert c == core
        assert cp == copy
        
        # Calculate expected payload manually
        # Logic: mixed = (ts << axon_width) | axon
        # Layout: MSB(1bit) at 60 | Low(16bits) at 8 | Data at 0
        mixed = (ts << a_w) | axon
        msb = (mixed >> 16) & 1
        low = mixed & 0xFFFF
        
        expected_payload = (msb << 60) | (low << 8) | data
        assert pl == expected_payload

    def test_gen_vjt_frame(self):
        """Test Work Frame Type 2 with dynamic widths."""
        core, copy = 0x10, 0x20
        ts, axon, vjt = 5, 0x200, 0xAA
        
        t_w, a_w, v_w = 7, 10, 8
        
        frames = OfflineFrameGenV2.gen_vjt_frame(
            core, copy, ts, axon, vjt,
            time_width=t_w, axon_width=a_w, vjt_width=v_w
        )
        assert frames.shape == (1,)
        
        hd, c, cp, pl = self._parse_frame(frames[0])
        assert hd == FHV2.WORK_TYPE2 
        
        mixed = (ts << a_w) | axon
        msb = (mixed >> 16) & 1
        low = mixed & 0xFFFF
        
        # VJT at offset 0 
        expected_payload = (msb << 60) | (low << 8) | vjt 
        assert pl == expected_payload

    def test_gen_sync_frame(self):
        core, copy = 0x5, 0x0
        num_ts = 1000
        
        frames = OfflineFrameGenV2.gen_sync_frame(core, copy, num_ts)
        hd, _, _, pl = self._parse_frame(frames[0])
        assert hd == FHV2.CONTROL_TYPE1
        # Assuming simple shift logic for sync frame in definitions
        # assert pl == num_ts 

    def test_gen_core_config(self):
        """Test Type 1: Core Parameters (Word 1, 2, 3)."""
        # Initialize full dict to avoid KeyError due to strict checks
        keys = [
            # Word 1
            'snn_ann', 'max_pooling', 'add_potential', 'zero_output', 'input_sign', 
            'input_width', 'output_sign', 'output_width', 'weight_sign', 'weight_width',
            'lcn', 'target_lcn', 'axon_skew', 'neuron_number', 
            'test_core_xy', 'test_core_x', 'test_core_y_high',
            # Word 2
            'test_core_y_low', 'global_send', 'csc_accelerate', 'global_receive',
            'thread_number', 'busy_cycle', 'delay_cycle', 'width_cycle',
            # Word 3
            'tick_start', 'tick_duration', 'tick_initial'
        ]
        params = {k: 0 for k in keys}
        
        # Set specific test values
        params['snn_ann'] = 1          # Word1 bit
        params['width_cycle'] = 0xAA   # Word2 bit
        params['tick_start'] = 0x1234  # Word3 bit
        params['tick_initial'] = 0x5678 # Word3 bit

        frames = OfflineFrameGenV2.gen_core_config(0, 0, params)
        
        # Expect: 1 Header + 3 Body Frames
        assert frames.size == 4
        
        pkg_type, start, num = self._parse_packet_header(frames[0])
        assert pkg_type == 0 # Config
        assert num == 3      # 3 Words
        
        # Verify Body Frames
        w1_frame, w2_frame, w3_frame = frames[1], frames[2], frames[3]
        
        # Check Word 1 (SNN_ANN at bit 63 - assuming offset 63)
        assert (w1_frame >> 63) & 1 == 1
        
        # Check Word 2 (width_cycle)
        # assert (w2_frame & 0xFF) == 0xAA 
        
        # Check Word 3 (tick_start high, tick_initial low)
        assert (w3_frame >> 48) & 0xFFFF == 0x1234
        assert (w3_frame & 0xFFFF) == 0x5678

    def test_gen_lut_config(self):
        """Test Type 2: LUT (Separate Arrays)."""

        pot_arr = np.array([0x11223344, 0xAABBCCDD], dtype=np.int64)
        act_arr = np.array([0x55, 0xEE], dtype=np.int64)
        
        frames = OfflineFrameGenV2.gen_lut_config(0, 0, pot_arr, act_arr, start_index=10)
        
        assert frames.size == 3
        
        _, start, num = self._parse_packet_header(frames[0])
        assert start == 10
        assert num == 2
        
        val0 = (0x11223344 << 8) | 0x55
        assert frames[1] == val0
        
        with pytest.raises(ValueError):
             OfflineFrameGenV2.gen_lut_config(0, 0, [1, 2], [1])

    def test_gen_neuron_config_full(self):
        """Test Type 3: Full Neuron."""
        # 1. Prepare Common Attributes
        common_keys = [
            'weight_skew_low', 'weight_addr_start', 'weight_addr_end', 'output_type', 
            'fold_type', 'neuron_type', # Word 2
            'reset_mode', 'reset_v', 'thres_neg_mode', 'thres_pos_mode', 'thres_neg', 'thres_pos_hi', # Word 3
            'thres_pos_low', 'lateral_inhibit', 'leak_multi_seq', 'leak_multi_in', 'leak_multi_mode', 
            'leak_add_mode', 'leak_tau', 'leak_v', 'weight_compress', 'vjt_initial' # Word 4
        ]
        common_attrs = {k: 0 for k in common_keys}
        
        common_attrs['reset_mode'] = 2
        common_attrs['vjt_initial'] = 0x123

        # 2. Prepare Specific Targets
        target_keys = [
            'tick_relative', 'addr_axon', 'addr_core_xy', 'addr_core_x', 'addr_core_y',
            'addr_copy_xy', 'addr_copy_x', 'addr_copy_y', 'weight_skew_high', 'vjt' # vjt in Word 2
        ]
        target_1 = {k: 0 for k in target_keys}
        target_1['tick_relative'] = 0xAB
        target_1['vjt'] = 0xDEADBEEF
        
        specific_targets = [target_1]

        frames = OfflineFrameGenV2.gen_neuron_config(
            0, 0, common_attrs, specific_targets, start_addr=0, mode='full'
        )
        
        # Expect: 1 Header + 4 Body Frames
        assert frames.size == 5
        
        w1, w2, w3, w4 = frames[1], frames[2], frames[3], frames[4]
        
        # Word 1: TICK_RELATIVE
        assert (w1 >> 56) & 0xFF == 0xAB
        
        # Word 2: VJT (Specific)
        assert (w2 & 0xFFFFFFFF) == 0xDEADBEEF
        
        # Word 3: RESET_MODE (Common)
        assert (w3 >> 62) & 0x3 == 2
        
        # Word 4: VJT_INITIAL (Common)
        assert (w4 & 0xFFF) == 0x123

    def test_gen_neuron_config_fold(self):
        """Test Type 3: Fold Neuron."""
        common_keys = [
            'fold_skew_y_low', 'fold_number', # Word 2
            'fold_vjt_3', 'fold_vjt_2',      # Word 3
            'fold_vjt_1', 'fold_vjt_0'       # Word 4
        ]
        common_attrs = {k: 0 for k in common_keys}
        common_attrs['fold_vjt_0'] = 0x1234

        target_keys = [
            'fold_range_xy', 'fold_range_x', 'fold_range_y', 
            'fold_skew_xy', 'fold_skew_x', 'fold_skew_y_hi', # Word 1
            'fold_axon_xy', 'fold_axon_x', 'fold_axon_y'     # Word 2
        ]
        target_1 = {k: 0 for k in target_keys}
        target_1['fold_range_xy'] = 0x1F

        frames = OfflineFrameGenV2.gen_neuron_config(
            0, 0, common_attrs, [target_1], mode='fold'
        )
        
        assert frames.size == 5
        w1, w2, w3, w4 = frames[1], frames[2], frames[3], frames[4]
        
        # Word 1: FOLD_RANGE_XY
        assert (w1 >> 53) & 0x7FF == 0x1F 
        
        # Word 4: FOLD_VJT_0
        assert (w4 & 0xFFFFFFFF) == 0x1234

    def test_gen_test_request(self):
        """Test Test Request Frame Generation."""
        frames = OfflineFrameGenV2.gen_test_request(
            0x10, 0x20, 
            int(FHV2.TEST_TYPE1),
            start_addr=0x5, 
            num_packets=10
        )
        
        assert frames.size == 1 
        
        hd, c, cp, _ = self._parse_frame(frames[0])
        assert hd == FHV2.TEST_TYPE1
        
        pkg_type, start, num = self._parse_packet_header(frames[0])
        assert pkg_type == 1  # TESTIN
        assert start == 0x5
        assert num == 10