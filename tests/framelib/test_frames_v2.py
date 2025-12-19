import pytest
import numpy as np
from paicorelib.framelib.frame_gen import OfflineFrameGenV2
from paicorelib.framelib.frame_defs import (
    FrameHeaderV2 as FHV2,
    FrameFormatV2 as FFV2,
    OfflineConfigFrame1FormatV2,
    OfflineConfigFrame3FormatV2,
)

FRAME_DTYPE = np.uint64

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
        core, copy = 0x10, 0x20
        ts, axon, data = 10, 0x100, 0x55
        
        frames = OfflineFrameGenV2.gen_data_frame(core, copy, ts, axon, data)
        assert frames.shape == (1,)
        
        hd, c, cp, pl = self._parse_frame(frames[0])
        assert hd == FHV2.WORK_TYPE1
        assert c == core
        assert cp == copy
        # Check payload fields
        assert (pl >> 17) & 0x7F == ts
        assert (pl >> 8) & 0x1FF == axon
        assert (pl & 0xFF) == data

    def test_gen_sync_frame(self):
        core, copy = 0x5, 0x0
        num_ts = 1000
        
        frames = OfflineFrameGenV2.gen_sync_frame(core, copy, num_ts)
        hd, _, _, pl = self._parse_frame(frames[0])
        assert hd == FHV2.CONTROL_TYPE1
        assert pl == num_ts


    def test_gen_core_config(self):
        """Test Type 1: Core Parameters (192 bits -> 3 Words)"""
        params = {
            'snn_ann': 1,          # Word0 [63] (Global [191])
            'test_core_y_high': 3, # Word0 [1:0]
            'width_cycle': 0xAA,   # Word1 [7:0]
            'tick_start': 0x1234,  # Word2 [63:48] -> Local [48]
            'tick_initial': 0x5678 # Word2 [15:0]
        }
        
        frames = OfflineFrameGenV2.gen_core_config(0, 0, params)
        
        # Expect: 1 Header + 3 Body Frames
        assert frames.size == 4
        
        # 1. Verify Packet Header
        pkg_type, start, num = self._parse_packet_header(frames[0])
        assert pkg_type == 0 # Config
        assert start == 0    # Reserved for Core Type
        assert num == 3      # 3 Words
        
        # 2. Verify Body Frames (Raw 64-bit)
        w0, w1, w2 = frames[1], frames[2], frames[3]
        
        # Check Word 0 (SNN_ANN at bit 63)
        assert (w0 >> 63) & 1 == 1
        # Check Word 0 (test_core_y_high at bit 0)
        assert (w0 & 3) == 3
        
        # Check Word 1 (width_cycle at bit 0)
        assert (w1 & 0xFF) == 0xAA
        
        # Check Word 2 (tick_start at bit 48, tick_initial at bit 0)
        assert (w2 >> 48) & 0xFFFF == 0x1234
        assert (w2 & 0xFFFF) == 0x5678

    def test_gen_lut_config(self):
        """Test Type 2: LUT (32+8 bits)"""
        entries = [
            {'potential': 0x11223344, 'activation': 0x55},
            {'potential': 0xAABBCCDD, 'activation': 0xEE}
        ]
        
        frames = OfflineFrameGenV2.gen_lut_config(0, 0, entries, start_index=10)
        
        # Expect: 1 Header + 2 Body Frames
        assert frames.size == 3
        
        # Header Check
        _, start, num = self._parse_packet_header(frames[0])
        assert start == 10
        assert num == 2
        
        # Body Check
        # LUT Layout: [39:8] Potential, [7:0] Activation
        # In gen code: (pot << 8) | act
        val0 = (0x11223344 << 8) | 0x55
        assert frames[1] == val0

    def test_gen_neuron_config_full(self):
        """Test Type 3: Full Neuron (256 bits -> 4 Words)"""
        neurons = [{
            'tick_relative': 0xAB,   # Word0 [63:56]
            'vjt': 0xDEADBEEF,       # Word1 [31:0]
            'reset_mode': 2,         # Word2 [63:62]
            'vjt_initial': 0x123     # Word3 [11:0]
        }]
        
        frames = OfflineFrameGenV2.gen_neuron_config(0, 0, neurons, start_addr=0, mode='full')
        
        # Expect: 1 Header + 4 Body Frames
        assert frames.size == 5
        
        w0, w1, w2, w3 = frames[1], frames[2], frames[3], frames[4]
        
        # Word 0: TICK_RELATIVE at [56] (Payload bit 56)
        # Check definition: TICK_RELATIVE_OFFSET = 56
        assert (w0 >> 56) & 0xFF == 0xAB
        
        # Word 1: VJT at [0]
        assert (w1 & 0xFFFFFFFF) == 0xDEADBEEF
        
        # Word 2: RESET_MODE at [62]
        assert (w2 >> 62) & 0x3 == 2
        
        # Word 3: VJT_INITIAL at [0]
        assert (w3 & 0xFFF) == 0x123

    def test_gen_neuron_config_fold(self):
        """Test Type 3: Fold Neuron"""
        neurons = [{
            'fold_range_xy': 0x1F, # Word0 [53] -> Actually [63:53]
            'fold_vjt_0': 0x1234   # Word3 [31:0]
        }]
        
        frames = OfflineFrameGenV2.gen_neuron_config(0, 0, neurons, mode='fold')
        
        assert frames.size == 5
        w0 = frames[1]
        
        # FOLD_RANGE_XY_OFFSET = 53
        assert (w0 >> 53) & 0x7FF == 0x1F

    def test_gen_input_config(self):
        """Test Type 4: Input SRAM (512 bits -> 8 Words)"""
        # Create a block of 8 dummy words
        block = [i for i in range(8)]
        frames = OfflineFrameGenV2.gen_input_config(0, 0, [block])
        
        # Expect: 1 Header + 8 Body Frames
        assert frames.size == 9
        _, _, num = self._parse_packet_header(frames[0])
        assert num == 8
        
        for i in range(8):
            assert frames[i+1] == i


    def test_gen_test_request(self):
        """Test Input Frame (Bit 23 = 1)"""
        frames = OfflineFrameGenV2.gen_test_request(
            0x10, 0x20, 
            int(FHV2.TEST_TYPE1), # Type 1
            start_addr=0x5, 
            num_packets=10
        )
        
        assert frames.size == 1 # Only header
        
        hd, c, cp, _ = self._parse_frame(frames[0])
        assert hd == FHV2.TEST_TYPE1
        
        # Parse Packet Header
        pkg_type, start, num = self._parse_packet_header(frames[0])
        
        assert pkg_type == 1  # Must be 1 for Test Request
        assert start == 0x5
        assert num == 10

if __name__ == "__main__":
    pytest.main([__file__])