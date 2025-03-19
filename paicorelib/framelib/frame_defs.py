from enum import IntEnum, unique


def _mask(mask_bit: int) -> int:
    return (1 << mask_bit) - 1


@unique
class FrameType(IntEnum):
    """Basic types of Frames"""

    CONFIG = 0
    TEST = 0x1
    WORK = 0x2
    UNKNOWN = 0x3


_FT = FrameType


@unique
class FrameHeader(IntEnum):
    """Frame headers"""

    CONFIG_TYPE1 = (_FT.CONFIG << 2) | 0b00
    CONFIG_TYPE2 = (_FT.CONFIG << 2) | 0b01
    CONFIG_TYPE3 = (_FT.CONFIG << 2) | 0b10
    CONFIG_TYPE4 = (_FT.CONFIG << 2) | 0b11

    TEST_TYPE1 = (_FT.TEST << 2) | 0b00
    TEST_TYPE2 = (_FT.TEST << 2) | 0b01
    TEST_TYPE3 = (_FT.TEST << 2) | 0b10
    TEST_TYPE4 = (_FT.TEST << 2) | 0b11

    WORK_TYPE1 = (_FT.WORK << 2) | 0b00
    WORK_TYPE2 = (_FT.WORK << 2) | 0b01
    WORK_TYPE3 = (_FT.WORK << 2) | 0b10
    WORK_TYPE4 = (_FT.WORK << 2) | 0b11


class FrameFormat:
    """General frame mask & offset."""

    FRAME_LENGTH = 64
    GENERAL_MASK = _mask(FRAME_LENGTH)

    # Header
    GENERAL_HEADER_OFFSET = 60
    GENERAL_HEADER_MASK = _mask(4)

    GENERAL_FRAME_TYPE_OFFSET = GENERAL_HEADER_OFFSET
    GENERAL_FRAME_TYPE_MASK = GENERAL_HEADER_MASK

    # Chip address
    GENERAL_CHIP_ADDR_OFFSET = 50
    GENERAL_CHIP_ADDR_MASK = _mask(10)
    # Chip X/Y address
    GENERAL_CHIP_X_ADDR_OFFSET = 55
    GENERAL_CHIP_X_ADDR_MASK = _mask(5)
    GENERAL_CHIP_Y_ADDR_OFFSET = GENERAL_CHIP_ADDR_OFFSET
    GENERAL_CHIP_Y_ADDR_MASK = _mask(5)

    # Core address
    GENERAL_CORE_ADDR_OFFSET = 40
    GENERAL_CORE_ADDR_MASK = _mask(10)
    # Core X/Y address
    GENERAL_CORE_X_ADDR_OFFSET = 45
    GENERAL_CORE_X_ADDR_MASK = _mask(5)
    GENERAL_CORE_Y_ADDR_OFFSET = GENERAL_CORE_ADDR_OFFSET
    GENERAL_CORE_Y_ADDR_MASK = _mask(5)

    # Core* address
    GENERAL_CORE_EX_ADDR_OFFSET = 30
    GENERAL_CORE_EX_ADDR_MASK = _mask(10)
    # Core* X/Y address
    GENERAL_CORE_X_EX_ADDR_OFFSET = 35
    GENERAL_CORE_X_EX_ADDR_MASK = _mask(5)
    GENERAL_CORE_Y_EX_ADDR_OFFSET = GENERAL_CORE_EX_ADDR_OFFSET
    GENERAL_CORE_Y_EX_ADDR_MASK = _mask(5)

    # Global core = Chip address + core address
    GENERAL_CORE_GLOBAL_ADDR_OFFSET = GENERAL_CORE_ADDR_OFFSET
    GENERAL_CORE_GLOBAL_ADDR_MASK = _mask(20)

    # Payload
    GENERAL_PAYLOAD_OFFSET = 0
    GENERAL_PAYLOAD_LENGTH = 30
    GENERAL_PAYLOAD_MASK = _mask(GENERAL_PAYLOAD_LENGTH)

    # Package SRAM address
    GENERAL_PACKAGE_SRAM_ADDR_OFFSET = 20
    GENERAL_PACKAGE_SRAM_ADDR_MASK = _mask(10)

    # Package type bit: 0 for config/test-out, 1 for test-in
    GENERAL_PACKAGE_TYPE_OFFSET = 19
    GENERAL_PACKAGE_TYPE_MASK = _mask(1)

    # #N of packages
    GENERAL_PACKAGE_NUM_OFFSET = 0
    GENERAL_PACKAGE_NUM_MASK = _mask(19)


class _TestFrameFormat_In(FrameFormat):
    """General test input frame format."""

    pass


class _TestFrameFormat_Out(FrameFormat):
    """General test output frame format."""

    TEST_CHIP_ADDR_OFFSET = 50
    TEST_CHIP_ADDR_MASK = _mask(10)


# Offline frame format


class OfflineFrameFormat(FrameFormat):
    """General offline frame format."""

    pass


class OfflineConfigFrame1Format(OfflineFrameFormat):
    """Offline config frame type I. Random seed register."""

    pass


class OfflineConfigFrame2Format(OfflineFrameFormat):
    """Offline config frame type II. Parameter registers."""

    # Frame #1
    WEIGHT_WIDTH_OFFSET = 28
    WEIGHT_WIDTH_MASK = _mask(2)

    LCN_OFFSET = 24
    LCN_MASK = _mask(4)

    INPUT_WIDTH_OFFSET = 23
    INPUT_WIDTH_MASK = 1

    SPIKE_WIDTH_OFFSET = 22
    SPIKE_WIDTH_MASK = 1

    # NOTE: This parameter actually means the effective number of dendrites,
    # which can be found in the manual for details.
    NUM_VALID_DENDRITE_OFFSET = 9
    NUM_VALID_DENDRITE_MASK = _mask(13)

    POOL_MAX_OFFSET = 8
    POOL_MAX_MASK = _mask(1)

    TICK_WAIT_START_HIGH8_OFFSET = 0
    TICK_WAIT_START_COMBINATION_OFFSET = 7
    TICK_WAIT_START_HIGH8_MASK = _mask(8)

    # Frame #2
    TICK_WAIT_START_LOW7_OFFSET = 23
    TICK_WAIT_START_LOW7_MASK = _mask(7)

    TICK_WAIT_END_OFFSET = 8
    TICK_WAIT_END_MASK = _mask(15)

    SNN_EN_OFFSET = 7
    SNN_EN_MASK = _mask(1)

    TARGET_LCN_OFFSET = 3
    TARGET_LCN_MASK = _mask(4)

    TEST_CHIP_ADDR_HIGH3_OFFSET = 0
    TEST_CHIP_ADDR_LOW7_OFFSET = 7
    TEST_CHIP_ADDR_HIGH3_MASK = _mask(3)

    # Frame #3
    TEST_CHIP_ADDR_LOW7_OFFSET = 23
    TEST_CHIP_ADDR_LOW7_MASK = _mask(7)


class OfflineConfigFrame3Format(OfflineFrameFormat):
    """Offline config frame type III. Neuron RAM.

    Total payload: 256 bits, LSB.
    Frame #1: RAM[63:0]
    Frame #2: RAM[127:64]
    Frame #3: RAM[191:128]
    Frame #4: 42'd0, RAM[213:192]
    """

    # Frame #1
    VJT_PRE_OFFSET = 0
    VJT_PRE_MASK = _mask(30)

    BIT_TRUNCATE_OFFSET = 30
    BIT_TRUNCATE_MASK = _mask(5)

    WEIGHT_DET_STOCH_OFFSET = 35
    WEIGHT_DET_STOCH_MASK = _mask(1)

    LEAK_V_LOW28_OFFSET = 36
    LEAK_V_LOW28_MASK = _mask(28)

    # Frame #2
    LEAK_V_HIGH2_OFFSET = 0
    LEAK_V_HIGH2_MASK = _mask(2)

    LEAK_DET_STOCH_OFFSET = 2
    LEAK_DET_STOCH_MASK = _mask(1)

    LEAK_REVERSAL_FLAG_OFFSET = 3
    LEAK_REVERSAL_FLAG_MASK = _mask(1)

    THRESHOLD_POS_OFFSET = 4
    THRESHOLD_POS_MASK = _mask(29)

    THRESHOLD_NEG_OFFSET = 33
    THRESHOLD_NEG_MASK = _mask(29)

    THRESHOLD_NEG_MODE_OFFSET = 62
    THRESHOLD_NEG_MODE_MASK = _mask(1)

    THRESHOLD_MASK_CTRL_LOW1_OFFSET = 63
    THRESHOLD_MASK_CTRL_LOW1_MASK = _mask(1)

    # Frame #3
    THRESHOLD_MASK_CTRL_HIGH4_OFFSET = 0
    THRESHOLD_MASK_CTRL_HIGH4_MASK = _mask(4)

    LEAK_POST_OFFSET = 4
    LEAK_POST_MASK = _mask(1)

    RESET_V_OFFSET = 5
    RESET_V_MASK = _mask(30)

    RESET_MODE_OFFSET = 35
    RESET_MODE_MASK = _mask(2)

    ADDR_CHIP_Y_OFFSET = 37
    ADDR_CHIP_Y_MASK = _mask(5)

    ADDR_CHIP_X_OFFSET = 42
    ADDR_CHIP_X_MASK = _mask(5)

    ADDR_CORE_Y_EX_OFFSET = 47
    ADDR_CORE_Y_EX_MASK = _mask(5)

    ADDR_CORE_X_EX_OFFSET = 52
    ADDR_CORE_X_EX_MASK = _mask(5)

    ADDR_CORE_Y_OFFSET = 57
    ADDR_CORE_Y_MASK = _mask(5)

    ADDR_CORE_X_LOW2_OFFSET = 62
    ADDR_CORE_X_LOW2_MASK = _mask(2)

    # Frame #4
    ADDR_CORE_X_HIGH3_OFFSET = 0
    ADDR_CORE_X_HIGH3_MASK = _mask(3)

    ADDR_AXON_OFFSET = 3
    ADDR_AXON_MASK = _mask(11)

    TICK_RELATIVE_OFFSET = 14
    TICK_RELATIVE_MASK = _mask(8)


class OfflineConfigFrame4Format(OfflineFrameFormat):
    """Offline config frame type IV. Weight RAM."""

    pass


OfflineRandomSeedFormat = OfflineConfigFrame1Format
OfflineCoreRegFormat = OfflineConfigFrame2Format
OfflineNeuronRAMFormat = OfflineConfigFrame3Format
OfflineWeightRAMFormat = OfflineConfigFrame4Format


class OfflineWorkFrame1Format(OfflineFrameFormat):
    """Work frame type I. Spike."""

    RESERVED_OFFSET = 27
    RESERVED_MASK = _mask(3)

    AXON_OFFSET = 16
    AXON_MASK = _mask(11)

    TIMESLOT_OFFSET = 8
    TIMESLOT_MASK = _mask(8)

    DATA_OFFSET = 0
    DATA_MASK = _mask(8)


class OfflineWorkFrame2Format(OfflineFrameFormat):
    """Work frame type II. Sync."""

    RESERVED_OFFSET = 30
    RESERVED_MASK = _mask(20)

    TIME_OFFSET = 0
    TIME_MASK = _mask(30)


class OfflineWorkFrame3Format(OfflineFrameFormat):
    """Work frame type III. Clear."""

    RESERVED_OFFSET = 0
    RESERVED_MASK = _mask(50)


class OfflineWorkFrame4Format(OfflineFrameFormat):
    """Work frame type IV. Init."""

    RESERVED_OFFSET = 0
    RESERVED_MASK = _mask(50)


SpikeFrameFormat = OfflineWorkFrame1Format
SyncFrameFormat = OfflineWorkFrame2Format
ClearFrameFormat = OfflineWorkFrame3Format
InitFrameFormat = OfflineWorkFrame4Format


class _OfflineTestFrameFormat_In(_TestFrameFormat_In):
    """General offline test input frame format."""

    pass


class _OfflineTestFrameFormat_Out(_TestFrameFormat_Out):
    """General offline test output frame format."""

    pass


class OfflineTestFrame1Format_In(_OfflineTestFrameFormat_In):
    """Test frame type I. Random seed register, input."""

    pass


class OfflineTestFrame2Format_In(_OfflineTestFrameFormat_In):
    """Test frame type II. Parameter registers, input."""

    pass


class OfflineTestFrame3Format_In(_OfflineTestFrameFormat_In):
    """Test frame type III. Neuron RAM, input."""

    pass


class OfflineTestFrame4Format_In(_OfflineTestFrameFormat_In):
    """Test frame type IV. Weight RAM, input."""

    pass


class OfflineTestFrame1Format_Out(
    _OfflineTestFrameFormat_Out, OfflineConfigFrame1Format
):
    """Test frame type I. Random seed register, output."""

    pass


class OfflineTestFrame2Format_Out(
    _OfflineTestFrameFormat_Out, OfflineConfigFrame2Format
):
    """Test frame type II. Parameter registers, output."""

    pass


class OfflineTestFrame3Format_Out(
    _OfflineTestFrameFormat_Out, OfflineConfigFrame3Format
):
    """Test frame type III. Neuron RAM, output."""

    pass


class OfflineTestFrame4Format_Out(
    _OfflineTestFrameFormat_Out, OfflineConfigFrame4Format
):
    """Test frame type IV. Weight RAM, output."""

    pass
