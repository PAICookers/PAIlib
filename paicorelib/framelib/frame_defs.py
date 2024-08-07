from enum import IntEnum, unique


def _mask(mask_bit: int) -> int:
    return (1 << mask_bit) - 1


@unique
class FrameType(IntEnum):
    """Types of Frames"""

    FRAME_CONFIG = 0
    FRAME_TEST = 0x1
    FRAME_WORK = 0x2
    FRAME_UNKNOWN = 0x3


@unique
class FrameHeader(IntEnum):
    """Frame headers"""

    CONFIG_TYPE1 = 0b0000
    """Type I, random seed"""
    CONFIG_TYPE2 = 0b0001
    """Type II, parameter REG"""
    CONFIG_TYPE3 = 0b0010
    """Type III, neuron RAM"""
    CONFIG_TYPE4 = 0b0011
    """Type IV, weight RAM"""

    TEST_TYPE1 = 0b0100
    TEST_TYPE2 = 0b0101
    TEST_TYPE3 = 0b0110
    TEST_TYPE4 = 0b0111

    WORK_TYPE1 = 0b1000
    """Type I, spike"""
    WORK_TYPE2 = 0b1001
    """Type II, sync"""
    WORK_TYPE3 = 0b1010
    """Type III, clear"""
    WORK_TYPE4 = 0b1011
    """Type IV, init"""


class FrameFormat:
    """General frame mask & offset."""

    GENERAL_MASK = _mask(64)

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

    # Frame 前34位
    GENERAL_FRAME_PRE_OFFSET = 30
    GENERAL_FRAME_PRE_MASK = _mask(34)

    # 通用数据帧LOAD掩码
    GENERAL_PAYLOAD_OFFSET = 0
    GENERAL_PAYLOAD_MASK = _mask(30)

    # 通用数据包LOAD掩码
    GENERAL_PACKAGE_SRAM_ADDR_OFFSET = 20
    GENERAL_PACKAGE_SRAM_ADDR_MASK = _mask(10)

    GENERAL_PACKAGE_TYPE_OFFSET = 19
    GENERAL_PACKAGE_TYPE_MASK = _mask(1)

    GENERAL_PACKAGE_NUM_OFFSET = 0
    GENERAL_PACKAGE_NUM_MASK = _mask(19)


class OfflineFrameFormat(FrameFormat):
    """Basic offline frame format"""

    pass


class OnlineFrameFormat(FrameFormat):
    """Basic online frame format"""

    pass


class OfflineConfigFrame1Format(OfflineFrameFormat):
    """Offline config frame type I, random seed"""

    pass


class OfflineConfigFrame2Format(OfflineFrameFormat):
    """Offline config frame type II, parameter registers"""

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

    # 用于配置帧2型test_chip_addr
    TEST_CHIP_ADDR_HIGH3_OFFSET = 0
    TEST_CHIP_ADDR_LOW7_OFFSET = 7
    TEST_CHIP_ADDR_HIGH3_MASK = _mask(3)

    # Frame #3
    TEST_CHIP_ADDR_LOW7_OFFSET = 23
    TEST_CHIP_ADDR_LOW7_MASK = _mask(7)


class OfflineConfigFrame3Format(OfflineFrameFormat):
    """Offline config frame type III, parameter RAM"""

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
    """Offline config frame type IV, weight RAM"""

    pass


RandomSeedFormat = OfflineConfigFrame1Format
ParameterRegFormat = OfflineConfigFrame2Format
ParameterRAMFormat = OfflineConfigFrame3Format
WeightRAMFormat = OfflineConfigFrame4Format


class OfflineWorkFrame1Format(OfflineFrameFormat):
    """Work frame type I"""

    RESERVED_OFFSET = 27
    RESERVED_MASK = _mask(3)

    AXON_OFFSET = 16
    AXON_MASK = _mask(11)

    TIMESLOT_OFFSET = 8
    TIMESLOT_MASK = _mask(8)

    DATA_OFFSET = 0
    DATA_MASK = _mask(8)


class OfflineWorkFrame2Format(OfflineFrameFormat):
    """Work frame type II"""

    RESERVED_OFFSET = 30
    RESERVED_MASK = _mask(20)

    TIME_OFFSET = 0
    TIME_MASK = _mask(30)


class OfflineWorkFrame3Format(OfflineFrameFormat):
    """Work frame type III"""

    RESERVED_OFFSET = 0
    RESERVED_MASK = _mask(50)


class OfflineWorkFrame4Format(OfflineFrameFormat):
    """Work frame type IV"""

    RESERVED_OFFSET = 0
    RESERVED_MASK = _mask(50)


SpikeFrameFormat = OfflineWorkFrame1Format
SyncFrameFormat = OfflineWorkFrame2Format
ClearFrameFormat = OfflineWorkFrame3Format
InitFrameFormat = OfflineWorkFrame4Format


# TODO frame format for online frames
