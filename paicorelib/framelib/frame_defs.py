from enum import IntEnum, unique

from ..utils import _mask


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


@unique
class FramePackageType(IntEnum):
    """Frame package type."""

    CONF_TESTOUT = 0  # For conf & test-out
    TESTIN = 1  # For test-in


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

    # Package neuron start address
    # NOTE: The argument `sram_base_addr` of config/test frame 3 & 4 is incorrectly
    # named. In fact, it is the neuron start address.
    GENERAL_PACKAGE_NEU_START_ADDR_OFFSET = 20
    GENERAL_PACKAGE_NEU_START_ADDR_MASK = _mask(10)

    # Package type
    GENERAL_PACKAGE_TYPE_OFFSET = 19
    GENERAL_PACKAGE_TYPE_MASK = _mask(1)

    # #N of packages
    GENERAL_PACKAGE_NUM_OFFSET = 0
    GENERAL_PACKAGE_NUM_MASK = _mask(19)

    # Package attributes
    GENERAL_PACKAGE_LEN = FRAME_LENGTH


FF = FrameFormat


class _TestFrameFormat_In(FF):
    """General test input frame format."""

    pass


class _TestFrameFormat_Out(FF):
    """General test output frame format."""

    TEST_CHIP_ADDR_OFFSET = FF.GENERAL_CHIP_ADDR_OFFSET
    TEST_CHIP_ADDR_MASK = FF.GENERAL_CHIP_ADDR_OFFSET


"""Frame format for offline cores"""


class OfflineConfigFrame1Format(FF):
    """Offline config frame type I. Random seed register."""

    pass


class OfflineConfigFrame2Format(FF):
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
    NUM_DENDRITE_OFFSET = 9
    NUM_DENDRITE_MASK = _mask(13)

    MAX_POOLING_EN_OFFSET = 8
    MAX_POOLING_EN_MASK = _mask(1)

    TICK_WAIT_START_HIGH8_OFFSET = 0
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
    TEST_CHIP_ADDR_HIGH3_MASK = _mask(3)

    # Frame #3
    TEST_CHIP_ADDR_LOW7_OFFSET = 23
    TEST_CHIP_ADDR_LOW7_MASK = _mask(7)


class OfflineConfigFrame3Format(FF):
    """Offline config frame type III. Neuron RAM.

    Total payload: 256 bits, LSB.
    Frame #1: RAM[63:0]
    Frame #2: RAM[127:64]
    Frame #3: RAM[191:128]
    Frame #4: 42'd0, RAM[213:192]
    """

    # Frame #1
    VOLTAGE_OFFSET = 0
    VOLTAGE_MASK = _mask(30)

    BIT_TRUNC_OFFSET = 30
    BIT_TRUNC_MASK = _mask(5)

    SYN_INTEGRATION_MODE_OFFSET = 35
    SYN_INTEGRATION_MODE_MASK = _mask(1)

    LEAK_V_LOW28_OFFSET = 36
    LEAK_V_LOW28_MASK = _mask(28)

    # Frame #2
    LEAK_V_HIGH2_OFFSET = 0
    LEAK_V_HIGH2_MASK = _mask(2)

    LEAK_INTEGRATION_MODE_OFFSET = 2
    LEAK_INTEGRATION_MODE_MASK = _mask(1)

    LEAK_DIRECTION_OFFSET = 3
    LEAK_DIRECTION_MASK = _mask(1)

    POS_THRESHOLD_OFFSET = 4
    POS_THRESHOLD_MASK = _mask(29)

    NEG_THRES_OFFSET = 33
    NEG_THRES_MASK = _mask(29)

    NEG_THRES_MODE_OFFSET = 62
    NEG_THRES_MODE_MASK = _mask(1)

    THRE_MASK_BITS_LOW1_OFFSET = 63
    THRE_MASK_BITS_LOW1_MASK = _mask(1)

    # Frame #3
    THRES_MASK_BITS_HIGH4_OFFSET = 0
    THRES_MASK_BITS_HIGH4_MASK = _mask(4)

    LEAK_COMPARISON_OFFSET = 4
    LEAK_COMPARISON_MASK = _mask(1)

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


class OfflineConfigFrame4Format(FF):
    """Offline config frame type IV. Weight RAM."""

    pass


OfflineRandomSeedFormat = OfflineConfigFrame1Format
OfflineCoreRegFormat = OfflineConfigFrame2Format
OfflineNeuRAMFormat = OfflineConfigFrame3Format
OfflineWeightRAMFormat = OfflineConfigFrame4Format


class OfflineWorkFrame1Format(FF):
    """Work frame type I. Spike."""

    RESERVED_OFFSET = 27
    RESERVED_MASK = _mask(3)

    AXON_OFFSET = 16
    AXON_MASK = _mask(11)

    TIMESLOT_OFFSET = 8
    TIMESLOT_MASK = _mask(8)

    DATA_OFFSET = 0
    DATA_MASK = _mask(8)


class OfflineWorkFrame2Format(FF):
    """Work frame type II. Sync."""

    RESERVED_OFFSET = 30
    RESERVED_MASK = _mask(20)

    # #N of sync signals
    N_SYNC_OFFSET = 0
    N_SYNC_MASK = _mask(30)


class OfflineWorkFrame3Format(FF):
    """Work frame type III. Clear."""

    RESERVED_OFFSET = 0
    RESERVED_MASK = _mask(50)


class OfflineWorkFrame4Format(FF):
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


"""Frame format for online cores"""


class OnlineConfigFrame1Format(FF):
    """Online config frame type I. LUT."""

    LUT_OFFSET = 0
    LUT_MASK = _mask(30)


class OnlineConfigFrame2Format(FF):
    """Online config frame type II. Core."""

    # Frame #1
    WEIGHT_WIDTH_OFFSET = 28
    WEIGHT_WIDTH_MASK = _mask(2)

    LCN_OFFSET = 26
    LCN_MASK = _mask(2)

    LATERAL_INHI_VALUE_HIGH26_OFFSET = 0
    LATERAL_INHI_VALUE_HIGH26_MASK = _mask(26)

    # Frame #2
    LATERAL_INHI_VALUE_LOW6_OFFSET = 24
    LATERAL_INHI_VALUE_LOW6_MASK = _mask(6)

    WEIGHT_DECAY_OFFSET = 16
    WEIGHT_DECAY_MASK = _mask(8)

    UPPER_WEIGHT_OFFSET = 8
    UPPER_WEIGHT_MASK = _mask(8)

    LOWER_WEIGHT_OFFSET = 0
    LOWER_WEIGHT_MASK = _mask(8)

    # Frame #3
    NEURON_START_OFFSET = 20
    NEURON_START_MASK = _mask(10)

    NEURON_END_OFFSET = 10
    NEURON_END_MASK = _mask(10)

    INHI_CORE_X_EX_OFFSET = 5
    INHI_CORE_X_EX_MASK = _mask(5)

    INHI_CORE_Y_EX_OFFSET = 0
    INHI_CORE_Y_EX_MASK = _mask(5)

    # Frame #4
    TICK_WAIT_START_OFFSET = 15
    TICK_WAIT_START_MASK = _mask(15)

    TICK_WAIT_END_OFFSET = 0
    TICK_WAIT_END_MASK = _mask(15)

    # Frame #5
    LUT_RANDOM_EN_HIGH30_OFFSET = 0
    LUT_RANDOM_EN_HIGH30_MASK = _mask(30)

    # Frame #6
    LUT_RANDOM_EN_LOW30_OFFSET = 30
    LUT_RANDOM_EN_LOW30_MASK = _mask(30)

    # Frame #7
    DECAY_RANDOM_EN_OFFSET = 29
    DECAY_RANDOM_EN_MASK = _mask(1)

    LEAK_ORDER_OFFSET = 27
    LEAK_ORDER_MASK = _mask(1)

    ONLINE_MODE_EN_OFFSET = 26
    ONLINE_MODE_EN_MASK = _mask(1)

    TEST_CHIP_ADDR_OFFSET = 16
    TEST_CHIP_ADDR_MASK = _mask(10)

    RANDOM_SEED_OFFSET = 0
    RANDOM_SEED_MASK = _mask(16)

    # Frame #8
    ZERO_PADDING_OFFSET = 0
    ZERO_PADDING_MASK = _mask(30)


class OnlineConfigFrame3Format(FF):
    """Online config frame type III. Neuron RAM.

    Total payload: 256 bits, MSB.
    Frame #1: RAM[127:64]
    Frame #2: RAM[63:0]

    NOTE: For neuron in cores with 1-bit weight width: use 2 frame only.
          For neuron in cores with other weight width: use 4 frames.
    """

    # Frame #1, common for WW=1 & WW2/4/8
    PLASTICITY_END_OFFSET = 0
    PLASTICITY_END_MASK = _mask(10)

    PLASTICITY_START_OFFSET = 10
    PLASTICITY_START_MASK = _mask(10)

    ADDR_AXON_OFFSET = 20
    ADDR_AXON_MASK = _mask(11)

    ADDR_CORE_Y_EX_OFFSET = 31
    ADDR_CORE_Y_EX_MASK = _mask(5)

    ADDR_CORE_X_EX_OFFSET = 36
    ADDR_CORE_X_EX_MASK = _mask(5)

    ADDR_CORE_Y_OFFSET = 41
    ADDR_CORE_Y_MASK = _mask(5)

    ADDR_CORE_X_OFFSET = 46
    ADDR_CORE_X_MASK = _mask(5)

    ADDR_CHIP_Y_OFFSET = 51
    ADDR_CHIP_Y_MASK = _mask(5)

    ADDR_CHIP_X_OFFSET = 56
    ADDR_CHIP_X_MASK = _mask(5)

    TIME_RELATIVE_OFFSET = 61
    TIME_RELATIVE_MASK = _mask(3)


class OnlineConfigFrame3Format_WW1(OnlineConfigFrame3Format):
    """Online config frame type III with 1-bit weight width."""

    # Frame #1
    LEAK_V_OFFSET = 49
    LEAK_V_MASK = _mask(15)

    POS_THRES_OFFSET = 34
    POS_THRES_MASK = _mask(15)

    NEG_THRES_OFFSET = 27
    NEG_THRES_MASK = _mask(7)

    RESET_V_OFFSET = 21
    RESET_V_MASK = _mask(6)

    INIT_V_OFFSET = 15
    INIT_V_MASK = _mask(6)

    VOLTAGE_OFFSET = 0
    VOLTAGE_MASK = _mask(15)


class OnlineConfigFrame3Format_WWn(OnlineConfigFrame3Format):
    """Online config frame type III with 2/4/8-bit weight width."""

    # Frame #1
    LEAK_V_OFFSET = 32
    LEAK_V_MASK = _mask(32)

    POS_THRES_OFFSET = 0
    POS_THRES_MASK = _mask(32)

    # Frame #2
    NEG_THRES_OFFSET = 32
    NEG_THRES_MASK = _mask(32)

    RESET_V_OFFSET = 0
    RESET_V_MASK = _mask(32)

    # Frame #3
    INIT_V_OFFSET = 32
    INIT_V_MASK = _mask(32)

    VOLTAGE_OFFSET = 32
    VOLTAGE_MASK = _mask(0)


class OnlineConfigFrame4Format(FF):
    """Online config frame type IV. Weight RAM."""

    pass


OnlineLUTFormat = OnlineConfigFrame1Format
OnlineCoreRegFormat = OnlineConfigFrame2Format
OnlineNeuRAMFormat = OnlineConfigFrame3Format
OnlineNeuRAMFormat_WW1 = OnlineConfigFrame3Format_WW1
OnlineNeuRAMFormat_WWn = OnlineConfigFrame3Format_WWn
OnlineWeightRAMFormat = OnlineConfigFrame4Format


@unique
class Online_WF1F_SubType(IntEnum):
    TYPE_I = 0b000
    TYPE_II = 0b100
    TYPE_III = 0b010
    TYPE_IV = 0b001


class OnlineWorkFrame1Format(FF):
    """Work frame type I."""

    SUBTYPE_OFFSET = 27
    SUBTYPE_MASK = _mask(3)


class OnlineWorkFrame1Format_1(OnlineWorkFrame1Format):
    """Work frame type I, sub-type I. Spike."""

    AXON_OFFSET = 16
    AXON_MASK = _mask(10)

    RESERVED_1_OFFSET = 11
    RESERVED_1_MASK = _mask(5)

    TIMESLOT_OFFSET = 8
    TIMESLOT_MASK = _mask(3)

    RESERVED_2_OFFSET = 0
    RESERVED_2_MASK = _mask(8)


class OnlineWorkFrame1Format_2(OnlineWorkFrame1Format):
    """Work frame type I, sub-type II. Start."""

    RESERVED_OFFSET = 0
    RESERVED_MASK = _mask(27)


class OnlineWorkFrame1Format_3(OnlineWorkFrame1Format):
    """Work frame type I, sub-type III. End."""

    RESERVED_OFFSET = 0
    RESERVED_MASK = _mask(27)


class OnlineWorkFrame1Format_4(OnlineWorkFrame1Format):
    """Work frame type I, sub-type IV. Lateral inhibition."""

    RESERVED_OFFSET = 0
    RESERVED_MASK = _mask(27)


class OnlineWorkFrame2Format(OfflineWorkFrame2Format):
    """Work frame type II. Sync."""


class OnlineWorkFrame3Format(OfflineWorkFrame3Format):
    """Work frame type III. Clear."""


class OnlineWorkFrame4Format(OfflineWorkFrame4Format):
    """Work frame type IV. Init."""


class _OnlineTestFrameFormat_In(_TestFrameFormat_In):
    """General online test input frame format."""

    pass


class _OnlineTestFrameFormat_Out(_TestFrameFormat_Out):
    """General online test output frame format."""

    pass


class OnlineTestFrame1Format_In(_OnlineTestFrameFormat_In):
    """Test frame type I. LUT, input."""

    pass


class OnlineTestFrame2Format_In(_OnlineTestFrameFormat_In):
    """Test frame type II. Core, input."""

    pass


class OnlineTestFrame3Format_In(_OnlineTestFrameFormat_In):
    """Test frame type III. Neuron RAM, input."""

    pass


class OnlineTestFrame4Format_In(_OnlineTestFrameFormat_In):
    """Test frame type IV. Weight RAM, input."""

    pass


class OnlineTestFrame1Format_Out(_OnlineTestFrameFormat_Out, OnlineConfigFrame1Format):
    """Test frame type I. LUT, output."""

    pass


class OnlineTestFrame2Format_Out(_OnlineTestFrameFormat_Out, OnlineConfigFrame2Format):
    """Test frame type II. Core, output."""

    pass


class OnlineTestFrame3Format_Out(_OnlineTestFrameFormat_Out, OnlineConfigFrame3Format):
    """Test frame type III. Neuron RAM, output."""

    pass


class OnlineTestFrame4Format_Out(_OnlineTestFrameFormat_Out, OnlineConfigFrame4Format):
    """Test frame type IV. Weight RAM, output."""

    pass
