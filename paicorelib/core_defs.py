from enum import IntEnum, unique

__all__ = [
    "CoreLim",
    "SNNMode",
    "PoolingMode",
    "PotentialAddMode",
    "ZeroOutputMode",
    "InputSignMode",
    "InputWidth",
    "OutputSignMode",
    "OutputWidth",
    "WeightSignMode",
    "WeightWidth",
    "LCNMode",
    "CSCAccelerateMode",
]


class CoreLim:
    """Core parameter limits."""

    AXON_SKEW_MIN = -32768
    AXON_SKEW_MAX = 32767
    NEURON_NUMBER_MAX = 4096
    TEST_CORE_OFFSET_MIN = -31
    TEST_CORE_OFFSET_MAX = 31
    GLOBAL_SEND_MAX = 127
    GLOBAL_RECEIVE_MAX = 63
    THREAD_NUMBER_MAX = 1023
    BUSY_CYCLE_MAX = 4095
    DELAY_CYCLE_MAX = 65535
    WIDTH_CYCLE_MAX = 255
    TICK_START_MAX = 65535
    TICK_DURATION_MAX = 4294967295
    TICK_INITIALIZER_MAX = 65535


@unique
class SNNMode(IntEnum):
    """SNN and ANN mode selection.
    0: SNN mode, using LIF operation rules;
    1: ANN mode, using activation function operation rules;
    """

    SNN = 0
    ANN = 1


@unique
class PoolingMode(IntEnum):
    """Pooling mode selection.
    0: Average pooling;
    1: Max pooling;
    """

    AVERAGE = 0
    MAX = 1


@unique
class PotentialAddMode(IntEnum):
    """Accumulation mode selection.
    0: Normal accumulation;
    1: Direct accumulation of membrane potential;
    """

    NORMAL = 0
    DIRECT_POTENTIAL = 1


@unique
class ZeroOutputMode(IntEnum):
    """Whether to output zero values.
    0: Do not output zero values;
    1: Output zero values;
    """

    DISABLE = 0
    ENABLE = 1


@unique
class InputSignMode(IntEnum):
    """Input data sign selection.
    0: Input is unsigned;
    1: Input is signed;
    """

    UNSIGNED = 0
    SIGNED = 1


@unique
class InputWidth(IntEnum):
    """Input data bit width selection.
    00: Input is 1-bit;
    01: Input is 2-bit;
    10: Input is 4-bit;
    11: Input is 8-bit;
    """

    WIDTH_1BIT = 0
    WIDTH_2BIT = 1
    WIDTH_4BIT = 2
    WIDTH_8BIT = 3


@unique
class OutputSignMode(IntEnum):
    """Output data sign selection.
    0: Output is unsigned;
    1: Output is signed;
    """

    UNSIGNED = 0
    SIGNED = 1


@unique
class OutputWidth(IntEnum):
    """Output data bit width selection.
    00: Output is 1-bit;
    01: Output is 2-bit;
    10: Output is 4-bit;
    11: Output is 8-bit;
    """

    WIDTH_1BIT = 0
    WIDTH_2BIT = 1
    WIDTH_4BIT = 2
    WIDTH_8BIT = 3


@unique
class WeightSignMode(IntEnum):
    """Weight data sign selection.
    0: Weight is unsigned;
    1: Weight is signed;
    """

    UNSIGNED = 0
    SIGNED = 1


@unique
class WeightWidth(IntEnum):
    """Weight data bit width selection.
    00: Weight is 1-bit;
    01: Weight is 2-bit;
    10: Weight is 4-bit;
    11: Weight is 8-bit;
    """

    WIDTH_1BIT = 0
    WIDTH_2BIT = 1
    WIDTH_4BIT = 2
    WIDTH_8BIT = 3


@unique
class LCNMode(IntEnum):
    """Control the scale of fan-in extension.
    1x is 512 fan-in / input bit width.
    """

    LCN_1X = 0
    LCN_2X = 1
    LCN_4X = 2
    LCN_8X = 3
    LCN_16X = 4
    LCN_32X = 5
    LCN_64X = 6
    LCN_128X = 7


@unique
class CSCAccelerateMode(IntEnum):
    """CSC compressed calculation acceleration mode.
    0: No acceleration, compressed storage calculation is slow;
    1: Acceleration, compressed storage calculation is faster, but vjt_initial needs to store weight_address_start again;
    """

    DISABLE = 0
    ENABLE = 1
