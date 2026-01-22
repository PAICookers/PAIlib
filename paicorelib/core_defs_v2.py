from enum import IntEnum, unique

from .utils import _mask

__all__ = [
    # Core reg limits
    "OfflineCoreRegLimV2",
    # Types
    "SNNMode",
    "PoolingMode",
    "AddPotentialMode",
    "ZeroOutputMode",
    "DataSign",
    "DataWidth",
    "CSCAccelerateMode",
]


class OfflineCoreRegLimV2:
    """Limits of offline core registers for chip v2.5."""

    AXON_SKEW_MIN = -_mask(15)
    AXON_SKEW_MAX = -AXON_SKEW_MIN
    NEURON_NUMBER_MAX = _mask(12) + 1  # <= 4096
    TEST_CORE_COORD_MIN = -_mask(5)
    TEST_CORE_COORD_MAX = -TEST_CORE_COORD_MIN
    GLOBAL_SEND_MAX = _mask(7)
    GLOBAL_RECEIVE_MAX = _mask(6)
    THREAD_NUMBER_MAX = _mask(10)
    BUSY_CYCLE_MAX = _mask(12)
    DELAY_CYCLE_MAX = _mask(16)
    WIDTH_CYCLE_MAX = _mask(8)
    TICK_START_MAX = _mask(16)
    TICK_DURATION_MAX = _mask(32)
    TICK_INITIAL_MAX = _mask(16)


@unique
class SNNMode(IntEnum):
    """SNN and ANN mode selection.
    0: SNN mode, using LIF operation rules
    1: ANN mode, using activation function operation rules
    """

    SNN = 0
    ANN = 1


@unique
class PoolingMode(IntEnum):
    """Pooling mode selection.
    0: Average pooling
    1: Max pooling
    """

    AVERAGE = 0
    MAX = 1


@unique
class AddPotentialMode(IntEnum):
    """Accumulation mode selection.
    0: Normal accumulation
    1: Direct accumulation of membrane potential
    """

    NORMAL = 0
    DIRECT_ADD = 1


@unique
class ZeroOutputMode(IntEnum):
    """Whether to output zero values.
    0: Do not output zero values
    1: Output zero values
    """

    DISABLE = 0
    ENABLE = 1


@unique
class DataWidth(IntEnum):
    """Input/output/weight data width."""

    WIDTH_1BIT = 0
    WIDTH_2BIT = 1
    WIDTH_4BIT = 2
    WIDTH_8BIT = 3


@unique
class DataSign(IntEnum):
    """Input/output/weight data sign."""

    UNSIGNED = 0
    SIGNED = 1


@unique
class CSCAccelerateMode(IntEnum):
    """CSC compressed calculation acceleration mode.
    0: No acceleration, compressed storage calculation is slow
    1: Acceleration, compressed storage calculation is faster, but vjt_initial needs to store weight_address_start again
    """

    DISABLE = 0
    ENABLE = 1
