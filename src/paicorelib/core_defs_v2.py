from enum import IntEnum, unique

from .utils import _mask

__all__ = [
    # Core reg limits
    "OfflineCoreRegLimV2",
    "OnlineCoreRegLimV2",
    # Types
    "SNNMode",
    "OnlineSNNMode",
    "PoolingMode",
    "AddPotentialMode",
    "ZeroOutputMode",
    "OnlineCoreWorkMode",
    "OnlineCoreType",
    "OnlineDataWidth",
    "OnlineCoreUpdateType",
    "DataSign",
    "DataWidth",
    "CSCAccelerateMode",
]


class _CommonCoreRegLimV2:
    """Common limits of core registers for chip v2.5."""

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


class OfflineCoreRegLimV2(_CommonCoreRegLimV2):
    """Limits of offline core registers for chip v2.5."""


class OnlineCoreRegLimV2(_CommonCoreRegLimV2):
    """Limits of online core registers for chip v2.5."""

    UPDATE_NUMBER_MAX = _mask(13)


@unique
class SNNMode(IntEnum):
    """SNN and ANN mode selection.
    0: SNN mode, using LIF operation rules
    1: ANN mode, using activation function operation rules
    """

    SNN = 0
    ANN = 1


@unique
class OnlineSNNMode(IntEnum):
    """Online core SNN/ANN mode selection.
    0: SNN mode, using LIF operation rules
    1: ANN mode, without activation-function rules
    2: ANN mode, using ReLU activation-function rules
    3: ANN mode, using LUT activation-function rules
    """

    SNN_LIF = 0
    ANN_NO_ACT = 1
    ANN_RELU = 2
    ANN_LUT = 3


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
class OnlineCoreWorkMode(IntEnum):
    """Online core work mode selection.
    0: Forward inference calculation mode
    1: Loss function calculation mode
    2: Output-layer gradient calculation mode
    3: Middle-layer gradient calculation mode
    4: Average-pooling-layer gradient calculation mode
    5: Max-pooling-layer gradient calculation mode
    6: Forward weight update mode
    7: Backward weight update mode
    """

    FORWARD_INFERENCE = 0
    LOSS_FN = 1
    OUTPUT_LAYER_GRADIENT = 2
    MIDDLE_LAYER_GRADIENT = 3
    AVG_POOLING_GRADIENT = 4
    MAX_POOLING_GRADIENT = 5
    FORWARD_WEIGHT_UPDATE = 6
    BACKWARD_WEIGHT_UPDATE = 7


@unique
class OnlineCoreType(IntEnum):
    """Online core input/output peer core type selection.
    0: Offline core
    1: Online core
    """

    OFFLINE = 0
    ONLINE = 1


@unique
class OnlineDataWidth(IntEnum):
    """Online core input/output data width selection."""

    TYPE_1BIT = 0
    TYPE_FP16 = 1
    TYPE_UINT8 = 2
    TYPE_INT8 = 3


@unique
class OnlineCoreUpdateType(IntEnum):
    """Online update-core update type selection, encoded in output_width."""

    WEIGHT = 0
    WEIGHT_BIAS = 1
    KAHAN_WEIGHT = 2
    KAHAN_WEIGHT_BIAS = 3


@unique
class DataWidth(IntEnum):
    """Input/output/weight data width."""

    WIDTH_1BIT = 0
    WIDTH_2BIT = 1
    WIDTH_4BIT = 2
    WIDTH_8BIT = 3
    WIDTH_16BIT = 4
    WIDTH_32BIT = 5


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
