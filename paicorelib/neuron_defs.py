from enum import Enum, unique

from .hw_defs import HwOfflineCoreParams as HwOffCore
from .hw_defs import HwOnlineCoreParams as HwOnCore
from .hw_defs import HwParams
from .utils import _mask

__all__ = [
    # Neuron reg limits
    "NeuRegLim",
    "OfflineNeuRegLim",
    "OnlineNeuRegLim",
    "OnlineNeuRegLim_WW1",
    "OnlineNeuRegLim_WWn",
    # Types
    "ResetMode",
    "LeakComparisonMode",
    "NegativeThresholdMode",
    "LeakDirectionMode",
    "LeakIntegrationMode",
    "SynapticIntegrationMode",
]


class NeuRegLim:
    """Limits of neuron registers."""

    ADDR_COORD_MAX = _mask(HwParams.N_BIT_COORD_ADDR)


class OfflineNeuRegLim(NeuRegLim):
    """Limits of offline neuron registers."""

    ADDR_TS_MAX = HwOffCore.N_TIMESLOT_MAX - 1
    ADDR_AXON_MAX = HwOffCore.ADDR_AXON_MAX
    RESET_V_MAX = _mask(29)
    RESET_V_MIN = -(RESET_V_MAX + 1)
    THRES_MASK_BITS_MAX = _mask(5)
    NEG_THRES_MAX = _mask(29)
    POS_THRES_MAX = _mask(29)
    LEAK_V_MAX = _mask(29)
    LEAK_V_MIN = -(LEAK_V_MAX + 1)
    BIT_TRUNC_MAX = 29  # The highest bit is the sign bit
    VOLTAGE_MAX = _mask(29)
    VOLTAGE_MIN = -(VOLTAGE_MAX + 1)


class OnlineNeuRegLim(NeuRegLim):
    """Basic limits of online neuron registers."""

    ADDR_TS_MAX = HwOnCore.N_TIMESLOT_MAX - 1
    ADDR_AXON_MAX = HwOnCore.ADDR_AXON_MAX
    PLASTICITY_START_MAX = _mask(10)  # <= ADDR_AXON_MAX
    PLASTICITY_END_MAX = PLASTICITY_START_MAX  # <= ADDR_AXON_MAX


class OnlineNeuRegLim_WW1(OnlineNeuRegLim):
    """Limits of online neuron registers with 1-bit weight width."""

    LEAK_V_MAX = _mask(14)
    LEAK_V_MIN = -(LEAK_V_MAX + 1)
    THRES_MAX = _mask(14)
    THRES_MIN = -(THRES_MAX + 1)
    FLOOR_THRES_MAX = _mask(6)
    FLOOR_THRES_MIN = -(FLOOR_THRES_MAX + 1)
    RESET_V_MAX = _mask(5)
    RESET_V_MIN = -(RESET_V_MAX + 1)
    INIT_V_MAX = _mask(5)
    INIT_V_MIN = -(INIT_V_MAX + 1)
    VOLTAGE_MAX = _mask(14)
    VOLTAGE_MIN = -(VOLTAGE_MAX + 1)


class OnlineNeuRegLim_WWn(OnlineNeuRegLim):
    """Limits of online neuron registers with weight width > 1-bit."""

    LEAK_V_MAX = _mask(31)
    LEAK_V_MIN = -(LEAK_V_MAX + 1)
    THRES_MAX = _mask(31)
    THRES_MIN = -(THRES_MAX + 1)
    FLOOR_THRES_MAX = _mask(31)
    FLOOR_THRES_MIN = -(FLOOR_THRES_MAX + 1)
    RESET_V_MAX = _mask(31)
    RESET_V_MIN = -(RESET_V_MAX + 1)
    INIT_V_MAX = _mask(31)
    INIT_V_MIN = -(INIT_V_MAX + 1)
    VOLTAGE_MAX = _mask(31)
    VOLTAGE_MIN = -(VOLTAGE_MAX + 1)


@unique
class ResetMode(Enum):
    """Reset modes of neurons. 2-bit.
    - `MODE_NORMAL`: normal mode. Default value.
    - `MODE_LINEAR`: linear mode.
    - `MODE_NONRESET`: non-reset mode.
    """

    MODE_NORMAL = 0  # Default value.
    MODE_LINEAR = 1
    MODE_NONRESET = 2


@unique
class LeakComparisonMode(Enum):
    """Leak after comparison or before. Default is `LEAK_BEFORE_COMP`."""

    LEAK_BEFORE_COMP = 0  # Default value.
    LEAK_AFTER_COMP = 1


@unique
class NegativeThresholdMode(Enum):
    """Modes of negative threshold. 1-bit.

    - `MODE_RESET`: reset mode. Default value.
    - `MODE_SATURATION`: saturation(floor) mode.

    NOTE: Same as `threshold_neg_mode` in V2.1.
    """

    MODE_RESET = 0  # Default value.
    MODE_SATURATION = 1


@unique
class LeakDirectionMode(Enum):
    """Direction of Leak, forward or reversal.

    - `MODE_FORWARD`: forward Leak. Default value.
    - `MODE_REVERSAL`: reversal Leak.

    NOTE: Same as `leak_reversal_flag` in V2.1.
    """

    MODE_FORWARD = 0  # Default value.
    MODE_REVERSAL = 1


@unique
class LeakIntegrationMode(Enum):
    """Mode of Leak integration, deterministic or stochastic.

    - `MODE_DETERMINISTIC`: deterministic Leak. Default value.
    - `MODE_STOCHASTIC`: stochastic Leak.

    NOTE: Same as `leak_det_stoch` in V2.1.
    """

    MODE_DETERMINISTIC = 0  # Default value.
    MODE_STOCHASTIC = 1


@unique
class SynapticIntegrationMode(Enum):
    """Modes of synaptic integration, deterministic or stochastic.

    - `MODE_DETERMINISTIC`: deterministic weights. Default value.
    - `MODE_STOCHASTIC`: stochastic weights.

    NOTE: Same as `weight_det_stoch` in V2.1.
    """

    MODE_DETERMINISTIC = 0  # Default value.
    MODE_STOCHASTIC = 1
