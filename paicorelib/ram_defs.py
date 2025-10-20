from enum import Enum, unique

from .hw_defs import HwOfflineCoreParams as HwOffCore
from .hw_defs import HwOnlineCoreParams as HwOnCore
from .hw_defs import HwParams
from .utils import _mask

__all__ = [
    # Type definitions
    "ResetMode",
    "LeakComparisonMode",
    "NegativeThresholdMode",
    "LeakDirectionMode",
    "LeakIntegrationMode",
    "SynapticIntegrationMode",
]

"""Type definitions of neuron RAM parameters for both online & offline cores."""


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


class RAMDefs:
    """Bit width & value range of neuron RAM parameters."""

    # Bit width
    COORD_BIT_MAX = HwParams.N_BIT_COORD_ADDR

    # Value range
    COORD_MAX = _mask(COORD_BIT_MAX)


class OfflineRAMDefs(RAMDefs):
    # Bit width
    TS_BIT_MAX = HwOffCore.N_BIT_TIMESLOT
    ADDR_AXON_BIT_MAX = 11  # Use `ADDR_AXON_MAX` as the high limit
    RESET_V_BIT_MAX = 30  # Signed
    THRES_MASK_BITS_BIT_MAX = 5  # Unsigned
    NEG_THRES_BIT_MAX = 29  # Unsigned
    POS_THRES_BIT_MAX = 29  # Unsigned
    LEAK_V_BIT_MAX = 30  # Signed
    BIT_TRUNC_BIT_MAX = 5  # Unsigned
    VOLTAGE_BIT_MAX = 30  # Signed

    # Value range
    ADDR_TS_MAX = HwOffCore.N_TIMESLOT_MAX - 1
    ADDR_AXON_MAX = HwOffCore.ADDR_AXON_MAX
    RESET_V_MAX = _mask(RESET_V_BIT_MAX - 1)
    RESET_V_MIN = -(RESET_V_MAX + 1)
    THRES_MASK_BITS_MAX = _mask(THRES_MASK_BITS_BIT_MAX)
    NEG_THRES_MAX = _mask(NEG_THRES_BIT_MAX)
    POS_THRES_MAX = _mask(POS_THRES_BIT_MAX)
    LEAK_V_MAX = _mask(LEAK_V_BIT_MAX - 1)  # Only for scalar
    LEAK_V_MIN = -(LEAK_V_MAX + 1)  # Only for scalar
    BIT_TRUNC_MAX = 29  # The highest bit is the sign bit
    VOLTAGE_MAX = _mask(VOLTAGE_BIT_MAX - 1)
    VOLTAGE_MIN = -(VOLTAGE_MAX + 1)


class OnlineRAMDefs(RAMDefs):
    """Bit width & value range of neuron RAM parameters for online cores."""

    # Bit width
    TS_BIT_MAX = HwOnCore.N_BIT_TIMESLOT
    ADDR_AXON_BIT_MAX = HwOnCore.ADDR_AXON_MAX
    PLASTICITY_START_BIT_MAX = 10  # Unsigned
    PLASTICITY_END_BIT_MAX = PLASTICITY_START_BIT_MAX  # Unsigned

    # Value range
    ADDR_TS_MAX = HwOnCore.N_TIMESLOT_MAX - 1
    ADDR_AXON_MAX = HwOnCore.ADDR_AXON_MAX
    PLASTICITY_START_MAX = _mask(PLASTICITY_START_BIT_MAX)  # <= ADDR_AXON_MAX
    PLASTICITY_END_MAX = PLASTICITY_START_MAX  # <= ADDR_AXON_MAX


class OnlineRAMDefs_WW1(OnlineRAMDefs):
    """Bit width & value range of neuron RAM parameters for online cores with 1-bit weight width."""

    # Bit width
    LEAK_V_BIT_MAX = 15  # Signed
    THRES_BIT_MAX = 15  #  Signed
    FLOOR_THRES_BIT_MAX = 7  # Signed
    RESET_V_MAX = 6  # Signed
    INIT_V_BIT_MAX = 6  # Signed
    VOLTAGE_BIT_MAX = 15  # Signed

    # Value range
    LEAK_V_MAX = _mask(LEAK_V_BIT_MAX - 1)  # Only for scalar
    LEAK_V_MIN = -(LEAK_V_MAX + 1)  # Only for scalar
    THRES_MAX = _mask(THRES_BIT_MAX - 1)
    THRES_MIN = -(THRES_MAX + 1)
    FLOOR_THRES_MAX = _mask(FLOOR_THRES_BIT_MAX - 1)
    FLOOR_THRES_MIN = -(FLOOR_THRES_MAX + 1)
    RESET_V_MAX = _mask(RESET_V_MAX - 1)
    RESET_V_MIN = -(RESET_V_MAX + 1)
    INIT_V_MAX = _mask(INIT_V_BIT_MAX - 1)
    INIT_V_MIN = -(INIT_V_MAX + 1)
    VOLTAGE_MAX = _mask(VOLTAGE_BIT_MAX - 1)
    VOLTAGE_MIN = -(VOLTAGE_MAX + 1)


class OnlineRAMDefs_WWn(OnlineRAMDefs):
    """Bit width & value range of neuron RAM parameters for online cores with weight width > 1-bit."""

    # Bit width
    LEAK_V_BIT_MAX = 32  # Signed
    THRES_BIT_MAX = 32  #  Signed
    FLOOR_THRES_BIT_MAX = 32  # Signed
    RESET_V_MAX = 32  # Signed
    INIT_V_BIT_MAX = 32  # Signed
    VOLTAGE_BIT_MAX = 32  # Signed

    # Value range
    LEAK_V_MAX = _mask(LEAK_V_BIT_MAX - 1)  # Only for scalar
    LEAK_V_MIN = -(LEAK_V_MAX + 1)  # Only for scalar
    THRES_MAX = _mask(THRES_BIT_MAX - 1)
    THRES_MIN = -(THRES_MAX + 1)
    FLOOR_THRES_MAX = _mask(FLOOR_THRES_BIT_MAX - 1)
    FLOOR_THRES_MIN = -(FLOOR_THRES_MAX + 1)
    RESET_V_MAX = _mask(RESET_V_MAX - 1)
    RESET_V_MIN = -(RESET_V_MAX + 1)
    INIT_V_MAX = _mask(INIT_V_BIT_MAX - 1)
    INIT_V_MIN = -(INIT_V_MAX + 1)
    VOLTAGE_MAX = _mask(VOLTAGE_BIT_MAX - 1)
    VOLTAGE_MIN = -(VOLTAGE_MAX + 1)
