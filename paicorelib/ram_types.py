from enum import Enum, IntEnum, unique

__all__ = [
    "ResetMode",
    "LeakComparisonMode",
    "NegativeThresholdMode",
    "LeakDirectionMode",
    "LeakIntegrationMode",
    "SynapticIntegrationMode",
    "ThresholdMode",
]

"""
    Type defines of RAM parameters.
    See Section 2.4.2 in V2.1 Manual for details.
"""


@unique
class ResetMode(Enum):
    """Reset modes of neurons. 2-bit.

    - `MODE_NORMAL`: normal mode. Default value.
    - `MODE_LINEAR`: linear mode.
    - `MODE_NONRESET`: non-reset mode.

    NOTE: Same as `reset_mode` in V2.1.
    """

    MODE_NORMAL = 0  # Default value.
    MODE_LINEAR = 1
    MODE_NONRESET = 2


@unique
class LeakComparisonMode(Enum):
    """Leak after comparison or before. 1-bit.

    - `LEAK_BEFORE_COMP`: leak before comparison. Default value.
    - `LEAK_AFTER_COMP`: leak after comparison.

    NOTE: Same as `leak_post` in V2.1.
    """

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


@unique
class ThresholdMode(IntEnum):
    """Indicates whether the neuron reaches the threshold or not.

    - `NOT_EXCEEDED`: dosen't exceed. Must reset after neuronal reset.
    - `EXCEED_POSITIVE`: exceeded positive threshold.
    - `EXCEED_NEGATIVE`: exceeded negative threshold.
    """

    NOT_EXCEEDED = 0
    EXCEED_POSITIVE = 1
    EXCEED_NEGATIVE = 2
