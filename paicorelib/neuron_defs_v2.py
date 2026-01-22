from enum import IntEnum, unique

from .utils import _mask

__all__ = [
    # Neuron reg limits
    "OfflineNeuRegLimV2",
    # Types
    "OutputType",
    "FoldType",
    "NeuronType",
    "ThresholdNegMode",
    "ThresholdPosMode",
    "LateralInhibitionMode",
    "LeakMultiComparisonOrder",
    "LeakMultiInputMode",
    "LeakMultiMode",
    "LeakAddMode",
    "WeightCompressType",
]


class OfflineNeuRegLimV2:
    """Limits of offline neuron registers for chip v2.5."""

    TICK_RELATIVE_MAX = _mask(8)
    ADDR_AXON_MAX = _mask(9)
    ADDR_CORE_COORD_MIN = -_mask(5)
    ADDR_CORE_COORD_MAX = _mask(5)
    WEIGHT_SKEW_MAX = _mask(16)
    WEIGHT_ADDRESS_MAX = _mask(12)
    RESET_V_MAX = _mask(15)
    RESET_V_MIN = -(RESET_V_MAX + 1)
    THRES_NEG_MAX = _mask(31)
    THRES_NEG_MIN = -(THRES_NEG_MAX + 1)
    THRES_POS_MAX = THRES_NEG_MAX
    THRES_POS_MIN = THRES_NEG_MIN
    LEAK_TAU_MAX = _mask(5)
    LEAK_TAU_MIN = -(LEAK_TAU_MAX + 1)
    LEAK_V_MAX = _mask(19)
    LEAK_V_MIN = -(LEAK_V_MAX + 1)
    VJT_INITIAL_MAX = _mask(11)
    VJT_INITIAL_MIN = -(VJT_INITIAL_MAX + 1)
    FOLD_RANGE_MAX = _mask(11)
    FOLD_SKEW_MAX = _mask(11)
    FOLD_AXON_MAX = _mask(11)
    FOLD_NUMBER_MAX = _mask(29)


@unique
class OutputType(IntEnum):
    """Output type selection.
    0: Output spike or activation value
    1: Output membrane potential
    """

    VALUE = 0
    POTENTIAL = 1


@unique
class FoldType(IntEnum):
    """Fold type selection.
    0: Unfolded neuron
    1: Folded neuron
    """

    UNFOLDED = 0
    FOLDED = 1


@unique
class NeuronType(IntEnum):
    """Neuron type selection.
    0: Half neuron mode, occupies 128 bits
    1: Full neuron mode, occupies 256 bits
    """

    HALF = 0
    FULL = 1


@unique
class ThresholdNegMode(IntEnum):
    """Negative threshold mode selection.
    0: Fire mode
    1: Floor mode
    """

    FIRE = 0
    FLOOR = 1


@unique
class ThresholdPosMode(IntEnum):
    """Positive threshold mode selection.
    0: Fire mode
    1: Ceiling mode
    """

    FIRE = 0
    CEILING = 1


@unique
class LateralInhibitionMode(IntEnum):
    """Lateral inhibition mode selection.
    0: No lateral inhibition
    1: With lateral inhibition, if the current core generated a spike in the last time step, membrane potential accumulates from reset_v
    """

    DISABLE = 0
    ENABLE = 1


@unique
class LeakMultiComparisonOrder(IntEnum):
    """Multiplicative leak sequence.
    0: Leak before threshold comparison
    1: Leak after threshold comparison reset
    """

    BEFORE_COMPARE = 0
    AFTER_COMPARE = 1


@unique
class LeakMultiInputMode(IntEnum):
    """Whether input participates in multiplicative leak.
    0: Input does not participate in multiplicative leak
    1: Input participates in multiplicative leak
    """

    DISABLE = 0
    ENABLE = 1


@unique
class LeakMultiMode(IntEnum):
    """Multiplicative leak mode selection.
    0: No multiplicative leak, but membrane potential can be shifted
    1: With multiplicative leak
    """

    DISABLE = 0
    ENABLE = 1


@unique
class LeakAddMode(IntEnum):
    """Additive leak mode selection.
    0: Forward leak mode
    1: Backward leak mode
    """

    FORWARD = 0
    BACKWARD = 1


@unique
class WeightCompressType(IntEnum):
    """Weight type.
    0: Dense mode, weight data uncompressed
    1: Sparse mode, weight data CSC compressed
    """

    DENSE = 0
    SPARSE = 1
