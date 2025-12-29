from enum import IntEnum, unique

__all__ = [
    "OutputType",
    "FoldType",
    "NeuronType",
    "ResetMode",
    "ThresholdNegMode",
    "ThresholdPosMode",
    "LateralInhibitionMode",
    "LeakMultiSequence",
    "LeakMultiInputMode",
    "LeakMultiMode",
    "LeakAddMode",
    "WeightCompressMode",
    "NeuronLim",
]


class NeuronLim:
    """Neuron parameter limits."""

    TICK_RELATIVE_MAX = 255
    ADDR_AXON_MAX = 511
    ADDR_CORE_OFFSET_MIN = -31
    ADDR_CORE_OFFSET_MAX = 31
    WEIGHT_SKEW_MAX = 65535
    WEIGHT_ADDRESS_MAX = 4095
    RESET_V_MIN = -32768
    RESET_V_MAX = 32767
    LEAK_TAU_MIN = -32
    LEAK_TAU_MAX = 31
    LEAK_V_MIN = -524288
    LEAK_V_MAX = 524287
    VJT_INITIAL_MIN = -2048
    VJT_INITIAL_MAX = 2047
    FOLD_RANGE_MAX = 2047
    FOLD_SKEW_MAX = 2047
    FOLD_AXON_MAX = 2047
    FOLD_NUMBER_MAX = 536870911


@unique
class OutputType(IntEnum):
    """Output type selection.
    0: Output spike or activation value;
    1: Output membrane potential;
    """

    SPIKE_OR_ACTIVATION = 0
    POTENTIAL = 1


@unique
class FoldType(IntEnum):
    """Fold type selection.
    0: Unfolded neuron;
    1: Folded neuron;
    """

    UNFOLDED = 0
    FOLDED = 1


@unique
class NeuronType(IntEnum):
    """Neuron type selection.
    0: Half neuron mode, occupies 128 bits;
    1: Full neuron mode, occupies 256 bits;
    """

    HALF = 0
    FULL = 1


@unique
class ResetMode(IntEnum):
    """Reset mode selection.
    00: Fixed reset (hard reset) mode;
    01: Subtraction reset (soft reset) mode;
    10: No reset mode;
    """

    FIXED = 0
    SUBTRACTION = 1
    NO_RESET = 2


@unique
class ThresholdNegMode(IntEnum):
    """Negative threshold mode selection.
    0: Fire mode;
    1: Floor mode;
    """

    FIRE = 0
    FLOOR = 1


@unique
class ThresholdPosMode(IntEnum):
    """Positive threshold mode selection.
    0: Fire mode;
    1: Ceiling mode;
    """

    FIRE = 0
    CEILING = 1


@unique
class LateralInhibitionMode(IntEnum):
    """Lateral inhibition mode selection.
    0: No lateral inhibition;
    1: With lateral inhibition, if the current core generated a spike in the last time step, membrane potential accumulates from reset_v;
    """

    DISABLE = 0
    ENABLE = 1


@unique
class LeakMultiSequence(IntEnum):
    """Multiplicative leak sequence.
    0: Leak before threshold comparison;
    1: Leak after threshold comparison reset;
    """

    BEFORE_COMPARE = 0
    AFTER_RESET = 1


@unique
class LeakMultiInputMode(IntEnum):
    """Whether input participates in multiplicative leak.
    0: Input does not participate in multiplicative leak;
    1: Input participates in multiplicative leak;
    """

    DISABLE = 0
    ENABLE = 1


@unique
class LeakMultiMode(IntEnum):
    """Multiplicative leak mode selection.
    0: No multiplicative leak, but membrane potential can be shifted;
    1: With multiplicative leak;
    """

    DISABLE = 0
    ENABLE = 1


@unique
class LeakAddMode(IntEnum):
    """Additive leak mode selection.
    0: Forward leak mode;
    1: Backward leak mode;
    """

    FORWARD = 0
    BACKWARD = 1


@unique
class WeightCompressMode(IntEnum):
    """Weight type.
    0: Dense mode, weight data uncompressed;
    1: Sparse mode, weight data CSC compressed;
    """

    DENSE = 0
    SPARSE = 1
