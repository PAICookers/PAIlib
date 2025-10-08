import sys
from enum import Enum, IntEnum, auto, unique
from functools import wraps
from typing import Any

from .utils import _mask

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

__all__ = [
    # Tyep definitions
    "WeightWidth",
    "LCN_EX",
    "InputWidthFormat",
    "SpikeWidthFormat",
    "MaxPoolingEnable",
    "SNNModeEnable",
    "LUTRandomEnable",
    "DecayRandomEnable",
    "LeakOrder",
    "OnlineModeEnable",
    # Auxiliary classes
    "CoreType",
    "CoreMode",
    # Functions
    "get_core_mode",
    "core_mode_check",
]

"""Type definitions of registers of cores in the chip."""

# Type definitions of offline core registers.


@unique
class WeightWidth(IntEnum):
    """Weight bit width of crossbar. 8-bit by default."""

    WEIGHT_WIDTH_1BIT = 0
    WEIGHT_WIDTH_2BIT = 1
    WEIGHT_WIDTH_4BIT = 2
    WEIGHT_WIDTH_8BIT = 3  # Default value.


@unique
class LCNExtension(IntEnum):
    """Scale of fan-in extension. 1X by default.
    - For `MODE_ANN`, `LCN_1X` = 144x.
    - For `MODE_SNN` or `MODE_BANN`, `LCN_1X` = 1152x.
    """

    LCN_1X = 0  # Default value.
    LCN_2X = 1
    LCN_4X = 2
    LCN_8X = 3
    LCN_16X = 4
    LCN_32X = 5
    LCN_64X = 6


LCN_EX = LCNExtension


@unique
class InputWidthFormat(IntEnum):
    """Format of input spike. 1-bit by default."""

    WIDTH_1BIT = 0  # Default value.
    WIDTH_8BIT = 1


@unique
class SpikeWidthFormat(IntEnum):
    """Format of output spike. 1-bit by default."""

    WIDTH_1BIT = 0  # Default value.
    WIDTH_8BIT = 1


@unique
class MaxPoolingEnable(IntEnum):
    """Enable max pooling or not in 8-bit input format. Disable by default."""

    DISABLE = 0  # Default value.
    ENABLE = 1


@unique
class SNNModeEnable(IntEnum):
    """Enable SNN mode or not. Enable by default."""

    DISABLE = 0
    ENABLE = 1  # Default value.


@unique
class CoreType(Enum):
    """Types of cores."""

    OFFLINE = auto()
    ONLINE = auto()


_ModeParamTuple: TypeAlias = tuple[InputWidthFormat, SpikeWidthFormat, SNNModeEnable]


@unique
class CoreMode(Enum):
    """Working mode of the offline cores. Decided by `input_width`, `spike_width` and `SNN_EN` of   \
        core parameters registers.

        Mode                        input_width    spike_width    SNN_EN
        BANN                            0               0           0
        SNN                             0               0           1
        BANN/SNN to ANN                 0               1           0
        BANN/SNN to SNN with values     0               1           1
        ANN to BANN/SNN                 1               0           0
        ANN                             1               1           0
        Undefined                       1               0           1
        Undefined                       1               1           1
    """

    MODE_BANN = (
        InputWidthFormat.WIDTH_1BIT,
        SpikeWidthFormat.WIDTH_1BIT,
        SNNModeEnable.DISABLE,
    )
    MODE_SNN = (
        InputWidthFormat.WIDTH_1BIT,
        SpikeWidthFormat.WIDTH_1BIT,
        SNNModeEnable.ENABLE,
    )
    MODE_BANN_OR_SNN_TO_ANN = (
        InputWidthFormat.WIDTH_1BIT,
        SpikeWidthFormat.WIDTH_8BIT,
        SNNModeEnable.DISABLE,
    )
    MODE_BANN_OR_SNN_TO_VSNN = (
        InputWidthFormat.WIDTH_1BIT,
        SpikeWidthFormat.WIDTH_8BIT,
        SNNModeEnable.ENABLE,
    )
    MODE_ANN_TO_BANN_OR_SNN = (
        InputWidthFormat.WIDTH_8BIT,
        SpikeWidthFormat.WIDTH_1BIT,
        SNNModeEnable.DISABLE,
    )
    MODE_ANN = (
        InputWidthFormat.WIDTH_8BIT,
        SpikeWidthFormat.WIDTH_8BIT,
        SNNModeEnable.DISABLE,
    )

    @property
    def is_snn(self) -> bool:
        """Whether the SNN mode is enabled."""
        return self is CoreMode.MODE_SNN or self is CoreMode.MODE_BANN_OR_SNN_TO_VSNN

    @property
    def is_iw8(self) -> bool:
        """Wether the input width is 8-bit."""
        return self is CoreMode.MODE_ANN_TO_BANN_OR_SNN or self is CoreMode.MODE_ANN

    @property
    def is_ow8(self) -> bool:
        return (
            self is CoreMode.MODE_BANN_OR_SNN_TO_ANN
            or self is CoreMode.MODE_BANN_OR_SNN_TO_VSNN
            or self is CoreMode.MODE_ANN
        )

    @property
    def conf(self) -> _ModeParamTuple:
        return self.value


def get_core_mode(
    iw: InputWidthFormat, sw: SpikeWidthFormat, sm: SNNModeEnable
) -> CoreMode:
    try:
        return CoreMode((iw, sw, sm))
    except ValueError:
        raise ValueError(
            f"invalid mode conf: (input_width, spike_width, snn_mode) = ({iw}, {sw}, {sm}).",
        )


def core_mode_check(func):
    @wraps(func)
    def wrapper(reg_dict: dict[str, Any], *args, **kwargs):
        _ = get_core_mode(
            reg_dict["input_width"], reg_dict["spike_width"], reg_dict["snn_en"]
        )
        return func(reg_dict, *args, **kwargs)

    return wrapper


# Type definitions of online core registers.


@unique
class LUTRandomEnable(IntEnum):
    """Enable random update for LUT or not. Disable by default."""

    DISABLE = 0  # Default value.
    ENABLE = 1


@unique
class DecayRandomEnable(IntEnum):
    """Enable random update for weight decay or not. Disable by default."""

    DISABLE = 0  # Default value.
    ENABLE = 1


@unique
class LeakOrder(IntEnum):
    """Leak after comparison or before. Default is `LEAK_BEFORE_COMP`."""

    LEAK_BEFORE_COMP = 0  # Default value.
    LEAK_AFTER_COMP = 1


@unique
class OnlineModeEnable(IntEnum):
    """Enable online mode or not (offline inference mode). Enable by default."""

    DISABLE = 0
    ENABLE = 1  # Default value.


class RegDefs:
    """Type definitions of registers of cores in the chip."""

    # Bit width
    TICK_WAIT_START_BIT_MAX = 15  # Unsigned
    TICK_WAIT_END_BIT_MAX = TICK_WAIT_START_BIT_MAX

    # Value range
    TICK_WAIT_START_MAX = _mask(TICK_WAIT_START_BIT_MAX)
    TICK_WAIT_END_MAX = TICK_WAIT_START_MAX


class OfflineRegDefs(RegDefs):
    """Type definitions of offline core registers."""

    pass


class OnlineRegDefs(RegDefs):
    """Type definitions of online core registers."""

    # Bit width
    LATERAL_INHI_VALUE_BIT_MAX = 32  # Signed
    WEIGHT_DECAY_VALUE_BIT_MAX = 8  # Signed
    UPPER_WEIGHT_BIT_MAX = 8  # Signed
    LOWER_WEIGHT_BIT_MAX = UPPER_WEIGHT_BIT_MAX
    NEU_START_BIT_MAX = 10  # Unsigned
    NEU_END_BIT_MAX = NEU_START_BIT_MAX
    RANDOM_SEED_BIT_MAX = 16  # Unsigned

    # Value range
    LATERAL_INHI_VALUE_MAX = _mask(LATERAL_INHI_VALUE_BIT_MAX - 1)
    LATERAL_INHI_VALUE_MIN = -(LATERAL_INHI_VALUE_MAX + 1)
    WEIGHT_DECAY_VALUE_MAX = _mask(WEIGHT_DECAY_VALUE_BIT_MAX - 1)
    WEIGHT_DECAY_VALUE_MIN = -(WEIGHT_DECAY_VALUE_MAX + 1)
    UPPER_WEIGHT_MAX = _mask(UPPER_WEIGHT_BIT_MAX - 1)
    UPPER_WEIGHT_MIN = -(UPPER_WEIGHT_MAX + 1)
    LOWER_WEIGHT_MAX = UPPER_WEIGHT_MAX
    LOWER_WEIGHT_MIN = UPPER_WEIGHT_MIN
    NEU_START_MAX = _mask(NEU_START_BIT_MAX)
    NEU_END_MAX = NEU_START_MAX
    RANDOM_SEED_MAX = _mask(RANDOM_SEED_BIT_MAX)
