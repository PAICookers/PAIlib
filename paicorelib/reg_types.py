import sys
from enum import Enum, IntEnum, auto, unique
from functools import wraps
from typing import Any

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

__all__ = [
    "WeightWidth",
    "LCN_EX",
    "InputWidthFormat",
    "SpikeWidthFormat",
    "MaxPoolingEnable",
    "SNNModeEnable",
    "CoreType",
    "CoreMode",
    "get_core_mode",
]

"""
    Type defines of registers of cores in the chip.
"""

# Type defines of offline core registers.


@unique
class WeightWidthType(IntEnum):
    """Weight bit width of crossbar. 2-bit.
    - `WEIGHT_WIDTH_XBIT` for X-bit. Default value is `WEIGHT_WIDTH_8BIT`.
    """

    WEIGHT_WIDTH_1BIT = 0
    WEIGHT_WIDTH_2BIT = 1
    WEIGHT_WIDTH_4BIT = 2
    WEIGHT_WIDTH_8BIT = 3  # Default value.


@unique
class LCNExtensionType(IntEnum):
    """Scale of fan-in extension. 4-bit. X-time LCN extension. Default value is `LCN_1X`.
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


@unique
class InputWidthFormatType(IntEnum):
    """Format of input spike. 1-bit.
    - `WIDTH_1BIT`: 1-bit spike. Default value.
    - `WIDTH_8BIT`: 8-bit activation.
    """

    WIDTH_1BIT = 0  # Default value.
    WIDTH_8BIT = 1


@unique
class SpikeWidthFormatType(IntEnum):
    """Format of output spike. 1-bit.
    - `WIDTH_1BIT`: 1-bit spike. Default value.
    - `WIDTH_8BIT`: 8-bit activation.
    """

    WIDTH_1BIT = 0  # Default value.
    WIDTH_8BIT = 1


@unique
class MaxPoolingEnableType(IntEnum):
    """Enable max pooling or not in 8-bit input format. 1-bit.
    - `DISABLE`: pooling max disable. Default value.
    - `ENABLE`: pooling max enable.
    """

    DISABLE = 0  # Default value.
    ENABLE = 1


@unique
class SNNModeEnableType(IntEnum):
    """Enable SNN mode or not. 1-bit.
    - `DISABLE`: SNN mode disable.
    - `ENABLE`: SNN mode enable. Default value.
    """

    DISABLE = 0
    ENABLE = 1  # Default value.


WeightWidth = WeightWidthType
LCN_EX = LCNExtensionType
InputWidthFormat = InputWidthFormatType
SpikeWidthFormat = SpikeWidthFormatType
MaxPoolingEnable = MaxPoolingEnableType
SNNModeEnable = SNNModeEnableType


@unique
class CoreType(Enum):
    """Types of cores."""

    TYPE_OFFLINE = auto()
    TYPE_ONLINE = auto()


_ModeParamTuple: TypeAlias = tuple[
    InputWidthFormatType, SpikeWidthFormatType, SNNModeEnableType
]


@unique
class CoreMode(Enum):
    """Working mode of cores. Decided by `input_width`, `spike_width` and `SNN_EN` of   \
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
        InputWidthFormatType.WIDTH_1BIT,
        SpikeWidthFormatType.WIDTH_1BIT,
        SNNModeEnableType.DISABLE,
    )
    MODE_SNN = (
        InputWidthFormatType.WIDTH_1BIT,
        SpikeWidthFormatType.WIDTH_1BIT,
        SNNModeEnableType.ENABLE,
    )
    MODE_BANN_OR_SNN_TO_ANN = (
        InputWidthFormatType.WIDTH_1BIT,
        SpikeWidthFormatType.WIDTH_8BIT,
        SNNModeEnableType.DISABLE,
    )
    MODE_BANN_OR_SNN_TO_VSNN = (
        InputWidthFormatType.WIDTH_1BIT,
        SpikeWidthFormatType.WIDTH_8BIT,
        SNNModeEnableType.ENABLE,
    )
    MODE_ANN_TO_BANN_OR_SNN = (
        InputWidthFormatType.WIDTH_8BIT,
        SpikeWidthFormatType.WIDTH_1BIT,
        SNNModeEnableType.DISABLE,
    )
    MODE_ANN = (
        InputWidthFormatType.WIDTH_8BIT,
        SpikeWidthFormatType.WIDTH_8BIT,
        SNNModeEnableType.DISABLE,
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
    input_width: InputWidthFormatType,
    spike_width: SpikeWidthFormatType,
    snn_mode: SNNModeEnableType,
) -> CoreMode:
    try:
        return CoreMode((input_width, spike_width, snn_mode))
    except ValueError:
        raise ValueError(
            f"invalid mode conf: (input_width, spike_width, snn_mode) = "
            f"({input_width}, {spike_width}, {snn_mode}).",
        )


def core_mode_check(func):
    @wraps(func)
    def wrapper(reg_dict: dict[str, Any], *args, **kwargs):
        _ = get_core_mode(
            reg_dict["input_width"], reg_dict["spike_width"], reg_dict["snn_en"]
        )
        return func(reg_dict, *args, **kwargs)

    return wrapper
