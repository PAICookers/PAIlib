import sys
from enum import Enum, IntEnum, auto, unique
from functools import wraps
from typing import Any

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

__all__ = [
    "WeightPrecisionType",
    "LCNExtensionType",
    "InputWidthFormatType",
    "SpikeWidthFormatType",
    "MaxPoolingEnableType",
    "SNNModeEnableType",
    "CoreType",
    "CoreMode",
    "get_core_mode",
]

"""
    Type defines of parameters of registers & parameters of neuron RAM.
    See Section 2.4.1 in V2.1 Manual for details.
"""


@unique
class WeightPrecisionType(IntEnum):
    """Weight precision of crossbar. 2-bit.

    - `WEIGHT_WIDTH_XBIT` for X-bit. Default value is `WEIGHT_WIDTH_8BIT`.
    """

    WEIGHT_WIDTH_1BIT = 0
    WEIGHT_WIDTH_2BIT = 1
    WEIGHT_WIDTH_4BIT = 2
    WEIGHT_WIDTH_8BIT = 3  # Default value.


@unique
class LCNExtensionType(IntEnum):
    """Scale of Fan-in extension. 4-bit.

    - X-time LCN extension. Default value is `LCN_1X`.

    NOTE:
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

    - `MAX_POOLING_DISABLE`: pooling max disable. Default value.
    - `MAX_POOLING_ENABLE`: pooling max enable.
    """

    DISABLE = 0  # Default value.
    ENABLE = 1


@unique
class SNNModeEnableType(IntEnum):
    """Enable SNN mode or not. 1-bit.

    - `SNN_MODE_DISABLE`: SNN mode disable.
    - `SNN_MODE_ENABLE`: SNN mode enable. Default value.
    """

    DISABLE = 0
    ENABLE = 1  # Default value.


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
        ANN to BANN/SNN                 1               0      Don't care(0)
        ANN                             1               1      Don't care(0)
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
    MODE_BANN_OR_SNN_TO_SNN = (
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
        return self is CoreMode.MODE_SNN or self is CoreMode.MODE_BANN_OR_SNN_TO_SNN

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
