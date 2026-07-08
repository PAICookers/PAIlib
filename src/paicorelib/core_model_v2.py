import warnings
from typing import Annotated

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from .core_defs import LCN_EX  # Reuse some types
from .core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    InputCoreType,
    OnlineCoreRegLimV2,
    OnlineCoreUpdateType,
    OnlineCoreWorkMode,
    OnlineDataWidth,
    OnlineSNNMode,
    OutputCoreType,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
    _CommonCoreRegLimV2,
)
from .core_model import CoreReg
from .float_codec import BF16Param

__all__ = ["OfflineCoreRegV2", "OnlineCoreRegV2"]


class _CommonCoreRegV2(CoreReg):
    """Common configurable core parameters shared by offline and online v2 cores."""

    max_pooling: Annotated[PoolingMode, Field(description="Pooling mode.")]
    add_potential: Annotated[AddPotentialMode, Field(description="Accumulation mode.")]
    zero_output: Annotated[
        ZeroOutputMode, Field(description="Whether to output zero values.")
    ]

    # Neuron Parameters
    axon_skew: Annotated[
        int,
        Field(
            ge=_CommonCoreRegLimV2.AXON_SKEW_MIN,
            le=_CommonCoreRegLimV2.AXON_SKEW_MAX,
            description="Axon address offset.",
        ),
    ]
    neuron_number: Annotated[
        NonNegativeInt,
        Field(
            le=_CommonCoreRegLimV2.NEURON_NUMBER_MAX,
            description="Number of valid neuron addresses.",
        ),
    ]

    # Test/Control Frame Addresses (Sign-Magnitude 6-bit: -31 to 31)
    test_core_xy: Annotated[
        int,
        Field(
            ge=_CommonCoreRegLimV2.TEST_CORE_COORD_MIN,
            le=_CommonCoreRegLimV2.TEST_CORE_COORD_MAX,
            description="Relative XY address of the test/control target core.",
        ),
    ]
    test_core_x: Annotated[
        int,
        Field(
            ge=_CommonCoreRegLimV2.TEST_CORE_COORD_MIN,
            le=_CommonCoreRegLimV2.TEST_CORE_COORD_MAX,
            description="Relative X address of the test/control target core.",
        ),
    ]
    test_core_y: Annotated[
        int,
        Field(
            ge=_CommonCoreRegLimV2.TEST_CORE_COORD_MIN,
            le=_CommonCoreRegLimV2.TEST_CORE_COORD_MAX,
            description="Relative Y address of the test/control target core.",
        ),
    ]

    # Global Signals
    global_send: Annotated[
        NonNegativeInt,
        Field(
            le=_CommonCoreRegLimV2.GLOBAL_SEND_MAX,
            description="Global signal send direction (local, xy+, xy-, x+, x-, y+, y-).",
        ),
    ]
    csc_accelerate: Annotated[
        CSCAccelerateMode,
        Field(description="Mode of CSC compressed calculation acceleration."),
    ]
    global_receive: Annotated[
        NonNegativeInt,
        Field(
            le=_CommonCoreRegLimV2.GLOBAL_RECEIVE_MAX,
            description="Global signal receive direction (xy+, xy-, x+, x-, y+, y-).",
        ),
    ]

    thread_number: Annotated[
        NonNegativeInt,
        Field(
            le=_CommonCoreRegLimV2.THREAD_NUMBER_MAX,
            description="Thread number of the current core.",
        ),
    ]
    busy_cycle: Annotated[
        PositiveInt,
        Field(
            gt=1,
            le=_CommonCoreRegLimV2.BUSY_CYCLE_MAX,
            description="Busy signal threshold.",
        ),
    ]
    delay_cycle: Annotated[
        PositiveInt,
        Field(
            gt=1,
            le=_CommonCoreRegLimV2.DELAY_CYCLE_MAX,
            description="Control signal delay.",
        ),
    ]
    width_cycle: Annotated[
        PositiveInt,
        Field(
            gt=1,
            le=_CommonCoreRegLimV2.WIDTH_CYCLE_MAX,
            description="Control signal width.",
        ),
    ]

    # Tick Control
    tick_start: Annotated[
        NonNegativeInt,
        Field(
            le=_CommonCoreRegLimV2.TICK_START_MAX,
            description="Start tick count. 0 for never.",
        ),
    ]
    tick_duration: Annotated[
        NonNegativeInt,
        Field(
            default=0,
            le=_CommonCoreRegLimV2.TICK_DURATION_MAX,
            description="Duration tick count. 0 for ever.",
        ),
    ] = 0
    tick_initial: Annotated[
        NonNegativeInt,
        Field(
            default=0,
            le=_CommonCoreRegLimV2.TICK_INITIAL_MAX,
            description="Auto-initialization tick count. 0 for never.",
        ),
    ] = 0


class OfflineCoreRegV2(_CommonCoreRegV2):
    """Core parameters model, corresponding to Section 2.3.1 Core Parameters."""

    # Configuration Parameters
    snn_ann: Annotated[SNNMode, Field(description="SNN and ANN mode.")]

    input_sign: Annotated[DataSign, Field(description="Input data sign.")]
    input_width: Annotated[DataWidth, Field(description="Input data width.")]

    output_sign: Annotated[DataSign, Field(description="Output data sign.")]
    output_width: Annotated[DataWidth, Field(description="Output data width.")]

    weight_sign: Annotated[DataSign, Field(description="Weight data sign.")]
    weight_width: Annotated[DataWidth, Field(description="Weight bit width.")]

    lcn: Annotated[LCN_EX, Field(description="Control the scale of fan-in extension.")]
    target_lcn: Annotated[LCN_EX, Field(description="LCN of the output target core.")]

    @model_validator(mode="after")
    def check_weight_width(self):
        if (
            self.max_pooling == PoolingMode.MAX
            or self.add_potential == AddPotentialMode.DIRECT_ADD
        ):
            if self.weight_width != DataWidth.WIDTH_1BIT:
                warnings.warn(
                    "'weight_width' is forced to WIDTH_1BIT when 'max_pooling' "
                    "is MAX or 'add_potential' is DIRECT_ADD.",
                    UserWarning,
                    stacklevel=2,
                )
                self.weight_width = DataWidth.WIDTH_1BIT

        return self


class OnlineCoreRegV2(_CommonCoreRegV2):
    """Online core parameters model, corresponding to Section 3.3.1 Core Parameters."""

    # Configuration Parameters
    snn_ann: Annotated[OnlineSNNMode, Field(description="SNN and ANN mode.")]
    work_mode: Annotated[OnlineCoreWorkMode, Field(description="Core work mode.")]

    # Input/Output Selection
    input_core: Annotated[
        InputCoreType, Field(description="The type of the source input core.")
    ]
    input_width: Annotated[OnlineDataWidth, Field(description="Input data width.")]
    output_core: Annotated[
        OutputCoreType, Field(description="The type of the target output core.")
    ]
    output_width: Annotated[
        OnlineDataWidth | OnlineCoreUpdateType,
        Field(
            description=(
                "Forward/gradient output data width. In update work modes, "
                "the same bits encode OnlineCoreUpdateType."
            ),
        ),
    ]

    # LCN Configuration
    lcn_at: Annotated[LCN_EX, Field(description="Activation LCN.")]
    lcn_mp: Annotated[LCN_EX, Field(description="Membrane LCN.")]
    lcn_lg: Annotated[LCN_EX, Field(description="Gradient LCN.")]
    target_lcn_at: Annotated[LCN_EX, Field(description="Target activation LCN.")]
    target_lcn_mp: Annotated[LCN_EX, Field(description="Target membrane LCN.")]
    target_lcn_lg: Annotated[LCN_EX, Field(description="Target gradient LCN.")]

    # Neuron Parameters
    update_number: Annotated[
        NonNegativeInt,
        Field(
            le=OnlineCoreRegLimV2.UPDATE_NUMBER_MAX,
            description="Number of neurons to update.",
        ),
    ]

    # BF16 Coefficients
    scale_in: Annotated[BF16Param, Field(description="Input scale coefficient, bf16.")]
    bias_in: Annotated[BF16Param, Field(description="Input bias coefficient, bf16.")]
    scale_out: Annotated[
        BF16Param, Field(description="Output scale coefficient, bf16.")
    ]
    bias_out: Annotated[BF16Param, Field(description="Output bias coefficient, bf16.")]
    learning_rate: Annotated[BF16Param, Field(description="Learning rate, bf16.")]

    # Update/Test Frame Addresses (Sign-Magnitude 6-bit: -31 to 31)
    update_core_xy: Annotated[
        int,
        Field(
            ge=OnlineCoreRegLimV2.TEST_CORE_COORD_MIN,
            le=OnlineCoreRegLimV2.TEST_CORE_COORD_MAX,
            description="Relative XY address of the updated core.",
        ),
    ]
    update_core_x: Annotated[
        int,
        Field(
            ge=OnlineCoreRegLimV2.TEST_CORE_COORD_MIN,
            le=OnlineCoreRegLimV2.TEST_CORE_COORD_MAX,
            description="Relative X address of the updated core.",
        ),
    ]
    update_core_y: Annotated[
        int,
        Field(
            ge=OnlineCoreRegLimV2.TEST_CORE_COORD_MIN,
            le=OnlineCoreRegLimV2.TEST_CORE_COORD_MAX,
            description="Relative Y address of the updated core.",
        ),
    ]
