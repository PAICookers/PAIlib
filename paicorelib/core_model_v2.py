from typing import Annotated

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from .core_defs import LCN_EX  # Reuse some types
from .core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    OfflineCoreRegLimV2,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)
from .core_model import CoreReg

__all__ = ["OfflineCoreRegV2"]


class OfflineCoreRegV2(CoreReg):
    """Core parameters model, corresponding to Section 2.3.1 Core Parameters."""

    # Configuration Parameters
    snn_ann: Annotated[SNNMode, Field(description="SNN and ANN mode selection.")]
    max_pooling: Annotated[PoolingMode, Field(description="Pooling mode selection.")]
    add_potential: Annotated[
        AddPotentialMode, Field(description="Accumulation mode selection.")
    ]
    zero_output: Annotated[
        ZeroOutputMode, Field(description="Whether to output zero values.")
    ]

    input_sign: Annotated[DataSign, Field(description="Input data sign selection.")]
    input_width: Annotated[
        DataWidth, Field(description="Input data bit width selection.")
    ]

    output_sign: Annotated[DataSign, Field(description="Output data sign selection.")]
    output_width: Annotated[
        DataWidth, Field(description="Output data bit width selection.")
    ]

    weight_sign: Annotated[DataSign, Field(description="Weight data sign selection.")]
    weight_width: Annotated[
        DataWidth, Field(description="Weight data bit width selection.")
    ]

    lcn: Annotated[LCN_EX, Field(description="Control the scale of fan-in extension.")]
    target_lcn: Annotated[
        LCN_EX, Field(description="LCN of the output target address core.")
    ]

    axon_skew: Annotated[
        int,
        Field(
            ge=OfflineCoreRegLimV2.AXON_SKEW_MIN,
            le=OfflineCoreRegLimV2.AXON_SKEW_MAX,
            description="Axon address offset for AER format input work frame.",
        ),
    ]
    neuron_number: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineCoreRegLimV2.NEURON_NUMBER_MAX,
            description="Number of valid neuron addresses.",
        ),
    ]

    # Test/Control Frame Addresses (Sign-Magnitude 6-bit: -31 to 31)
    test_core_xy: Annotated[
        int,
        Field(
            ge=OfflineCoreRegLimV2.TEST_CORE_COORD_MIN,
            le=OfflineCoreRegLimV2.TEST_CORE_COORD_MAX,
            description="Relative XY address of the core sent by test/control frame.",
        ),
    ]
    test_core_x: Annotated[
        int,
        Field(
            ge=OfflineCoreRegLimV2.TEST_CORE_COORD_MIN,
            le=OfflineCoreRegLimV2.TEST_CORE_COORD_MAX,
            description="Relative X address of the core sent by test/control frame.",
        ),
    ]
    test_core_y: Annotated[
        int,
        Field(
            ge=OfflineCoreRegLimV2.TEST_CORE_COORD_MIN,
            le=OfflineCoreRegLimV2.TEST_CORE_COORD_MAX,
            description="Relative Y address of the core sent by test/control frame.",
        ),
    ]

    # Global Signals
    global_send: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineCoreRegLimV2.GLOBAL_SEND_MAX,
            description="Global signal send direction (local, xy+, xy-, x+, x-, y+, y-).",
        ),
    ]

    csc_accelerate: Annotated[
        CSCAccelerateMode,
        Field(description="CSC compressed calculation acceleration mode."),
    ]

    # Global Signals
    global_receive: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineCoreRegLimV2.GLOBAL_RECEIVE_MAX,
            description="Global signal receive direction (xy+, xy-, x+, x-, y+, y-).",
        ),
    ]

    # Thread and Timing
    thread_number: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineCoreRegLimV2.THREAD_NUMBER_MAX,
            description="Thread number of the current core.",
        ),
    ]
    busy_cycle: Annotated[
        PositiveInt,
        Field(
            gt=1,
            le=OfflineCoreRegLimV2.BUSY_CYCLE_MAX,
            description="Mask threshold for busy signal, usually >1.",
        ),
    ]
    delay_cycle: Annotated[
        PositiveInt,
        Field(
            gt=1,
            le=OfflineCoreRegLimV2.DELAY_CYCLE_MAX,
            description="Delay time for control signal to take effect, usually >1.",
        ),
    ]
    width_cycle: Annotated[
        PositiveInt,
        Field(
            gt=1,
            le=OfflineCoreRegLimV2.WIDTH_CYCLE_MAX,
            description="Multi-cycle width for control global signals sync_all, initial_all, usually >1.",
        ),
    ]

    # Tick Control
    tick_start: Annotated[
        NonNegativeInt,
        Field(le=OfflineCoreRegLimV2.TICK_START_MAX, description="Start tick count."),
    ]
    tick_duration: Annotated[
        NonNegativeInt,
        Field(
            default=0,
            le=OfflineCoreRegLimV2.TICK_DURATION_MAX,
            description="Duration tick count.",
        ),
    ] = 0
    tick_initial: Annotated[
        NonNegativeInt,
        Field(
            default=0,
            le=OfflineCoreRegLimV2.TICK_INITIAL_MAX,
            description="Auto-initialization tick count.",
        ),
    ] = 0

    @model_validator(mode="after")
    def check_weight_width(self):
        if (
            self.max_pooling == PoolingMode.MAX
            or self.add_potential == AddPotentialMode.DIRECT_ADD
        ):
            if self.weight_width != DataWidth.WIDTH_1BIT:
                self.weight_width = DataWidth.WIDTH_1BIT

        return self
