from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator
from .core_defs import (
    SNNMode,
    PoolingMode,
    PotentialAddMode,
    ZeroOutputMode,
    InputSignMode,
    InputWidth,
    OutputSignMode,
    OutputWidth,
    WeightSignMode,
    WeightWidth,
    LCNMode,
    CSCAccelerateMode,
    CoreLim,
)


__all__ = ["CoreReg"]


class CoreReg(BaseModel):
    """Core parameters model, corresponding to Section 2.3.1 Core Parameters."""

    model_config = ConfigDict(
        extra="ignore", validate_assignment=True, use_enum_values=True, strict=True
    )

    # Configuration Parameters
    snn_ann: Annotated[SNNMode, Field(
        description="SNN and ANN mode selection.")]
    max_pooling: Annotated[PoolingMode, Field(
        description="Pooling mode selection.")]
    add_potential: Annotated[PotentialAddMode, Field(
        description="Accumulation mode selection.")]
    zero_output: Annotated[ZeroOutputMode, Field(
        description="Whether to output zero values.")]

    input_sign: Annotated[InputSignMode, Field(
        description="Input data sign selection.")]
    input_width: Annotated[InputWidth, Field(
        description="Input data bit width selection.")]

    output_sign: Annotated[OutputSignMode, Field(
        description="Output data sign selection.")]
    output_width: Annotated[OutputWidth, Field(
        description="Output data bit width selection.")]

    weight_sign: Annotated[WeightSignMode, Field(
        description="Weight data sign selection.")]
    weight_width: Annotated[WeightWidth, Field(
        description="Weight data bit width selection.")]

    lcn: Annotated[LCNMode, Field(
        description="Control the scale of fan-in extension.")]
    target_lcn: Annotated[LCNMode, Field(
        description="LCN of the output target address core.")]

    axon_skew: Annotated[int, Field(
        ge=CoreLim.AXON_SKEW_MIN, le=CoreLim.AXON_SKEW_MAX, description="Axon address offset for AER format input work frame.")]
    neuron_number: Annotated[NonNegativeInt, Field(
        le=CoreLim.NEURON_NUMBER_MAX, description="Number of valid neuron addresses.")]

    # Test/Control Frame Addresses (Sign-Magnitude 6-bit: -31 to 31)
    test_core_xy: Annotated[int, Field(
        ge=CoreLim.TEST_CORE_OFFSET_MIN, le=CoreLim.TEST_CORE_OFFSET_MAX, description="Relative XY address of the core sent by test/control frame.")]
    test_core_x: Annotated[int, Field(
        ge=CoreLim.TEST_CORE_OFFSET_MIN, le=CoreLim.TEST_CORE_OFFSET_MAX, description="Relative X address of the core sent by test/control frame.")]
    test_core_y: Annotated[int, Field(
        ge=CoreLim.TEST_CORE_OFFSET_MIN, le=CoreLim.TEST_CORE_OFFSET_MAX, description="Relative Y address of the core sent by test/control frame.")]

    # Global Signals
    global_send: Annotated[NonNegativeInt, Field(
        le=CoreLim.GLOBAL_SEND_MAX, description="Global signal send direction (local, xy+, xy-, x+, x-, y+, y-).")]

    csc_accelerate: Annotated[CSCAccelerateMode, Field(
        description="CSC compressed calculation acceleration mode.")]

    # Global Signals
    global_receive: Annotated[NonNegativeInt, Field(
        le=CoreLim.GLOBAL_RECEIVE_MAX, description="Global signal receive direction (xy+, xy-, x+, x-, y+, y-).")]

    # Thread and Timing
    thread_number: Annotated[NonNegativeInt, Field(
        le=CoreLim.THREAD_NUMBER_MAX, description="Thread number of the current core.")]
    busy_cycle: Annotated[NonNegativeInt, Field(
        le=CoreLim.BUSY_CYCLE_MAX, description="Mask threshold for busy signal.")]
    delay_cycle: Annotated[NonNegativeInt, Field(
        le=CoreLim.DELAY_CYCLE_MAX, description="Delay time for control signal to take effect.")]
    width_cycle: Annotated[NonNegativeInt, Field(
        le=CoreLim.WIDTH_CYCLE_MAX, description="Multi-cycle width for control global signals sync_all, initial_all.")]

    # Tick Control
    tick_start: Annotated[NonNegativeInt, Field(
        le=CoreLim.TICK_START_MAX, description="Start tick count.")]
    tick_duration: Annotated[NonNegativeInt, Field(
        le=CoreLim.TICK_DURATION_MAX, description="Duration tick count.")]
    tick_initializer: Annotated[NonNegativeInt, Field(
        le=CoreLim.TICK_INITIALIZER_MAX, description="Auto-initialization tick count.")]

    @model_validator(mode="after")
    def check_weight_width(self) -> "CoreReg":
        if (
            self.max_pooling == PoolingMode.MAX
            or self.add_potential == PotentialAddMode.DIRECT_POTENTIAL
        ):
            self.weight_width = WeightWidth.WIDTH_1BIT
        return self
