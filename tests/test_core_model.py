import json

import pytest
from pydantic import ValidationError

from paicorelib.core_defs import (
    AddPotentialMode,
    CoreLim,
    CSCAccelerateMode,
    InputSignMode,
    OutputSignMode,
    PoolingMode,
    SNNMode,
    WeightSignMode,
    ZeroOutputMode,
)
from paicorelib.core_model import OfflineCoreReg2_5
from paicorelib.reg_defs import LCN_EX, WeightWidth


class TestCoreRegModel:
    @pytest.fixture
    def default_params(self):
        return {
            "snn_ann": SNNMode.SNN,
            "max_pooling": PoolingMode.AVERAGE,
            "add_potential": AddPotentialMode.NORMAL,
            "zero_output": ZeroOutputMode.DISABLE,
            "input_sign": InputSignMode.UNSIGNED,
            "input_width": WeightWidth.WEIGHT_WIDTH_1BIT,
            "output_sign": OutputSignMode.UNSIGNED,
            "output_width": WeightWidth.WEIGHT_WIDTH_1BIT,
            "weight_sign": WeightSignMode.UNSIGNED,
            "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
            "lcn": LCN_EX.LCN_1X,
            "target_lcn": LCN_EX.LCN_1X,
            "axon_skew": 0,
            "neuron_number": 100,
            "test_core_xy": 0,
            "test_core_x": 0,
            "test_core_y": 0,
            "global_send": 0,
            "csc_accelerate": CSCAccelerateMode.DISABLE,
            "global_receive": 0,
            "thread_number": 1,
            "busy_cycle": 10,
            "delay_cycle": 0,
            "width_cycle": 1,
            "tick_start": 0,
            "tick_duration": 100,
            "tick_initializer": 0,
        }

    @pytest.mark.parametrize(
        "params_update",
        [
            {},  # Default legal params
            {
                "snn_ann": SNNMode.ANN,
                "max_pooling": PoolingMode.MAX,
                "add_potential": AddPotentialMode.DIRECT_POTENTIAL,
                "zero_output": ZeroOutputMode.ENABLE,
                "input_sign": InputSignMode.SIGNED,
                "input_width": WeightWidth.WEIGHT_WIDTH_8BIT,
                "output_sign": OutputSignMode.SIGNED,
                "output_width": WeightWidth.WEIGHT_WIDTH_8BIT,
                "weight_sign": WeightSignMode.SIGNED,
                "weight_width": WeightWidth.WEIGHT_WIDTH_8BIT,
                "lcn": LCN_EX.LCN_4X,
                "target_lcn": LCN_EX.LCN_4X,
                "axon_skew": CoreLim.AXON_SKEW_MAX,
                "neuron_number": CoreLim.NEURON_NUMBER_MAX,
                "test_core_xy": CoreLim.TEST_CORE_OFFSET_MAX,
                "test_core_x": CoreLim.TEST_CORE_OFFSET_MAX,
                "test_core_y": CoreLim.TEST_CORE_OFFSET_MAX,
                "global_send": CoreLim.GLOBAL_SEND_MAX,
                "csc_accelerate": CSCAccelerateMode.ENABLE,
                "global_receive": CoreLim.GLOBAL_RECEIVE_MAX,
                "thread_number": CoreLim.THREAD_NUMBER_MAX,
                "busy_cycle": CoreLim.BUSY_CYCLE_MAX,
                "delay_cycle": CoreLim.DELAY_CYCLE_MAX,
                "width_cycle": CoreLim.WIDTH_CYCLE_MAX,
                "tick_start": CoreLim.TICK_START_MAX,
                "tick_duration": CoreLim.TICK_DURATION_MAX,
                "tick_initializer": CoreLim.TICK_INITIALIZER_MAX,
            },
        ],
    )
    def test_legal(self, ensure_dump_dir, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        core_reg = OfflineCoreReg2_5.model_validate(params, strict=True)
        core_reg_dict = core_reg.model_dump_json(indent=2)

        with open(ensure_dump_dir / "core_reg.json", "w") as f:
            f.write(core_reg_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"axon_skew": CoreLim.AXON_SKEW_MIN - 1},
            {"axon_skew": CoreLim.AXON_SKEW_MAX + 1},
            {"neuron_number": CoreLim.NEURON_NUMBER_MAX + 1},
            {"test_core_xy": CoreLim.TEST_CORE_OFFSET_MIN - 1},
            {"test_core_xy": CoreLim.TEST_CORE_OFFSET_MAX + 1},
            {"global_send": CoreLim.GLOBAL_SEND_MAX + 1},
            {"global_receive": CoreLim.GLOBAL_RECEIVE_MAX + 1},
            {"thread_number": CoreLim.THREAD_NUMBER_MAX + 1},
            {"busy_cycle": CoreLim.BUSY_CYCLE_MAX + 1},
            {"delay_cycle": CoreLim.DELAY_CYCLE_MAX + 1},
            {"width_cycle": CoreLim.WIDTH_CYCLE_MAX + 1},
            {"tick_start": CoreLim.TICK_START_MAX + 1},
            {"tick_duration": CoreLim.TICK_DURATION_MAX + 1},
            {"tick_initializer": CoreLim.TICK_INITIALIZER_MAX + 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises(ValidationError):
            OfflineCoreReg2_5.model_validate(params, strict=True)
