import pytest
from pydantic import ValidationError

from paicorelib.core_defs import LCN_EX, WeightWidth
from paicorelib.core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    InputSignMode,
    OfflineCoreRegLimV2,
    OutputSignMode,
    PoolingMode,
    SNNMode,
    WeightSignMode,
    ZeroOutputMode,
)
from paicorelib.core_model_v2 import OfflineCoreRegV2


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
            "delay_cycle": 1,
            "width_cycle": 1,
            "tick_start": 0,
            "tick_duration": 100,
            "tick_initial": 0,
        }

    @pytest.mark.parametrize(
        "params_update",
        [
            {},  # Default legal params
            {
                "snn_ann": SNNMode.ANN,
                "max_pooling": PoolingMode.MAX,
                "add_potential": AddPotentialMode.DIRECT_ADD,
                "zero_output": ZeroOutputMode.ENABLE,
                "input_sign": InputSignMode.SIGNED,
                "input_width": WeightWidth.WEIGHT_WIDTH_8BIT,
                "output_sign": OutputSignMode.SIGNED,
                "output_width": WeightWidth.WEIGHT_WIDTH_8BIT,
                "weight_sign": WeightSignMode.SIGNED,
                "weight_width": WeightWidth.WEIGHT_WIDTH_8BIT,
                "lcn": LCN_EX.LCN_4X,
                "target_lcn": LCN_EX.LCN_4X,
                "axon_skew": OfflineCoreRegLimV2.AXON_SKEW_MAX,
                "neuron_number": OfflineCoreRegLimV2.NEURON_NUMBER_MAX,
                "test_core_xy": OfflineCoreRegLimV2.TEST_CORE_COORD_MAX,
                "test_core_x": OfflineCoreRegLimV2.TEST_CORE_COORD_MAX,
                "test_core_y": OfflineCoreRegLimV2.TEST_CORE_COORD_MAX,
                "global_send": OfflineCoreRegLimV2.GLOBAL_SEND_MAX,
                "csc_accelerate": CSCAccelerateMode.ENABLE,
                "global_receive": OfflineCoreRegLimV2.GLOBAL_RECEIVE_MAX,
                "thread_number": OfflineCoreRegLimV2.THREAD_NUMBER_MAX,
                "busy_cycle": OfflineCoreRegLimV2.BUSY_CYCLE_MAX,
                "delay_cycle": OfflineCoreRegLimV2.DELAY_CYCLE_MAX,
                "width_cycle": OfflineCoreRegLimV2.WIDTH_CYCLE_MAX,
                "tick_start": OfflineCoreRegLimV2.TICK_START_MAX,
                "tick_duration": OfflineCoreRegLimV2.TICK_DURATION_MAX,
                "tick_initial": OfflineCoreRegLimV2.TICK_INITIAL_MAX,
            },
        ],
    )
    def test_legal(self, ensure_dump_dir, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        core_reg = OfflineCoreRegV2.model_validate(params, strict=True)
        core_reg_dict = core_reg.model_dump_json(indent=2)

        with open(ensure_dump_dir / "core_reg.json", "w") as f:
            f.write(core_reg_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"axon_skew": OfflineCoreRegLimV2.AXON_SKEW_MIN - 1},
            {"axon_skew": OfflineCoreRegLimV2.AXON_SKEW_MAX + 1},
            {"neuron_number": OfflineCoreRegLimV2.NEURON_NUMBER_MAX + 1},
            {"test_core_xy": OfflineCoreRegLimV2.TEST_CORE_COORD_MIN - 1},
            {"test_core_xy": OfflineCoreRegLimV2.TEST_CORE_COORD_MAX + 1},
            {"global_send": OfflineCoreRegLimV2.GLOBAL_SEND_MAX + 1},
            {"global_receive": OfflineCoreRegLimV2.GLOBAL_RECEIVE_MAX + 1},
            {"thread_number": OfflineCoreRegLimV2.THREAD_NUMBER_MAX + 1},
            {"busy_cycle": OfflineCoreRegLimV2.BUSY_CYCLE_MAX + 1},
            {"delay_cycle": OfflineCoreRegLimV2.DELAY_CYCLE_MAX + 1},
            {"width_cycle": OfflineCoreRegLimV2.WIDTH_CYCLE_MAX + 1},
            {"tick_start": OfflineCoreRegLimV2.TICK_START_MAX + 1},
            {"tick_duration": OfflineCoreRegLimV2.TICK_DURATION_MAX + 1},
            {"tick_initial": OfflineCoreRegLimV2.TICK_INITIAL_MAX + 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises(ValidationError):
            OfflineCoreRegV2.model_validate(params, strict=True)
