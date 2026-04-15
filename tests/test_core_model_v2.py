import pytest
from pydantic import ValidationError

from paicorelib.core_defs import LCN_EX
from paicorelib.core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    OfflineCoreRegLimV2,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)
from paicorelib.core_model_v2 import OfflineCoreRegV2

from .utils import build_v2_core_reg_params


class TestCoreRegModel:
    @pytest.mark.parametrize(
        "params",
        [
            build_v2_core_reg_params(),
            build_v2_core_reg_params(
                snn_ann=SNNMode.ANN,
                max_pooling=PoolingMode.MAX,
                add_potential=AddPotentialMode.DIRECT_ADD,
                zero_output=ZeroOutputMode.ENABLE,
                input_sign=DataSign.SIGNED,
                input_width=DataWidth.WIDTH_8BIT,
                output_sign=DataSign.SIGNED,
                output_width=DataWidth.WIDTH_8BIT,
                weight_sign=DataSign.SIGNED,
                weight_width=DataWidth.WIDTH_8BIT,
                lcn=LCN_EX.LCN_4X,
                target_lcn=LCN_EX.LCN_4X,
                axon_skew=OfflineCoreRegLimV2.AXON_SKEW_MAX,
                neuron_number=OfflineCoreRegLimV2.NEURON_NUMBER_MAX,
                test_core_xy=OfflineCoreRegLimV2.TEST_CORE_COORD_MAX,
                test_core_x=OfflineCoreRegLimV2.TEST_CORE_COORD_MAX,
                test_core_y=OfflineCoreRegLimV2.TEST_CORE_COORD_MAX,
                global_send=OfflineCoreRegLimV2.GLOBAL_SEND_MAX,
                csc_accelerate=CSCAccelerateMode.ENABLE,
                global_receive=OfflineCoreRegLimV2.GLOBAL_RECEIVE_MAX,
                thread_number=OfflineCoreRegLimV2.THREAD_NUMBER_MAX,
                busy_cycle=OfflineCoreRegLimV2.BUSY_CYCLE_MAX,
                delay_cycle=OfflineCoreRegLimV2.DELAY_CYCLE_MAX,
                width_cycle=OfflineCoreRegLimV2.WIDTH_CYCLE_MAX,
                tick_start=OfflineCoreRegLimV2.TICK_START_MAX,
                tick_duration=OfflineCoreRegLimV2.TICK_DURATION_MAX,
                tick_initial=OfflineCoreRegLimV2.TICK_INITIAL_MAX,
            ),
        ],
    )
    def test_legal(self, ensure_dump_dir, params):
        core_reg = OfflineCoreRegV2.model_validate(params, strict=True)

        with open(ensure_dump_dir / "core_reg.json", "w") as f:
            f.write(core_reg.model_dump_json(indent=2))

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
    def test_illegal(self, params_update):
        with pytest.raises(ValidationError):
            OfflineCoreRegV2.model_validate(
                build_v2_core_reg_params(**params_update), strict=True
            )

    @pytest.mark.parametrize(
        "params_update",
        [
            {
                "max_pooling": PoolingMode.MAX,
                "weight_width": DataWidth.WIDTH_8BIT,
            },
            {
                "add_potential": AddPotentialMode.DIRECT_ADD,
                "weight_width": DataWidth.WIDTH_4BIT,
            },
        ],
    )
    def test_force_weight_width_to_1bit(self, params_update):
        core_reg = OfflineCoreRegV2.model_validate(
            build_v2_core_reg_params(**params_update), strict=True
        )

        assert core_reg.weight_width == DataWidth.WIDTH_1BIT

    def test_keep_weight_width_when_not_forced(self):
        core_reg = OfflineCoreRegV2.model_validate(
            build_v2_core_reg_params(weight_width=DataWidth.WIDTH_8BIT), strict=True
        )

        assert core_reg.weight_width == DataWidth.WIDTH_8BIT
