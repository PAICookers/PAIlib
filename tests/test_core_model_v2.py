import warnings
from collections.abc import Callable
from enum import IntEnum
from typing import Any, get_args

import numpy as np
import pytest
from pydantic import ValidationError

import paicorelib
from paicorelib import core_model_v2 as core_models_v2
from paicorelib.core_defs import LCN_EX, CoreType
from paicorelib.core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    InputCoreType,
    OfflineCoreRegLimV2,
    OnlineCoreRegLimV2,
    OnlineCoreUpdateType,
    OnlineCoreWorkMode,
    OnlineDataWidth,
    OnlineSNNMode,
    OutputCoreType,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)
from paicorelib.core_model_v2 import OfflineCoreRegV2, OnlineCoreRegV2
from paicorelib.float_codec import cast_bf16_scalar

from .utils import build_online_v2_core_reg_params, build_v2_core_reg_params

CoreRegBuilder = Callable[..., dict[str, Any]]
CoreRegModel = type[OfflineCoreRegV2] | type[OnlineCoreRegV2]

COMMON_CORE_REG_V2_FIELDS = {
    "max_pooling",
    "add_potential",
    "zero_output",
    "axon_skew",
    "neuron_number",
    "test_core_xy",
    "test_core_x",
    "test_core_y",
    "global_send",
    "csc_accelerate",
    "global_receive",
    "thread_number",
    "busy_cycle",
    "delay_cycle",
    "width_cycle",
    "tick_start",
    "tick_duration",
    "tick_initial",
}

PROTOCOL_SPECIFIC_CORE_REG_V2_FIELDS = {"snn_ann", "input_width", "output_width"}

OFFLINE_ONLY_CORE_REG_V2_FIELDS = {
    "input_sign",
    "output_sign",
    "weight_sign",
    "weight_width",
    "lcn",
    "target_lcn",
}

ONLINE_ONLY_CORE_REG_V2_FIELDS = {
    "work_mode",
    "input_core",
    "output_core",
    "lcn_at",
    "lcn_mp",
    "lcn_lg",
    "target_lcn_at",
    "target_lcn_mp",
    "target_lcn_lg",
    "update_number",
    "scale_in",
    "bias_in",
    "scale_out",
    "bias_out",
    "learning_rate",
    "update_core_xy",
    "update_core_x",
    "update_core_y",
}

CORE_REG_CASES = (
    pytest.param(
        OfflineCoreRegV2, build_v2_core_reg_params, OfflineCoreRegLimV2, id="offline"
    ),
    pytest.param(
        OnlineCoreRegV2,
        build_online_v2_core_reg_params,
        OnlineCoreRegLimV2,
        id="online",
    ),
)

COMMON_LIMIT_CASES = (
    pytest.param("axon_skew", lambda lim: lim.AXON_SKEW_MIN - 1, id="axon-skew-low"),
    pytest.param("axon_skew", lambda lim: lim.AXON_SKEW_MAX + 1, id="axon-skew-high"),
    pytest.param(
        "neuron_number",
        lambda lim: lim.NEURON_NUMBER_MAX + 1,
        id="neuron-number-high",
    ),
    pytest.param(
        "test_core_xy",
        lambda lim: lim.TEST_CORE_COORD_MIN - 1,
        id="test-core-low",
    ),
    pytest.param(
        "test_core_xy",
        lambda lim: lim.TEST_CORE_COORD_MAX + 1,
        id="test-core-high",
    ),
    pytest.param(
        "global_send", lambda lim: lim.GLOBAL_SEND_MAX + 1, id="global-send-high"
    ),
    pytest.param(
        "global_receive",
        lambda lim: lim.GLOBAL_RECEIVE_MAX + 1,
        id="global-receive-high",
    ),
    pytest.param(
        "thread_number",
        lambda lim: lim.THREAD_NUMBER_MAX + 1,
        id="thread-number-high",
    ),
    pytest.param("busy_cycle", lambda _lim: 1, id="busy-cycle-too-small"),
    pytest.param(
        "busy_cycle", lambda lim: lim.BUSY_CYCLE_MAX + 1, id="busy-cycle-high"
    ),
    pytest.param(
        "delay_cycle", lambda lim: lim.DELAY_CYCLE_MAX + 1, id="delay-cycle-high"
    ),
    pytest.param(
        "width_cycle", lambda lim: lim.WIDTH_CYCLE_MAX + 1, id="width-cycle-high"
    ),
    pytest.param(
        "tick_start", lambda lim: lim.TICK_START_MAX + 1, id="tick-start-high"
    ),
    pytest.param(
        "tick_duration", lambda lim: lim.TICK_DURATION_MAX + 1, id="tick-duration-high"
    ),
    pytest.param(
        "tick_initial", lambda lim: lim.TICK_INITIAL_MAX + 1, id="tick-initial-high"
    ),
)


def validate_without_warning(
    model: CoreRegModel,
    params: dict[str, Any],
) -> OfflineCoreRegV2 | OnlineCoreRegV2:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        core_reg = model.model_validate(params, strict=True)

    assert captured == []
    assert isinstance(core_reg, (OfflineCoreRegV2, OnlineCoreRegV2))
    return core_reg


def validate_offline_without_warning(params: dict[str, Any]) -> OfflineCoreRegV2:
    core_reg = validate_without_warning(OfflineCoreRegV2, params)
    assert isinstance(core_reg, OfflineCoreRegV2)
    return core_reg


class TestCoreRegV2FieldLayout:
    def test_common_fields_live_on_shared_base(self):
        common_fields = set(core_models_v2._CommonCoreRegV2.model_fields)

        assert common_fields == {"name"} | COMMON_CORE_REG_V2_FIELDS
        assert COMMON_CORE_REG_V2_FIELDS <= set(OfflineCoreRegV2.model_fields)
        assert COMMON_CORE_REG_V2_FIELDS <= set(OnlineCoreRegV2.model_fields)
        assert PROTOCOL_SPECIFIC_CORE_REG_V2_FIELDS.isdisjoint(common_fields)
        assert OFFLINE_ONLY_CORE_REG_V2_FIELDS.isdisjoint(common_fields)
        assert ONLINE_ONLY_CORE_REG_V2_FIELDS.isdisjoint(common_fields)

    def test_protocol_specific_field_types_stay_separate(self):
        assert OfflineCoreRegV2.model_fields["snn_ann"].annotation is SNNMode
        assert OnlineCoreRegV2.model_fields["snn_ann"].annotation is OnlineSNNMode
        assert OfflineCoreRegV2.model_fields["input_width"].annotation is DataWidth
        assert OnlineCoreRegV2.model_fields["input_width"].annotation is OnlineDataWidth
        assert OfflineCoreRegV2.model_fields["output_width"].annotation is DataWidth
        assert get_args(OnlineCoreRegV2.model_fields["output_width"].annotation) == (
            OnlineDataWidth,
            OnlineCoreUpdateType,
        )

    def test_compatibility_aliases_remain_exported(self):
        assert issubclass(CoreType, IntEnum)
        assert CoreType.OFFLINE.value == 0
        assert CoreType.ONLINE.value == 1
        assert InputCoreType is CoreType
        assert OutputCoreType is CoreType
        assert paicorelib.InputCoreType is CoreType
        assert paicorelib.OutputCoreType is CoreType
        assert not hasattr(paicorelib, "OnlineCoreType")

    @pytest.mark.parametrize(
        ("old_member", "new_member", "old_name"),
        [
            pytest.param(
                OnlineDataWidth.TYPE_1BIT,
                OnlineDataWidth.WIDTH_1BIT,
                "TYPE_1BIT",
                id="data-width-1bit",
            ),
            pytest.param(
                OnlineDataWidth.TYPE_FP16,
                OnlineDataWidth.WIDTH_FP16,
                "TYPE_FP16",
                id="data-width-fp16",
            ),
            pytest.param(
                OnlineDataWidth.TYPE_UINT8,
                OnlineDataWidth.WIDTH_UINT8,
                "TYPE_UINT8",
                id="data-width-uint8",
            ),
            pytest.param(
                OnlineDataWidth.TYPE_INT8,
                OnlineDataWidth.WIDTH_INT8,
                "TYPE_INT8",
                id="data-width-int8",
            ),
            pytest.param(
                OnlineCoreUpdateType.WEIGHT_BIAS,
                OnlineCoreUpdateType.WEIGHT_AND_BIAS,
                "WEIGHT_BIAS",
                id="update-weight-bias",
            ),
            pytest.param(
                OnlineCoreUpdateType.KAHAN_WEIGHT_BIAS,
                OnlineCoreUpdateType.KAHAN_WEIGHT_AND_BIAS,
                "KAHAN_WEIGHT_BIAS",
                id="update-kahan-weight-bias",
            ),
        ],
    )
    def test_renamed_enum_members_keep_compatibility_aliases(
        self, old_member, new_member, old_name
    ):
        assert old_member is new_member
        assert type(new_member)[old_name] is new_member
        assert old_name in type(new_member).__members__


class TestCoreRegV2CommonModel:
    @pytest.mark.parametrize(("model", "builder", "limits"), CORE_REG_CASES)
    @pytest.mark.parametrize(("field", "value_factory"), COMMON_LIMIT_CASES)
    def test_common_limits_reject_out_of_range_values(
        self,
        model: CoreRegModel,
        builder: CoreRegBuilder,
        limits: type[OfflineCoreRegLimV2] | type[OnlineCoreRegLimV2],
        field: str,
        value_factory: Callable[[Any], int],
    ):
        with pytest.raises(ValidationError):
            model.model_validate(builder(**{field: value_factory(limits)}), strict=True)

    @pytest.mark.parametrize(
        ("model", "builder"),
        (
            pytest.param(OfflineCoreRegV2, build_v2_core_reg_params, id="offline"),
            pytest.param(OnlineCoreRegV2, build_online_v2_core_reg_params, id="online"),
        ),
    )
    def test_tick_defaults_to_zero_when_omitted(
        self, model: CoreRegModel, builder: CoreRegBuilder
    ):
        params = builder()
        params.pop("tick_duration")
        params.pop("tick_initial")

        core_reg = validate_without_warning(model, params)

        assert core_reg.tick_duration == 0
        assert core_reg.tick_initial == 0


class TestOfflineCoreRegModel:
    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param({}, id="defaults"),
            pytest.param(
                {
                    "snn_ann": SNNMode.ANN,
                    "zero_output": ZeroOutputMode.ENABLE,
                    "input_sign": DataSign.SIGNED,
                    "input_width": DataWidth.WIDTH_8BIT,
                    "output_sign": DataSign.SIGNED,
                    "output_width": DataWidth.WIDTH_8BIT,
                    "weight_sign": DataSign.SIGNED,
                    "weight_width": DataWidth.WIDTH_8BIT,
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
                id="boundaries",
            ),
        ],
    )
    def test_accepts_legal_values_without_weight_width_rewrite(self, params_update):
        core_reg = validate_offline_without_warning(
            build_v2_core_reg_params(**params_update),
        )

        assert core_reg.weight_width == params_update.get(
            "weight_width", DataWidth.WIDTH_1BIT
        )

    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param(
                {"max_pooling": PoolingMode.MAX, "weight_width": DataWidth.WIDTH_8BIT},
                id="max-pooling",
            ),
            pytest.param(
                {
                    "add_potential": AddPotentialMode.DIRECT_ADD,
                    "weight_width": DataWidth.WIDTH_4BIT,
                },
                id="direct-add",
            ),
        ],
    )
    def test_force_weight_width_to_1bit_with_warning(self, params_update):
        with pytest.warns(UserWarning, match="weight_width"):
            core_reg = OfflineCoreRegV2.model_validate(
                build_v2_core_reg_params(**params_update),
                strict=True,
            )

        assert core_reg.weight_width == DataWidth.WIDTH_1BIT

    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param(
                {"max_pooling": PoolingMode.MAX, "weight_width": DataWidth.WIDTH_1BIT},
                id="max-pooling-already-1bit",
            ),
            pytest.param(
                {
                    "add_potential": AddPotentialMode.DIRECT_ADD,
                    "weight_width": DataWidth.WIDTH_1BIT,
                },
                id="direct-add-already-1bit",
            ),
        ],
    )
    def test_does_not_warn_when_forced_weight_width_is_already_1bit(
        self, params_update
    ):
        core_reg = validate_offline_without_warning(
            build_v2_core_reg_params(**params_update),
        )

        assert core_reg.weight_width == DataWidth.WIDTH_1BIT


class TestOnlineCoreRegModel:
    @pytest.mark.parametrize(
        ("output_width", "expected_value"),
        [
            pytest.param(
                OnlineDataWidth.WIDTH_FP16,
                OnlineDataWidth.WIDTH_FP16.value,
                id="data-width",
            ),
            pytest.param(
                OnlineCoreUpdateType.KAHAN_WEIGHT_AND_BIAS,
                OnlineCoreUpdateType.KAHAN_WEIGHT_AND_BIAS.value,
                id="update-type",
            ),
        ],
    )
    def test_accepts_documented_output_width_modes(self, output_width, expected_value):
        core_reg = OnlineCoreRegV2.model_validate(
            build_online_v2_core_reg_params(
                snn_ann=OnlineSNNMode.ANN_LUT,
                work_mode=OnlineCoreWorkMode.BACKWARD_WEIGHT_UPDATE,
                input_core=InputCoreType.OFFLINE,
                input_width=OnlineDataWidth.WIDTH_FP16,
                output_core=OutputCoreType.ONLINE,
                output_width=output_width,
            ),
            strict=True,
        )

        assert core_reg.snn_ann == OnlineSNNMode.ANN_LUT
        assert core_reg.work_mode == OnlineCoreWorkMode.BACKWARD_WEIGHT_UPDATE
        assert core_reg.input_core == InputCoreType.OFFLINE
        assert core_reg.input_width == OnlineDataWidth.WIDTH_FP16
        assert core_reg.output_core == OutputCoreType.ONLINE
        assert core_reg.output_width == expected_value

    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param({"snn_ann": SNNMode.SNN}, id="legacy-snn-enum"),
            pytest.param({"input_core": DataWidth.WIDTH_1BIT}, id="wrong-core-enum"),
            pytest.param({"input_width": DataWidth.WIDTH_1BIT}, id="wrong-width-enum"),
            pytest.param(
                {"output_width": DataWidth.WIDTH_1BIT}, id="wrong-output-enum"
            ),
            pytest.param({"output_width": 1}, id="raw-output-int"),
            pytest.param(
                {"update_number": OnlineCoreRegLimV2.UPDATE_NUMBER_MAX + 1},
                id="update-number-high",
            ),
        ],
    )
    def test_rejects_protocol_mismatches_and_online_only_limits(self, params_update):
        with pytest.raises(ValidationError):
            OnlineCoreRegV2.model_validate(
                build_online_v2_core_reg_params(**params_update),
                strict=True,
            )

    def test_preserves_bf16_fp32_carriers(self):
        scale_in = np.float32(0.1)
        bias_in = np.float32(-0.2)
        scale_out = np.float32(1.0 / 3.0)
        bias_out = np.float32(-1.0 / 7.0)
        learning_rate = np.float32(0.0156251)
        params = build_online_v2_core_reg_params(
            scale_in=scale_in,
            bias_in=bias_in,
            scale_out=scale_out,
            bias_out=bias_out,
            learning_rate=learning_rate,
        )
        core_reg = OnlineCoreRegV2.model_validate(params, strict=True)

        assert core_reg.scale_in == cast_bf16_scalar(scale_in)
        assert core_reg.bias_in == cast_bf16_scalar(bias_in)
        assert core_reg.scale_out == cast_bf16_scalar(scale_out)
        assert core_reg.bias_out == cast_bf16_scalar(bias_out)
        assert core_reg.learning_rate == cast_bf16_scalar(learning_rate)
