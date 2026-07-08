import json

import numpy as np
import pytest
from pydantic import ValidationError

from paicorelib.coordinate import Coord
from paicorelib.core_defs import (
    LCN_EX,
    DecayRandomEnable,
    InputWidthFormat,
    LeakOrder,
    LUTRandomEnable,
    MaxPoolingEnable,
    OnlineModeEnable,
    SNNModeEnable,
    SpikeWidthFormat,
    WeightWidth,
)
from paicorelib.core_model import OfflineCoreReg, OnlineCoreReg


class TestOfflineCoreRegModel:
    @pytest.mark.parametrize(
        "coord, params",
        [
            (
                Coord(0, 0),
                {
                    "name": "Core0",
                    "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
                    "lcn": LCN_EX.LCN_2X,
                    "input_width": InputWidthFormat.WIDTH_1BIT,
                    "spike_width": SpikeWidthFormat.WIDTH_1BIT,
                    "num_dendrite": 100,
                    "max_pooling_en": MaxPoolingEnable.DISABLE,
                    "tick_wait_start": 0,
                    "tick_wait_end": 0,
                    "snn_en": SNNModeEnable.ENABLE,
                    "target_lcn": LCN_EX.LCN_1X,
                    "test_chip_addr": Coord(0, 0),
                },
            ),
            (
                Coord(0, 1),
                {
                    "name": "Core1",
                    "weight_width": WeightWidth.WEIGHT_WIDTH_4BIT,
                    "lcn": LCN_EX.LCN_2X,
                    "input_width": InputWidthFormat.WIDTH_1BIT,
                    "spike_width": SpikeWidthFormat.WIDTH_1BIT,
                    "num_dendrite": 500,
                    "max_pooling_en": MaxPoolingEnable.DISABLE,
                    "tick_wait_start": 0,
                    "tick_wait_end": 0,
                    "snn_en": SNNModeEnable.ENABLE,
                    "target_lcn": LCN_EX.LCN_2X,
                    "test_chip_addr": 30,
                    "unused_key": 999,
                },
            ),
            (
                Coord(1, 1),
                {
                    "name": "Core2",
                    "weight_width": WeightWidth.WEIGHT_WIDTH_8BIT,
                    "lcn": LCN_EX.LCN_2X,
                    "input_width": InputWidthFormat.WIDTH_8BIT,
                    "spike_width": SpikeWidthFormat.WIDTH_8BIT,
                    "num_dendrite": 500,
                    "max_pooling_en": MaxPoolingEnable.DISABLE,
                    "tick_wait_start": 0,
                    "tick_wait_end": 0,
                    "snn_en": SNNModeEnable.DISABLE,
                    "target_lcn": LCN_EX.LCN_4X,
                    "test_chip_addr": Coord(2, 1),
                },
            ),
        ],
    )
    def test_legal(self, ensure_dump_dir, coord, params):
        core_reg = OfflineCoreReg.model_validate(params, strict=True)
        core_reg_dict = core_reg.model_dump_json(by_alias=True)

        with open(ensure_dump_dir / f"offline_core_reg_{core_reg.name}.json", "w") as f:
            f.write(json.dumps({coord.address: json.loads(core_reg_dict)}, indent=2))

    @pytest.mark.parametrize(
        "params",
        [
            {
                # wrong 'tick_wait_end'
                "name": "Core0",
                "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
                "lcn": LCN_EX.LCN_2X,
                "input_width": InputWidthFormat.WIDTH_1BIT,
                "spike_width": SpikeWidthFormat.WIDTH_1BIT,
                "num_dendrite": 100,
                "max_pooling_en": MaxPoolingEnable.DISABLE,
                "tick_wait_start": 1,
                "tick_wait_end": -1,
                "snn_en": SNNModeEnable.ENABLE,
                "target_lcn": LCN_EX.LCN_1X,
                "test_chip_addr": Coord(0, 0),
            },
            {
                # missing key 'test_chip_addr'
                "name": "Core1",
                "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
                "lcn": LCN_EX.LCN_2X,
                "input_width": InputWidthFormat.WIDTH_1BIT,
                "spike_width": SpikeWidthFormat.WIDTH_1BIT,
                "num_dendrite": 500,
                "max_pooling_en": MaxPoolingEnable.DISABLE,
                "tick_wait_start": 1,
                "tick_wait_end": 0,
                "snn_en": SNNModeEnable.ENABLE,
                "target_lcn": LCN_EX.LCN_2X,
                "unused_key": 999,
            },
            {
                # wrong core mode (1,0,1)
                "name": "Core2",
                "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
                "lcn": LCN_EX.LCN_1X,
                "input_width": InputWidthFormat.WIDTH_8BIT,
                "spike_width": SpikeWidthFormat.WIDTH_1BIT,
                "num_dendrite": 500,
                "max_pooling_en": MaxPoolingEnable.DISABLE,
                "tick_wait_start": 1,
                "tick_wait_end": 0,
                "snn_en": SNNModeEnable.ENABLE,
                "target_lcn": LCN_EX.LCN_2X,
                "test_chip_addr": Coord(0, 1),
            },
            {
                "name": "Core3",
                "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
                "lcn": LCN_EX.LCN_2X,
                "input_width": InputWidthFormat.WIDTH_1BIT,
                "spike_width": SpikeWidthFormat.WIDTH_1BIT,
                "num_dendrite": 5000,  # <= 4096 when SNN disabled
                "max_pooling_en": MaxPoolingEnable.DISABLE,
                "tick_wait_start": 1,
                "tick_wait_end": 0,
                "snn_en": SNNModeEnable.DISABLE,
                "target_lcn": LCN_EX.LCN_2X,
                "test_chip_addr": Coord(0, 1),
            },
        ],
    )
    def test_illegal(self, params):
        with pytest.raises(ValidationError):
            _ = OfflineCoreReg.model_validate(params, strict=True)


class TestOnlineCoreRegModel:
    @pytest.mark.parametrize(
        "coord, params",
        [
            (
                Coord(28, 29),
                {
                    "name": "Core1",
                    "weight_width": WeightWidth.WEIGHT_WIDTH_8BIT,
                    "lcn": LCN_EX.LCN_1X,
                    "lateral_inhi_value": (1 << 16) - 1,
                    "weight_decay_value": 127,
                    "upper_weight": 127,
                    "lower_weight": -128,
                    "neuron_start": 0,
                    "neuron_end": 100,
                    "inhi_core_x_ex": 0,
                    "inhi_core_y_ex": 0,
                    "tick_wait_start": 0,
                    "tick_wait_end": 0,
                    "lut_random_en": [LUTRandomEnable.ENABLE] * 60,
                    "decay_random_en": DecayRandomEnable.DISABLE,
                    "leak_order": LeakOrder.LEAK_BEFORE_COMP,
                    "online_mode_en": OnlineModeEnable.ENABLE,
                    "random_seed": 1,  # not zero
                    "test_chip_addr": Coord(2, 0),
                    "extra_key": 999,
                },
            ),
            (
                Coord(30, 28),
                {
                    "name": "Core2",
                    "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
                    "lcn": LCN_EX.LCN_8X,
                    "lateral_inhi_value": np.int16(-1),
                    "weight_decay_value": np.int8(-128),
                    "upper_weight": 100,
                    "lower_weight": -100,
                    "neuron_start": 100,
                    "neuron_end": 200,
                    "inhi_core_x_ex": 1,
                    "inhi_core_y_ex": 1,
                    "tick_wait_start": 0,
                    "tick_wait_end": 10,
                    "lut_random_en": [LUTRandomEnable.DISABLE] * 60,
                    "decay_random_en": DecayRandomEnable.ENABLE,
                    "leak_order": LeakOrder.LEAK_AFTER_COMP,
                    "online_mode_en": OnlineModeEnable.DISABLE,
                    "random_seed": (1 << 16) - 1,  # not zero
                    "test_chip_addr": (1, 0),
                },
            ),
        ],
    )
    def test_legal(self, ensure_dump_dir, coord, params):
        core_reg = OnlineCoreReg.model_validate(params, strict=True)
        core_reg_dict = core_reg.model_dump_json(by_alias=True)

        with open(ensure_dump_dir / f"online_core_reg_{core_reg.name}.json", "w") as f:
            f.write(json.dumps({coord.address: json.loads(core_reg_dict)}, indent=2))

    online_core_reg_legal = {
        "name": "Core1",
        "weight_width": WeightWidth.WEIGHT_WIDTH_8BIT,
        "lcn": LCN_EX.LCN_1X,
        "lateral_inhi_value": (1 << 16) - 1,
        "weight_decay_value": 127,
        "upper_weight": 127,
        "lower_weight": -128,
        "neuron_start": 0,
        "neuron_end": 100,
        "inhi_core_x_ex": 0,
        "inhi_core_y_ex": 0,
        "tick_wait_start": 1,
        "tick_wait_end": 0,
        "lut_random_en": [LUTRandomEnable.ENABLE] * 60,
        "decay_random_en": DecayRandomEnable.DISABLE,
        "leak_order": LeakOrder.LEAK_BEFORE_COMP,
        "online_mode_en": OnlineModeEnable.ENABLE,
        "random_seed": 1,
        "test_chip_addr": Coord(2, 0),
    }

    error_items = {
        "upper_weight": 255,  # out of int8 range
        "lut_random_en": [LUTRandomEnable.ENABLE] * 50,  # not a len=60 list
        "random_seed": 0,  # zero is not allowed
        "neuron_start": 101,  # start > end
        "inhi_core_x_ex": 0b01101,  # Non-online cores within the multicast range
    }

    @pytest.mark.parametrize("field, error_value", error_items.items())
    def test_illegal(self, field, error_value, monkeypatch):
        monkeypatch.setitem(self.online_core_reg_legal, field, error_value)

        with pytest.raises(ValidationError):
            _ = OnlineCoreReg.model_validate(self.online_core_reg_legal, strict=True)
