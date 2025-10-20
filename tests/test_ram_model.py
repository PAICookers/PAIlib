from contextlib import nullcontext

import numpy as np
import pytest
from pydantic import ValidationError

from paicorelib import LCM, LDM, LIM, NTM, RM, SIM
from paicorelib.ram_defs import OfflineRAMDefs as OffRAMDefs
from paicorelib.ram_defs import OnlineRAMDefs as OnRAMDefs
from paicorelib.ram_defs import OnlineRAMDefs_WWn as OnRAMDefs_WWn
from paicorelib.ram_model import (
    OfflineNeuAttrs,
    OfflineNeuDestInfo,
    OnlineNeuAttrs,
    OnlineNeuDestInfo,
)
from paicorelib.reg_defs import WeightWidth as WW

OFFLINE_ADDR_AXON_MAX = OffRAMDefs.ADDR_AXON_MAX
OFFLINE_ADDR_TS_MAX = OffRAMDefs.ADDR_TS_MAX
ONLINE_ADDR_AXON_MAX = OnRAMDefs.ADDR_AXON_MAX
ONLINE_ADDR_TS_MAX = OnRAMDefs.ADDR_TS_MAX


class TestOfflineNeuRAMModel:
    @pytest.mark.parametrize(
        "params, expectation",
        [
            (
                {
                    "tick_relative": [0] * 100 + [1] * 100,
                    "addr_axon": list(range(0, 200)),
                    "addr_core_x": 0,
                    "addr_core_y": 1,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 0,
                    "addr_chip_y": 0,
                },
                nullcontext(),
            ),
            (
                {
                    "tick_relative": [0] * 100,
                    "addr_axon": list(range(0, 100)),
                    "addr_core_x": 0,
                    "addr_core_y": 1,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 0,
                    "addr_chip_y": 0,
                },
                nullcontext(),
            ),
            (
                {
                    "tick_relative": list(range(200)),
                    "addr_axon": list(range(100)),  # lenght != 200
                    "addr_core_x": 0,
                    "addr_core_y": 1,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 0,
                    "addr_chip_y": 0,
                },
                pytest.raises(ValidationError),
            ),
            (
                {
                    # out of range
                    "tick_relative": list(range(2, OFFLINE_ADDR_TS_MAX + 2)),
                    "addr_axon": list(range(OFFLINE_ADDR_TS_MAX)),
                    "addr_core_x": 0,
                    "addr_core_y": 1,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 0,
                    "addr_chip_y": 0,
                },
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_neu_dest_info(self, ensure_dump_dir, params, expectation):
        with expectation:
            dest_info = OfflineNeuDestInfo.model_validate(params, strict=True)
            dest_info_dict = dest_info.model_dump_json(indent=2, by_alias=True)

            with open(ensure_dump_dir / "offline_dest_info.json", "w") as f:
                f.write(dest_info_dict)

    @pytest.mark.parametrize(
        "params, expectation",
        [
            (
                {
                    "reset_mode": RM.MODE_NORMAL,
                    "reset_v": -1,
                    "leak_comparison": LCM.LEAK_BEFORE_COMP,
                    "thres_mask_bits": 1,
                    "neg_thres_mode": NTM.MODE_RESET,
                    "neg_threshold": 1,
                    "pos_threshold": 0,
                    "leak_direction": LDM.MODE_FORWARD,
                    "leak_integration_mode": LIM.MODE_DETERMINISTIC,
                    "leak_v": np.array([1, 2, 3, 4, 5, 6]),
                    "syn_integration_mode": SIM.MODE_DETERMINISTIC,
                    "bit_trunc": 0,
                },
                nullcontext(),
            ),
            # neg_threshold is a non-negative
            (
                {
                    "reset_mode": RM.MODE_NORMAL,
                    "reset_v": -1,
                    "leak_comparison": LCM.LEAK_BEFORE_COMP,
                    "thres_mask_bits": 1,
                    "neg_thres_mode": NTM.MODE_RESET,
                    "neg_threshold": -1,
                    "pos_threshold": 0,
                    "leak_direction": LDM.MODE_FORWARD,
                    "leak_integration_mode": LIM.MODE_DETERMINISTIC,
                    "leak_v": 1,
                    "syn_integration_mode": SIM.MODE_DETERMINISTIC,
                    "bit_trunc": 1,
                },
                pytest.raises(ValidationError),
            ),
            # thres_mask_bits is a non-negative
            (
                {
                    "reset_mode": RM.MODE_NORMAL,
                    "reset_v": 0,
                    "leak_comparison": LCM.LEAK_BEFORE_COMP,
                    "thres_mask_bits": -1,
                    "neg_thres_mode": NTM.MODE_RESET,
                    "neg_threshold": 1 << 10,
                    "pos_threshold": 1 << 10,
                    "leak_direction": LDM.MODE_REVERSAL,
                    "leak_integration_mode": LIM.MODE_STOCHASTIC,
                    "leak_v": -1,
                    "syn_integration_mode": SIM.MODE_STOCHASTIC,
                    "bit_trunc": 0,
                },
                pytest.raises(ValidationError),
            ),
            # bit_trunc is a non-negative
            (
                {
                    "reset_mode": RM.MODE_NONRESET,
                    "reset_v": 1,
                    "leak_comparison": LCM.LEAK_AFTER_COMP,
                    "thres_mask_bits": 0,
                    "neg_thres_mode": NTM.MODE_SATURATION,
                    "neg_threshold": 1 << 10,
                    "pos_threshold": 1 << 10,
                    "leak_direction": LDM.MODE_FORWARD,
                    "leak_integration_mode": LIM.MODE_DETERMINISTIC,
                    "leak_v": -1,
                    "syn_integration_mode": SIM.MODE_STOCHASTIC,
                    "bit_trunc": -1,
                },
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_neu_attrs(self, ensure_dump_dir, params, expectation):
        with expectation:
            neu_attrs = OfflineNeuAttrs.model_validate(params, strict=True)
            neu_attrs_dict = neu_attrs.model_dump_json(indent=2, by_alias=True)

            with open(ensure_dump_dir / "offline_neu_attrs.json", "w") as f:
                f.write(neu_attrs_dict)


class TestOnlineNeuRAMModel:
    @pytest.mark.parametrize(
        "params, expectation",
        [
            (
                {
                    "tick_relative": [0] * 100 + [1] * 100,
                    "addr_axon": list(range(0, 200)),
                    "addr_core_x": 28,
                    "addr_core_y": 28,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 0,
                    "addr_chip_y": 0,
                },
                nullcontext(),
            ),
            (
                {
                    "tick_relative": [0] * 100,
                    "addr_axon": list(range(0, 100)),
                    "addr_core_x": 30,
                    "addr_core_y": 29,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 0,
                    "addr_chip_y": 0,
                },
                nullcontext(),
            ),
            (
                {
                    "tick_relative": [0, 1, 2, 3, 4, 5, 6] * 100,
                    "addr_axon": list(range(600)),  # lenght != 700
                    "addr_core_x": 29,
                    "addr_core_y": 31,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 0,
                    "addr_chip_y": 0,
                },
                pytest.raises(ValidationError, match="addr_axon"),
            ),
            (
                {
                    # out of range
                    "tick_relative": list(range(2, ONLINE_ADDR_TS_MAX + 2)),
                    "addr_axon": list(range(ONLINE_ADDR_TS_MAX)),
                    "addr_core_x": 31,
                    "addr_core_y": 31,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 0,
                    "addr_chip_y": 0,
                },
                pytest.raises(ValidationError, match="tick_relative"),
            ),
        ],
    )
    def test_neu_dest_info(self, ensure_dump_dir, params, expectation):
        with expectation:
            dest_info = OnlineNeuDestInfo.model_validate(params, strict=True)
            dest_info_dict = dest_info.model_dump_json(indent=2, by_alias=True)

            with open(ensure_dump_dir / "online_dest_info.json", "w") as f:
                f.write(dest_info_dict)

    @pytest.mark.parametrize(
        "params, context, expectation",
        [
            (
                {
                    "leak_v": -1,
                    "pos_threshold": 1000,
                    "neg_threshold": 1,
                    "reset_v": 0,
                    "init_v": 1,
                    # Use default values
                    # "plasticity_start": 0,
                    # "plasticity_end": ONLINE_ADDR_AXON_MAX,
                },
                {"weight_width": WW.WEIGHT_WIDTH_1BIT},
                nullcontext(),
            ),
            (
                {
                    "leak_v": 1,
                    "pos_threshold": 2000,
                    "neg_threshold": -1000,
                    "reset_v": -1,
                    "init_v": 0,
                    # Use default values
                    # "plasticity_start": 0,
                    # "plasticity_end": ONLINE_ADDR_AXON_MAX,
                },
                {"weight_width": WW.WEIGHT_WIDTH_8BIT},
                nullcontext(),
            ),
            (
                {
                    "leak_v": -1,
                    "pos_threshold": 1000,
                    "neg_threshold": 1,
                    "reset_v": 0,
                    "init_v": 1,
                    # Use default values
                    # "plasticity_start": 0,
                    # "plasticity_end": ONLINE_ADDR_AXON_MAX,
                },
                # 'weight_width' is not in context, a warning will be raised
                {},
                pytest.warns(UserWarning, match="weight_width"),
            ),
            (
                {
                    "leak_v": -1,
                    "pos_threshold": 1000,
                    "neg_threshold": 1,
                    "reset_v": 0,
                    "init_v": 1,
                    "plasticity_start": 500,
                    "plasticity_end": 200,  # start > end
                },
                {"weight_width": WW.WEIGHT_WIDTH_1BIT},
                pytest.raises(ValidationError, match="plasticity_start"),
            ),
            (
                {
                    "leak_v": -1,
                    "pos_threshold": 1000,
                    "neg_threshold": 1,
                    # 'reset_v' is out of range when ww=1
                    "reset_v": OnRAMDefs_WWn.RESET_V_MAX,
                    "init_v": 1,
                    "plasticity_start": 500,
                    "plasticity_end": 1000,
                },
                {"weight_width": WW.WEIGHT_WIDTH_1BIT, "extra_context": 123},
                pytest.raises(ValidationError, match="reset_v"),
            ),
        ],
    )
    def test_neu_attrs(self, ensure_dump_dir, params, context, expectation):
        with expectation:
            neu_attrs = OnlineNeuAttrs.model_validate(
                params, strict=True, context=context
            )
            dest_info_dict = neu_attrs.model_dump_json(indent=2, by_alias=True)

            with open(ensure_dump_dir / "online_neu_attrs.json", "w") as f:
                f.write(dest_info_dict)
