import random
from typing import Any

import numpy as np
import pytest

from paicorelib import (
    LCM,
    LDM,
    LIM,
    NTM,
    RM,
    SIM,
    Coord,
    OffCoreCfg,
    OffRAMDefs,
    OnCoreCfg,
    OnRAMDefs,
    OnRAMDefs_WW1,
    OnRAMDefs_WWn,
    OnRegDefs,
    RegDefs,
)
from paicorelib.reg_defs import *
from paicorelib.reg_model import LUT_RANDOM_EN_LEN

__all__ = [
    "gen_offline_core_reg_test_cases",
    "gen_offline_neu_test_cases",
    "gen_online_core_reg_test_cases",
    "gen_online_neu_test_cases",
    "validate_offline_neu_test_data",
    "validate_online_neu_test_data",
]


def gen_random_offline_coord_tuple() -> tuple[int, int]:
    return random.randint(OffCoreCfg.CORE_X_MIN, OffCoreCfg.CORE_X_MAX), random.randint(
        OffCoreCfg.CORE_Y_MIN, OffCoreCfg.CORE_Y_MAX
    )


def gen_random_online_coord_tuple() -> tuple[int, int]:
    return random.randint(OnCoreCfg.CORE_X_MIN, OnCoreCfg.CORE_X_MAX), random.randint(
        OnCoreCfg.CORE_Y_MIN, OnCoreCfg.CORE_Y_MAX
    )


def gen_tick_wait_start_and_end() -> tuple[int, int]:
    # Common for both online & offline cores
    tws = random.randint(0, RegDefs.TICK_WAIT_START_MAX)
    _choice = random.choice([0, 1])
    if _choice == 0:
        twe = 0
    else:
        twe = random.randint(tws, RegDefs.TICK_WAIT_END_MAX)

    return tws, twe


def gen_offline_core_reg_test_cases():
    test_cases = []
    ww = random.choice(list(WeightWidth))
    lcn_ex = random.choice(list(LCN_EX))

    core_mode = [CoreMode.MODE_SNN, CoreMode.MODE_ANN]

    mpe = random.choice(list(MaxPoolingEnable))
    tws, twe = gen_tick_wait_start_and_end()
    target_lcn = random.choice(list(LCN_EX))
    test_chip_addr = Coord(*gen_random_offline_coord_tuple())

    for cm in core_mode:
        iwf, swf, sme = cm.conf

        if cm.is_iw8:
            num_den = random.randint(1, OffCoreCfg.N_DENDRITE_MAX_ANN)
        else:
            num_den = random.randint(1, OffCoreCfg.N_DENDRITE_MAX_SNN)

        test_cases.append(
            dict(
                weight_width=ww,
                lcn=lcn_ex,
                input_width=iwf,
                spike_width=swf,
                num_dendrite=num_den,
                max_pooling_en=mpe,
                tick_wait_start=tws,
                tick_wait_end=twe,
                snn_en=sme,
                target_lcn=target_lcn,
                test_chip_addr=test_chip_addr,
            )
        )

    return test_cases


def gen_offline_neu_test_n():
    return [10, 128]


def gen_offline_neu_leak_v():
    return [-1, np.arange(128, dtype=np.int32)]


def gen_offline_neu_attrs_test_cases():
    test_cases = []
    reset_mode = random.choice(list(RM))
    reset_v = random.randint(OffRAMDefs.RESET_V_MIN, OffRAMDefs.RESET_V_MAX)
    leak_comparison = random.choice(list(LCM))
    threshold_mask_bits = random.randint(0, OffRAMDefs.THRES_MASK_BITS_MAX)
    neg_thres_mode = random.choice(list(NTM))
    neg_threshold = random.randint(0, OffRAMDefs.NEG_THRES_MAX)
    pos_threshold = random.randint(0, OffRAMDefs.POS_THRES_MAX)
    leak_direction = random.choice(list(LDM))
    leak_integration_mode = random.choice(list(LIM))
    syn_integration_mode = random.choice(list(SIM))
    bit_trunc = random.randint(0, OffRAMDefs.BIT_TRUNC_MAX)

    leak_v = gen_offline_neu_leak_v()

    for leak in leak_v:
        test_cases.append(
            {
                "reset_mode": reset_mode,
                "reset_v": reset_v,
                "leak_comparison": leak_comparison,
                "thres_mask_bits": threshold_mask_bits,
                "neg_thres_mode": neg_thres_mode,
                "neg_threshold": neg_threshold,
                "pos_threshold": pos_threshold,
                "leak_direction": leak_direction,
                "leak_integration_mode": leak_integration_mode,
                "leak_v": leak,
                "syn_integration_mode": syn_integration_mode,
                "bit_trunc": bit_trunc,
            }
        )

    return test_cases


def gen_offline_neu_dest_info_test_cases():
    test_cases = []
    addr_chip_x, addr_chip_y = gen_random_offline_coord_tuple()
    addr_core_x, addr_core_y = gen_random_offline_coord_tuple()
    addr_core_x_ex, addr_core_y_ex = gen_random_offline_coord_tuple()

    test_n = gen_offline_neu_test_n()

    for n in test_n:
        tick_relative = list(range(n))
        addr_axon = random.sample(
            list(range(OffRAMDefs.ADDR_AXON_MAX)), len(tick_relative)
        )
        case = {
            "addr_chip_x": addr_chip_x,
            "addr_chip_y": addr_chip_y,
            "addr_core_x": addr_core_x,
            "addr_core_y": addr_core_y,
            "addr_core_x_ex": addr_core_x_ex,
            "addr_core_y_ex": addr_core_y_ex,
            "tick_relative": tick_relative,
            "addr_axon": addr_axon,
            "n_neuron": n,
        }
        test_cases.append(case)

    return test_cases


def gen_offline_neu_test_cases():
    # Return (neu_attrs, dest_info)
    return zip(
        gen_offline_neu_attrs_test_cases(),
        gen_offline_neu_dest_info_test_cases(),
        strict=True,
    )


def validate_offline_neu_test_data(
    neu_attrs: dict[str, Any], dest_info: dict[str, Any]
):
    n = dest_info.get("n_neuron")
    if n is None:
        pytest.fail("test data 'n_neuron' not found")

    if not 1 <= n <= OffRAMDefs.ADDR_TS_MAX:
        pytest.fail(
            f"test data 'n_neuron' must be in [1, {OffRAMDefs.ADDR_TS_MAX}], but got {n}"
        )

    leak_v = neu_attrs.get("leak_v")
    if isinstance(leak_v, np.ndarray) and n < leak_v.size:
        pytest.fail(f"size of test data 'leak_v'({leak_v.size}) is smaller than {n}")

    return True


def gen_online_core_reg_test_cases():
    test_cases = []
    ww = random.choice(list(WeightWidth))
    lcn_ex = random.choice([LCN_EX.LCN_1X, LCN_EX.LCN_2X, LCN_EX.LCN_4X, LCN_EX.LCN_8X])
    tws, twe = gen_tick_wait_start_and_end()
    lateral_inhi_value = random.randint(
        OnRegDefs.LATERAL_INHI_VALUE_MIN, OnRegDefs.LATERAL_INHI_VALUE_MAX
    )
    weight_decay_value = random.randint(
        OnRegDefs.WEIGHT_DECAY_VALUE_MIN, OnRegDefs.WEIGHT_DECAY_VALUE_MAX
    )
    upper_weight = random.randint(
        OnRegDefs.UPPER_WEIGHT_MIN, OnRegDefs.UPPER_WEIGHT_MAX
    )
    # <= upper_weight
    lower_weight = random.randint(OnRegDefs.LOWER_WEIGHT_MIN, upper_weight)
    neuron_start = random.randint(0, OnRegDefs.NEU_START_MAX)
    # >= neuron_start
    neuron_end = random.randint(neuron_start, OnRegDefs.NEU_END_MAX)
    inhi_core_x_ex = random.randint(0, 0b00011)
    inhi_core_y_ex = random.randint(0, 0b00011)
    lut_random_en = [
        random.choice(list(LUTRandomEnable)) for _ in range(LUT_RANDOM_EN_LEN)
    ]
    decay_random_en = random.choice(list(DecayRandomEnable))
    leak_order = random.choice(list(LeakOrder))
    online_mode_en = random.choice(list(OnlineModeEnable))
    test_chip_addr = Coord(*gen_random_online_coord_tuple())
    random_seed = random.randint(1, OnRegDefs.RANDOM_SEED_MAX)

    test_cases.append(
        dict(
            weight_width=ww,
            lcn=lcn_ex,
            lateral_inhi_value=lateral_inhi_value,
            weight_decay_value=weight_decay_value,
            upper_weight=upper_weight,
            lower_weight=lower_weight,
            neuron_start=neuron_start,
            neuron_end=neuron_end,
            inhi_core_x_ex=inhi_core_x_ex,
            inhi_core_y_ex=inhi_core_y_ex,
            tick_wait_start=tws,
            tick_wait_end=twe,
            lut_random_en=lut_random_en,
            decay_random_en=decay_random_en,
            leak_order=leak_order,
            online_mode_en=online_mode_en,
            test_chip_addr=test_chip_addr,
            random_seed=random_seed,
        )
    )

    return test_cases


def gen_online_neu_test_n():
    return [1, 4, 7]


def gen_online_neu_attrs():
    leak_v = [
        -1,
        np.full((4,), -1, dtype=np.int32),
        np.arange(7, dtype=np.int32),
    ]
    init_v = [
        np.full((1,), -1, dtype=np.int32),
        1,
        np.arange(7, dtype=np.int32),
    ]
    return leak_v, init_v


def gen_online_neu_attrs_test_cases():
    test_cases = []

    ww = random.choice(list(WeightWidth))
    if ww == WeightWidth.WEIGHT_WIDTH_1BIT:
        getter = OnRAMDefs_WW1
    else:
        getter = OnRAMDefs_WWn

    leak_v, init_v = gen_online_neu_attrs()

    threshold = random.randint(getter.THRES_MIN, getter.THRES_MAX)
    neg_threshold = random.randint(getter.FLOOR_THRES_MIN, getter.FLOOR_THRES_MAX)
    reset_v = random.randint(getter.RESET_V_MIN, getter.RESET_V_MAX)

    # Not used in production
    plasticity_start = random.randint(0, getter.PLASTICITY_START_MAX)
    plasticity_end = random.randint(plasticity_start, getter.PLASTICITY_END_MAX)

    for leak, v in zip(leak_v, init_v, strict=True):
        test_cases.append(
            {
                "leak_v": leak,
                "pos_threshold": threshold,
                "neg_threshold": neg_threshold,
                "reset_v": reset_v,
                "init_v": v,
                "plasticity_start": plasticity_start,
                "plasticity_end": plasticity_end,
                "weight_width": ww,
            }
        )

    return test_cases


def gen_online_neu_dest_info_test_cases():
    test_cases = []
    addr_chip_x, addr_chip_y = gen_random_offline_coord_tuple()
    addr_core_x, addr_core_y = gen_random_offline_coord_tuple()
    addr_core_x_ex, addr_core_y_ex = gen_random_offline_coord_tuple()

    test_n = gen_online_neu_test_n()

    for n in test_n:
        tick_relative = list(range(n))
        addr_axon = random.sample(
            list(range(OnRAMDefs.ADDR_AXON_MAX)), len(tick_relative)
        )
        case = {
            "addr_chip_x": addr_chip_x,
            "addr_chip_y": addr_chip_y,
            "addr_core_x": addr_core_x,
            "addr_core_y": addr_core_y,
            "addr_core_x_ex": addr_core_x_ex,
            "addr_core_y_ex": addr_core_y_ex,
            "tick_relative": tick_relative,
            "addr_axon": addr_axon,
            "n_neuron": n,
        }
        test_cases.append(case)

    return test_cases


def gen_online_neu_test_cases():
    # Return (neu_attrs, dest_info)
    return zip(
        gen_online_neu_attrs_test_cases(),
        gen_online_neu_dest_info_test_cases(),
        strict=True,
    )


def validate_online_neu_test_data(neu_attrs: dict[str, Any], dest_info: dict[str, Any]):
    n = dest_info.get("n_neuron")
    if n is None:
        pytest.fail("test data 'n_neuron' not found")

    if not 1 <= n <= OnRAMDefs.ADDR_TS_MAX:
        pytest.fail(
            f"test data 'n_neuron' must be in [1, {OnRAMDefs.ADDR_TS_MAX}], but got {n}"
        )

    leak_v = neu_attrs.get("leak_v")
    init_v = neu_attrs.get("init_v")
    if isinstance(leak_v, np.ndarray) and n < leak_v.size:
        pytest.fail(f"size of test data 'leak_v'({leak_v.size}) is smaller than {n}")

    if isinstance(init_v, np.ndarray) and n < init_v.size:
        pytest.fail(f"size of test data 'init_v'({init_v.size}) is smaller than {n}")

    return True
