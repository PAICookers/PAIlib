import random

import numpy as np
import pytest

from paicorelib import RM, LCM, NTM, LDM, LIM, SIM
from paicorelib.coordinate import Coord
from paicorelib.hw_defs import (
    HwOfflineCoreParams as OffCoreParams,
    HwOnlineCoreParams as OnCoreParams,
)
from paicorelib.reg_defs import *
from paicorelib.reg_defs import OnlineRegDefs
from paicorelib.ram_defs import (
    OfflineRAMDefs as OffRAMDefs,
    OnlineRAMDefs as OnRAMDefs,
    OnlineRAMDefs_WW1 as OnRAMDefs_WW1,
    OnlineRAMDefs_WWn as OnRAMDefs_WWn,
)
from paicorelib.reg_defs import OfflineRegDefs as OffRegDefs
from paicorelib.reg_model import LUT_RANDOM_EN_LEN


def _gen_random_offline_coord_tuple() -> tuple[int, int]:
    return random.randint(
        OffCoreParams.CORE_X_MIN, OffCoreParams.CORE_X_MAX
    ), random.randint(OffCoreParams.CORE_Y_MIN, OffCoreParams.CORE_Y_MAX)


def _gen_random_online_coord_tuple() -> tuple[int, int]:
    return random.randint(
        OnCoreParams.CORE_X_MIN, OnCoreParams.CORE_X_MAX
    ), random.randint(OnCoreParams.CORE_Y_MIN, OnCoreParams.CORE_Y_MAX)


def _gen_tick_wait() -> tuple[int, int]:
    tws = random.randint(0, OffRegDefs.TICK_WAIT_START_MAX)
    _choice = random.choice([0, 1])
    if _choice == 0:
        twe = 0
    else:
        twe = random.randint(tws, OffRegDefs.TICK_WAIT_END_MAX)

    return tws, twe


@pytest.fixture(scope="class")
def gen_offline_core_reg():
    ww = random.choice(list(WeightWidth))
    lcn_ex = random.choice(list(LCN_EX))

    _core_mode = random.choice(list(CoreMode))
    iwf, swf, sme = _core_mode.conf

    if _core_mode.is_iw8:
        num_den = random.randint(1, OffCoreParams.N_DENDRITE_MAX_ANN)
    else:
        num_den = random.randint(1, OffCoreParams.N_DENDRITE_MAX_SNN)

    mpe = random.choice(list(MaxPoolingEnable))
    tws, twe = _gen_tick_wait()
    target_lcn = random.choice(list(LCN_EX))
    test_chip_addr = Coord(*_gen_random_offline_coord_tuple())

    return dict(
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


@pytest.fixture(scope="class")
def gen_online_core_reg():
    ww = random.choice(list(WeightWidth))
    lcn_ex = random.choice([LCN_EX.LCN_1X, LCN_EX.LCN_2X, LCN_EX.LCN_4X, LCN_EX.LCN_8X])
    tws, twe = _gen_tick_wait()
    lateral_inhi_value = random.randint(
        OnlineRegDefs.LATERAL_INHI_VALUE_MIN, OnlineRegDefs.LATERAL_INHI_VALUE_MAX
    )
    weight_decay_value = random.randint(
        OnlineRegDefs.WEIGHT_DECAY_VALUE_MIN, OnlineRegDefs.WEIGHT_DECAY_VALUE_MAX
    )
    upper_weight = random.randint(
        OnlineRegDefs.UPPER_WEIGHT_MIN, OnlineRegDefs.UPPER_WEIGHT_MAX
    )
    # <= upper_weight
    lower_weight = random.randint(OnlineRegDefs.LOWER_WEIGHT_MIN, upper_weight)
    neuron_start = random.randint(0, OnlineRegDefs.NEU_START_MAX)
    # >= neuron_start
    neuron_end = random.randint(neuron_start, OnlineRegDefs.NEU_END_MAX)
    inhi_core_x_ex = random.randint(0, 0b00011)
    inhi_core_y_ex = random.randint(0, 0b00011)
    lut_random_en = [
        random.choice(list(LUTRandomEnable)) for _ in range(LUT_RANDOM_EN_LEN)
    ]
    decay_random_en = random.choice(list(DecayRandomEnable))
    leak_order = random.choice(list(LeakOrder))
    online_mode_en = random.choice(list(OnlineModeEnable))
    test_chip_addr = Coord(*_gen_random_online_coord_tuple())
    random_seed = random.randint(1, OnlineRegDefs.RANDOM_SEED_MAX)

    return dict(
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


@pytest.fixture(
    scope="class",
    params=[10, np.arange(100, dtype=np.int32)],
)
def gen_OfflineNeuAttrs(request):
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

    return dict(
        **{
            "reset_mode": reset_mode,
            "reset_v": reset_v,
            "leak_comparison": leak_comparison,
            "thres_mask_bits": threshold_mask_bits,
            "neg_thres_mode": neg_thres_mode,
            "neg_threshold": neg_threshold,
            "pos_threshold": pos_threshold,
            "leak_direction": leak_direction,
            "leak_integration_mode": leak_integration_mode,
            "leak_v": request.param,
            "syn_integration_mode": syn_integration_mode,
            "bit_trunc": bit_trunc,
        }
    )


@pytest.fixture(scope="class")
def gen_OnlineNeuAttrs():
    ww = random.choice(list(WeightWidth))
    if ww == WeightWidth.WEIGHT_WIDTH_1BIT:
        getter = OnRAMDefs_WW1
    else:
        getter = OnRAMDefs_WWn

    leak_v = random.randint(getter.LEAK_V_MIN, getter.LEAK_V_MAX)
    threshold = random.randint(getter.THRES_MIN, getter.THRES_MAX)
    floor_thres = random.randint(getter.FLOOR_THRES_MIN, getter.FLOOR_THRES_MAX)
    reset_v = random.randint(getter.RESET_V_MIN, getter.RESET_V_MAX)
    init_v = random.randint(getter.INIT_V_MIN, getter.INIT_V_MAX)
    plasticity_start = random.randint(0, getter.PLASTICITY_START_MAX)
    plasticity_end = random.randint(plasticity_start, getter.PLASTICITY_END_MAX)

    return ww, dict(
        **{
            "leak_v": leak_v,
            "threshold": threshold,
            "floor_thres": floor_thres,
            "reset_v": reset_v,
            "init_v": init_v,
            "plasticity_start": plasticity_start,
            "plasticity_end": plasticity_end,
        }
    )


@pytest.fixture(scope="class")
def gen_OfflineNeuDestInfo():
    addr_chip_x, addr_chip_y = _gen_random_offline_coord_tuple()
    addr_core_x, addr_core_y = _gen_random_offline_coord_tuple()
    addr_core_x_ex, addr_core_y_ex = _gen_random_offline_coord_tuple()

    n = 100
    tick_relative = list(range(n))

    addr_axon = random.sample(list(range(OffRAMDefs.ADDR_AXON_MAX)), len(tick_relative))

    return dict(
        **{
            "addr_chip_x": addr_chip_x,
            "addr_chip_y": addr_chip_y,
            "addr_core_x": addr_core_x,
            "addr_core_y": addr_core_y,
            "addr_core_x_ex": addr_core_x_ex,
            "addr_core_y_ex": addr_core_y_ex,
            "tick_relative": tick_relative,
            "addr_axon": addr_axon,
        }
    )


@pytest.fixture(scope="class")
def gen_OnlineNeuDestInfo():
    addr_chip_x, addr_chip_y = _gen_random_offline_coord_tuple()
    addr_core_x, addr_core_y = _gen_random_offline_coord_tuple()
    addr_core_x_ex, addr_core_y_ex = _gen_random_offline_coord_tuple()

    n = random.randint(1, OnRAMDefs.ADDR_TS_MAX)
    tick_relative = 2 * list(range(n))

    addr_axon = random.sample(list(range(OnRAMDefs.ADDR_AXON_MAX)), len(tick_relative))

    return dict(
        **{
            "addr_chip_x": addr_chip_x,
            "addr_chip_y": addr_chip_y,
            "addr_core_x": addr_core_x,
            "addr_core_y": addr_core_y,
            "addr_core_x_ex": addr_core_x_ex,
            "addr_core_y_ex": addr_core_y_ex,
            "tick_relative": tick_relative,
            "addr_axon": addr_axon,
        }
    )
