import random

import numpy as np
import pytest

from paicorelib import *
from paicorelib.ram_model import (
    BIT_TRUNCATE_MAX,
    BIT_TRUNCATE_MIN,
    NEG_THRES_MAX,
    NEG_THRES_MIN,
    POS_THRES_MAX,
    POS_THRES_MIN,
    RESET_V_MAX,
    RESET_V_MIN,
    THRES_MASK_CTRL_MAX,
    THRES_MASK_CTRL_MIN,
)
from paicorelib.reg_model import (
    TICK_WAIT_END_MAX,
    TICK_WAIT_START_MAX,
    _OfflineCoreRegDict,
)


@pytest.fixture(scope="class")
def gen_random_params_reg_dict():
    ww = random.choice(list(WeightWidth))
    lcn_ex = random.choice(list(LCN_EX))

    _core_mode = random.choice(list(CoreMode))
    iwf, swf, sme = _core_mode.conf

    if _core_mode.is_iw8:
        num_den = random.randint(1, HwConfig.N_DENDRITE_MAX_ANN)
    else:
        num_den = random.randint(1, HwConfig.N_DENDRITE_MAX_SNN)

    mpe = random.choice(list(MaxPoolingEnable))
    tws = random.randint(0, TICK_WAIT_START_MAX)
    twe = random.randint(0, TICK_WAIT_END_MAX)
    target_lcn = random.choice(list(LCN_EX))
    test_chip_addr = Coord(random.randint(0, 31), random.randint(0, 31))

    return _OfflineCoreRegDict(
        weight_width=ww.value,
        LCN=lcn_ex.value,
        input_width=iwf.value,
        spike_width=swf.value,
        num_dendrite=num_den,
        pool_max=mpe.value,
        tick_wait_start=tws,
        tick_wait_end=twe,
        snn_en=sme.value,
        target_LCN=target_lcn.value,
        test_chip_addr=test_chip_addr.address,
    )


@pytest.fixture(
    scope="class",
    params=[10, np.arange(100, dtype=np.int32)],
)
def gen_NeuronAttrs(request):
    reset_mode = random.choice(list(RM))
    reset_v = random.randint(RESET_V_MIN, RESET_V_MAX)
    leak_comparison = random.choice(list(LCM))
    threshold_mask_bits = random.randint(THRES_MASK_CTRL_MIN, THRES_MASK_CTRL_MAX)
    neg_thres_mode = random.choice(list(NTM))
    neg_threshold = random.randint(NEG_THRES_MIN, NEG_THRES_MAX)
    pos_threshold = random.randint(POS_THRES_MIN, POS_THRES_MAX)
    leak_direction = random.choice(list(LDM))
    leak_integration_mode = random.choice(list(LIM))
    synaptic_integration_mode = random.choice(list(SIM))
    bit_truncation = random.randint(BIT_TRUNCATE_MIN, BIT_TRUNCATE_MAX)

    return NeuronAttrs.model_validate(
        {
            "reset_mode": reset_mode,
            "reset_v": reset_v,
            "leak_comparison": leak_comparison,
            "threshold_mask_bits": threshold_mask_bits,
            "neg_thres_mode": neg_thres_mode,
            "neg_threshold": neg_threshold,
            "pos_threshold": pos_threshold,
            "leak_direction": leak_direction,
            "leak_integration_mode": leak_integration_mode,
            "leak_v": request.param,
            "synaptic_integration_mode": synaptic_integration_mode,
            "bit_truncation": bit_truncation,
        },
        strict=True,
    )


@pytest.fixture(scope="class")
def gen_NeuronDestInfo():
    addr_chip_x, addr_chip_y = random.randint(0, 31), random.randint(0, 31)
    addr_core_x, addr_core_y = random.randint(0, 31), random.randint(0, 31)
    addr_core_x_ex, addr_core_y_ex = random.randint(0, 31), random.randint(0, 31)

    n = 100
    tick_relative = [0] * n
    addr_axon = random.sample(list(range(1152)), n)

    return NeuronDestInfo.model_validate(
        {
            "addr_chip_x": addr_chip_x,
            "addr_chip_y": addr_chip_y,
            "addr_core_x": addr_core_x,
            "addr_core_y": addr_core_y,
            "addr_core_x_ex": addr_core_x_ex,
            "addr_core_y_ex": addr_core_y_ex,
            "tick_relative": tick_relative,
            "addr_axon": addr_axon,
        },
        strict=True,
    )
