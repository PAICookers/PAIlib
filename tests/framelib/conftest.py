import random
import pytest
from paicorelib import *


@pytest.fixture(scope="class")
def gen_random_params_reg_dict():
    wp = random.choice(list(WeightPrecision))
    lcn_ex = random.choice(list(LCN_EX))
    iwf = random.choice(list(InputWidthFormat))
    swf = random.choice(list(SpikeWidthFormat))
    num_den = random.randint(1, 512)
    mpe = random.choice(list(MaxPoolingEnable))
    tws = random.randint(0, 100)
    twe = random.randint(0, 100)
    sme = random.choice(list(SNNModeEnable))
    target_lcn = random.choice(list(LCN_EX))
    test_chip_addr = Coord(random.randint(0, 31), random.randint(0, 31))

    return {
        "weight_width": wp.value,
        "LCN": lcn_ex.value,
        "input_width": iwf.value,
        "spike_width": swf.value,
        "neuron_num": num_den,
        "pool_max": mpe.value,
        "tick_wait_start": tws,
        "tick_wait_end": twe,
        "snn_en": sme.value,
        "target_LCN": target_lcn.value,
        "test_chip_addr": test_chip_addr.address,
    }


@pytest.fixture(scope="class")
def gen_NeuronAttrs():
    reset_mode = random.choice(list(RM))
    reset_v = random.randint(-(1 << 29), 1 << 29)
    leak_comparison = random.choice(list(LCM))
    threshold_mask_bits = random.randint(0, 1 << 5)
    neg_thres_mode = random.choice(list(NTM))
    neg_threshold = random.randint(0, 1 << 29)
    pos_threshold = random.randint(0, 1 << 29)
    leak_direction = random.choice(list(LDM))
    leak_integration_mode = random.choice(list(LIM))
    leak_v = random.randint(-(1 << 29), 1 << 29)
    synaptic_integration_mode = random.choice(list(SIM))
    bit_truncation = random.randint(0, 31)

    return NeuronAttrs(
        **{
            "reset_mode": reset_mode,
            "reset_v": reset_v,
            "leak_comparison": leak_comparison,
            "threshold_mask_bits": threshold_mask_bits,
            "neg_thres_mode": neg_thres_mode,
            "neg_threshold": neg_threshold,
            "pos_threshold": pos_threshold,
            "leak_direction": leak_direction,
            "leak_integration_mode": leak_integration_mode,
            "leak_v": leak_v,
            "synaptic_integration_mode": synaptic_integration_mode,
            "bit_truncation": bit_truncation,
            "vjt_init": 0,
        }
    )


@pytest.fixture(scope="class")
def gen_NeuronDestInfo():
    addr_chip_x, addr_chip_y = random.randint(0, 31), random.randint(0, 31)
    addr_core_x, addr_core_y = random.randint(0, 31), random.randint(0, 31)
    addr_core_x_ex, addr_core_y_ex = random.randint(0, 31), random.randint(0, 31)

    n = 100
    tick_relative = [0] * n
    addr_axon = random.sample(list(range(1152)), n)

    return NeuronDestInfo(
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
