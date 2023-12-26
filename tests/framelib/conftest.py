import os
import random
import tempfile
import pytest

from pathlib import Path

from paicorelib import *


@pytest.fixture
def ensure_dump_dir():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)

    yield p
    # Clean up
    # for f in p.iterdir():
    #     f.unlink()


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)


@pytest.fixture
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

    return dict(
        {
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
    )


@pytest.fixture
def gen_random_neuron_attr_dict():
    reset_mode = random.choice(list(RM))
    reset_v = random.randint(-(1 << 29), 1 << 29)
    leaking_comparison = random.choice(list(LCM))
    threshold_mask_bits = random.randint(0, 1 << 5)
    neg_thres_mode = random.choice(list(NTM))
    neg_threshold = random.randint(0, 1 << 29)
    pos_threshold = random.randint(0, 1 << 29)
    leaking_direction = random.choice(list(LDM))
    leaking_integration_mode = random.choice(list(LIM))
    leak_v = random.randint(-(1 << 29), 1 << 29)
    synaptic_integration_mode = random.choice(list(SIM))
    bit_truncate = random.randint(0, 1 << 5)
    vjt_init = random.randint(-(1 << 29), 1 << 29)

    return dict(
        {
            "reset_mode": reset_mode,
            "reset_v": reset_v,
            "leak_post": leaking_comparison,
            "threshold_mask_ctrl": threshold_mask_bits,
            "threshold_neg_mode": neg_thres_mode,
            "threshold_neg": neg_threshold,
            "threshold_pos": pos_threshold,
            "leak_reversal_flag": leaking_direction,
            "leak_det_stoch": leaking_integration_mode,
            "leak_v": leak_v,
            "weight_det_stoch": synaptic_integration_mode,
            "bit_truncate": bit_truncate,
            "vjt_pre": vjt_init,
        }
    )


@pytest.fixture
def gen_random_dest_info_dict():
    addr_chip_x, addr_chip_y = random.randint(0, 31), random.randint(0, 31)
    addr_core_x, addr_core_y = random.randint(0, 31), random.randint(0, 31)
    addr_core_x_ex, addr_core_y_ex = random.randint(0, 31), random.randint(0, 31)

    n = 100
    tick_relative = [0] * n
    addr_axon = random.sample(list(range(1152)), n)

    return dict(
        {
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


@pytest.fixture
def gen_random_one_input_proj():
    pass