from typing import Literal

__all__ = ["HwParams"]


class HwParams:
    """Basic hardware configuration of PAICORE."""

    COORD_Y_PRIORITY: bool = True
    """Coordinate priority"""

    WEIGHT_BITORDER: Literal["little", "big"] = "little"
    N_CHIP_MAX = 1024
    N_BIT_COORD_ADDR = 5  # Address for X/Y chip/core/core* coordinates

    CORE_X_MIN = 0
    CORE_X_MAX = (1 << N_BIT_COORD_ADDR) - 1
    CORE_Y_MIN = 0
    CORE_Y_MAX = (1 << N_BIT_COORD_ADDR) - 1

    N_CORE_MAX_INCHIP = 1024
    N_CORE_OFFLINE = 1008
    N_CORE_ONLINE = N_CORE_MAX_INCHIP - N_CORE_OFFLINE

    CORE_X_OFFLINE_MIN = CORE_X_MIN
    CORE_Y_OFFLINE_MIN = CORE_Y_MIN
    CORE_X_OFFLINE_MAX = CORE_X_MAX
    CORE_Y_OFFLINE_MAX = CORE_Y_MAX
    CORE_X_ONLINE_MIN = 0b11100  # 28
    CORE_Y_ONLINE_MIN = 0b11100  # 28
    CORE_X_ONLINE_MAX = CORE_X_MAX
    CORE_Y_ONLINE_MAX = CORE_Y_MAX

    N_FANIN_PER_DENDRITE_MAX = 1152
    N_FANIN_PER_DENDRITE_SNN = N_FANIN_PER_DENDRITE_MAX
    N_FANIN_PER_DENDRITE_ANN = 144  # in 8-bit
    """The #N of fan-in per dendrite."""

    N_DENDRITE_MAX_SNN = 512
    N_DENDRITE_MAX_ANN = 4096
    """The maximum #N of dendrites in one core."""

    N_NEURON_MAX_SNN = 512
    N_NEURON_MAX_ANN = 1888
    """The maximum #N of neurons in one core."""

    ADDR_RAM_MAX = N_NEURON_MAX_SNN - 1
    """The maximum RAM address (starting from 0)."""

    ADDR_AXON_MAX = N_FANIN_PER_DENDRITE_MAX - 1
    """The maximum axons address (starting from 0)."""

    N_TIMESLOT_MAX = 1 << 8
    N_OFFLINE_TIMESLOT_MAX = 1 << 8

    N_ROUTING_PATH_LENGTH_MAX = 5
    N_SUB_ROUTING_NODE = 4
    """The number of sub-level routing nodes of a routing node."""

    FANOUT_IW8: list[int] = [N_NEURON_MAX_ANN, 1364, 876, 512, 256, 128, 64, 32, 16, 8]
    """The fan-out of 8-bit input width under different combination rate of dendrites(LCN + WW)."""
