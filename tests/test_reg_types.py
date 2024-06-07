import itertools
import pytest
from paicorelib import InputWidthFormat, SpikeWidthFormat, SNNModeEnable, CoreMode


def test_CoreMode_instance():
    for iw, sw, snn_en in itertools.product(
        InputWidthFormat.__members__.values(),
        SpikeWidthFormat.__members__.values(),
        SNNModeEnable.__members__.values(),
    ):
        if iw is InputWidthFormat.WIDTH_8BIT and snn_en is SNNModeEnable.ENABLE:
            with pytest.raises(ValueError):
                cm = CoreMode((iw, sw, snn_en))

        else:
            cm = CoreMode((iw, sw, snn_en))
            assert cm.conf == (iw, sw, snn_en)
            assert cm.is_snn == snn_en
