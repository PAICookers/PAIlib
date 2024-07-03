import itertools

import pytest

from paicorelib import CoreMode, InputWidthFormat, SNNModeEnable, SpikeWidthFormat


def test_CoreMode_instance():
    for iw, sw, snn_en in itertools.product(
        InputWidthFormat, SpikeWidthFormat, SNNModeEnable
    ):
        if iw is InputWidthFormat.WIDTH_8BIT and snn_en is SNNModeEnable.ENABLE:
            with pytest.raises(ValueError):
                cm = CoreMode((iw, sw, snn_en))

        else:
            cm = CoreMode((iw, sw, snn_en))
            assert cm.conf == (iw, sw, snn_en)
            assert cm.is_snn == snn_en
