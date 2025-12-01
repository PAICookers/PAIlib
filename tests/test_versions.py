import sys

import paicorelib


def test_pailib_version():
    assert isinstance(paicorelib.__version__, str) and paicorelib.__version__ >= "0.0.1"


def test_numpy_version_on_zynq():
    if sys.platform == "armv7l":
        import numpy as np

        assert np.__version__ < "2.0.0"
