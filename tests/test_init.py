import paicorelib


def test_pailib_version():
    assert isinstance(paicorelib.__version__, str) and paicorelib.__version__ >= "0.0.1"
