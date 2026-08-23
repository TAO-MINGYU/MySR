import numpy as np

import mysr


def test_version():
    assert mysr.__version__ == "0.1.0"


def test_config_defaults():
    c = mysr.SearchConfig()
    assert "+" in c.binary_operators
    assert c.niterations == 40


def test_shape_validation():
    try:
        mysr.fit(np.ones((3, 2)), np.ones(4))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")
