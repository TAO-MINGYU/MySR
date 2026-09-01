import numpy as np
import pytest

from mysr.dimensions import (
    normalize_dimension,
    normalize_input_dimensions,
    normalize_output_dimensions,
)


def test_normalize_dimension_accepts_vector_mapping_and_numpy_array():
    length = (1, 0, 0, 0, 0, 0, 0)
    assert normalize_dimension(length) == length
    assert normalize_dimension({"length": 1}) == length
    assert normalize_dimension({"L": 1}) == length
    assert normalize_dimension(np.asarray(length)) == length


def test_normalize_dimension_rejects_physical_unit_strings():
    with pytest.raises(TypeError, match="unit strings are not accepted"):
        normalize_dimension("m")


def test_normalize_input_dimensions_preserves_feature_order():
    length = (1, 0, 0, 0, 0, 0, 0)
    time = (0, 0, 1, 0, 0, 0, 0)
    assert normalize_input_dimensions([length, time], 2) == [length, time]


def test_normalize_output_dimension_accepts_single_vector():
    area = (2, 0, 0, 0, 0, 0, 0)
    assert normalize_output_dimensions(area, 1) == area
    assert normalize_output_dimensions([area], 1) == [area]

