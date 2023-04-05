import pytest
import numpy as np

from quantum_perceptron.utils.data_utils import (
    get_vector_from_int,
    get_num_bits,
    get_possible_state_strings
)


@pytest.mark.parametrize("data, expected_result", [
    (12, 4),
    (8, 4),
    (7, 3),
    (0, 1),
    (-5, False),
])
def test_get_num_bits(data, expected_result):
    if isinstance(expected_result, bool) and not expected_result:
        with pytest.raises(ValueError):
            get_num_bits(data)
    else:
        assert expected_result == get_num_bits(data)


@pytest.mark.parametrize("data, expected_result", [
    (12, np.array([-1, -1, 1, 1])),
    (1, np.array([1, 1, 1, -1])),
    (0, np.array([1, 1, 1, 1])),
    (-5, False)
])
def test_get_vector_from_int(data, expected_result):
    if isinstance(expected_result, bool) and not expected_result:
        with pytest.raises(ValueError):
            get_vector_from_int(data)
    else:
        np.array_equal(
            expected_result,
            get_vector_from_int(data)
        )


@pytest.mark.parametrize("num_bits, expected_result", [
    (1, np.array(['0', '1'])),
    (2, np.array(['00', '01', '10', '11'])),
    (3, np.array(['000', '001', '010', '011', '100', '101', '110', '111'])),
    (-5, False),
    (0, False)
])
def test_get_possible_state_strings(num_bits, expected_result):
    if isinstance(expected_result, bool) and not expected_result:
        with pytest.raises(ValueError):
            get_possible_state_strings(num_bits)
    else:
        np.array_equal(
            expected_result,
            get_possible_state_strings(num_bits)
        )
