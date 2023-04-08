import pytest
import numpy as np

from quantum_perceptron.utils import (
    get_vector_from_int,
    get_bin_int,
    get_possible_state_strings,
    get_ones_counts_to_states
)


@pytest.mark.parametrize("data, num_qubits, expected_result", [
    (12, 4, '1100'),
    (12, 5, '01100'),
    (1, 1, '1'),
    (2, None, '10'),
    (-5, 2, False)
])
def test_get_bin_int(data, num_qubits, expected_result):
    if isinstance(expected_result, bool) and not expected_result:
        with pytest.raises(ValueError):
            get_bin_int(data, num_qubits)
    else:
        np.array_equal(
            expected_result,
            get_bin_int(data, num_qubits)
        )


@pytest.mark.parametrize("data, num_qubits, expected_result", [
    (12, 4, np.array([1]*12 + [-1, -1, 1, 1])),
    (12, 5, np.array([1]*28 + [-1, -1, 1, 1])),
    (1, 1, np.array([1, -1])),
    (12, 3, np.array([1]*4 + [-1, -1, 1, 1])),
    (16, 2, False),
    (-5, 2, False)
])
def test_get_vector_from_int(data, num_qubits, expected_result):
    if isinstance(expected_result, bool) and not expected_result:
        with pytest.raises(ValueError):
            get_vector_from_int(data, num_qubits)
    else:
        np.array_equal(
            expected_result,
            get_vector_from_int(data, num_qubits)
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


@pytest.mark.parametrize("states, expected_result", [
    (np.array(['0', '1']), {0: [0], 1: [1]}),
    (np.array(['00', '01', '10', '11']), {0: [0], 1: [1, 2], 2: [3]}),
    (np.array(['000', '001', '010', '011', '100', '101', '110', '111']), {
        0: [0],
        1: [1, 2, 4],
        2: [3, 5, 6],
        3: [7]
    }),
    (np.array([]), False)
])
def test_get_ones_counts_to_states(states, expected_result):
    if isinstance(expected_result, bool) and not expected_result:
        with pytest.raises(ValueError):
            get_ones_counts_to_states(states)
    else:
        assert expected_result == get_ones_counts_to_states(states)
