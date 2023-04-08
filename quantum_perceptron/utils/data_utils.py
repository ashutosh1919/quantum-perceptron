import numpy as np
from typing import Dict, List, Optional


def assert_negative(data: int):
    """
    General method to prevent negative values of integers.
    """
    if data < 0:
        raise ValueError("Currently we do not support negative data values.")


def get_bin_int(data: int, num_qubits: Optional[int] = None) -> str:
    """
    Get binary representation of integer.
    """
    assert_negative(data)
    if num_qubits:
        return bin(data)[2:].zfill(np.power(2, num_qubits))
    return bin(data)[2:]


def assert_bits(data: int, num_bits: int):
    """
    General method to prevent invalid number of bits.
    """
    if len(get_bin_int(data)) > np.power(2, num_bits):
        raise ValueError("data has more bits than num_bits")


def get_vector_from_int(data: int, num_qubits: int) -> np.ndarray:
    """
    This method returns the vector where each element is (-1)^b_i where b_i is
    the bit value at index i.

    Args:
      data: `int` representing data value
        (correspponding toinput or weight vector)
      num_qubits: `int` representing number of qubits.

    Returns: Vector in  form of `np.ndarray`.
    """
    assert_negative(data)
    assert_bits(data, num_qubits)

    bin_data = get_bin_int(data, num_qubits)
    data_vector = np.empty(np.power(2, num_qubits))

    for i, bit in enumerate(bin_data):
        data_vector[i] = np.power(-1, int(bit))

    return data_vector


def get_possible_state_strings(num_bits: int) -> np.ndarray:
    """
    Get all the state bit strings corresponding to given number of bits.
    For example for 2 bits, we get ['00', '01', '10', '11'].
    Note that we are only allowing bits < 10 as of now.

    Args:
      num_bits: `int` representing number of bits

    Returns: `np.ndarray` containing all states of `num_bits` bits.
    """
    assert_negative(num_bits)
    if num_bits == 0:
        raise ValueError("Number of bits cannot be 0")
    if num_bits >= 10:
        raise ValueError("We are not currently supporting bits >= 10")

    total_states = np.power(2, num_bits)
    states = np.empty(total_states, dtype="<U10")
    state_template = "{:0" + str(num_bits) + "b}"
    for i in range(total_states):
        states[i] = state_template.format(i)

    return states


def get_ones_counts_to_states(states: np.ndarray) -> Dict[int, List[int]]:
    """
    Get the mapping from number of 1's to the states which has that many number
    of 1 bits.

    Args:
      states: `np.ndarray` containing the bit strings of the states.

    Returns: `dict` containing the mappings from count of 1's to the list
      of states.
    """
    if len(states) == 0:
        raise ValueError("The states array is empty")

    ones_count: Dict[int, List[int]] = dict()
    for i in range(len(states)):
        ct = states[i].count('1')
        if ct not in ones_count:
            ones_count[ct] = []
        ones_count[ct].append(i)

    return ones_count
