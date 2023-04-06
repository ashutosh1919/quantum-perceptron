import numpy as np


def assert_negative(data: int):
    """
    General method to prevent negative values of integers.
    """
    if data < 0:
        raise ValueError("Currently we do not support negative data values.")


def get_bin_int(data: int) -> str:
    """
    Get binary representation of integer.
    """
    assert_negative(data)
    return bin(data)[2:]


def get_num_bits(data: int) -> int:
    """
    Get number of bits in an integer.
    """
    assert_negative(data)
    return len(get_bin_int(data))


def get_vector_from_int(data: int) -> np.ndarray:
    """
    This method returns the vector where each element is (-1)^b_i where b_i is
    the bit value at index i.

    Args:
      data: `int` representing data value
        (correspponding toinput or weight vector)

    Returns: Vector in  form of `np.ndarray`.
    """
    assert_negative(data)

    bin_data = get_bin_int(data)
    num_qubits = get_num_bits(data)
    data_vector = np.empty(num_qubits)

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
