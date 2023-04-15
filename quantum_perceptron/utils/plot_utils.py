import numpy as np
import matplotlib.pyplot as plt
from quantum_perceptron.utils.data_utils import (
    get_bin_int,
    assert_bits,
    assert_negative
)


def get_img_from_data(data: int, num_qubits: int) -> np.ndarray:
    """
    Get n x n matrix representing the image of the data where n is
    num_qubits.

    Args:
      data: `int` representing data value
        (correspponding to input or weight vector)
      num_qubits: `int` representing number of qubits.

    Returns: Image in  form of `np.ndarray`.
    """
    assert_negative(data)
    assert_bits(data, num_qubits)
    bin_str = get_bin_int(data, num_qubits)
    img = np.zeros((np.power(2, num_qubits)))

    for i, bit in enumerate(bin_str):
        if bit == '0':
            img[i] = 255

    return img.reshape((num_qubits, num_qubits))


def plot_img_from_data(data: int, num_qubits: int):
    """
    Plot image from data.
    """
    img = get_img_from_data(data, num_qubits)
    ax = plt.imshow(img, cmap='gray')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
