import numpy as np
from typing import List, Dict
from qiskit import QuantumCircuit
from quantum_perceptron.utils.data_utils import (
    get_possible_state_strings,
    get_ones_counts_to_states
)


def append_hypergraph_state(
        circuit: QuantumCircuit,
        data_vector: np.ndarray,
        states: np.ndarray,
        ones_count: Dict[int, List[int]]) -> QuantumCircuit:
    """
    Append the computed hypergraph state to the circuit.

    Args:
      circuit: `QuantumCircuit` object corresponding to the perceptron.
      data_vector: `np.ndarray` containing the data vector containing -1s & 1s.
      states: `list` of `str` containing the bit strings for states.
      ones_count: `dict` containing mapping of the count of ones with
        index of states

    Returns: `QuantumCircuit` object denoting the circuit containing
      hypergraph states.
    """
    num_qubits = int(np.log2(len(data_vector)))
    is_sign_inverted = [1] * len(data_vector)

    # Flipping all signs if all zero state has coef -1.
    if data_vector[0] == -1:
        for i in range(len(data_vector)):
            data_vector[i] *= -1

    for ct in range(1, num_qubits + 1):
        for i in ones_count.get(ct, []):
            if data_vector[i] == is_sign_inverted[i]:
                state = states[i]
                ones_idx = [j for j, x in enumerate(state) if x == '1']
                if ct == 1:
                    circuit.z(ones_idx[0])
                elif ct == 2:
                    circuit.cz(ones_idx[0], ones_idx[1])
                else:
                    circuit.mcrz(
                        -np.pi,
                        [circuit.qubits[j] for j in ones_idx[1:]],
                        circuit.qubits[ones_idx[0]]
                    )
                for j, state in enumerate(states):
                    is_one = np.array([bit == '1' for bit in state])
                    if np.all(is_one[ones_idx]):
                        is_sign_inverted[j] *= -1
    return circuit


def create_hypergraph_state(circuit: QuantumCircuit,
                            data_vector: np.ndarray) -> QuantumCircuit:
    """
    Creating hypergraph state for specific data vector corresponding to
    the provided data (input or weight value).
    It is as per https://arxiv.org/abs/1811.02266.

    Args:
      circuit: `QuantumCircuit` object corresponding to the perceptron.
      data_vector: `np.ndarray` containing the data vector containing -1s & 1s.

    Returns: `QuantumCircuit` object denoting the circuit containing
      hypergraph states.
    """
    num_qubits = int(np.log2(len(data_vector)))
    states = get_possible_state_strings(num_qubits)
    ones_count = get_ones_counts_to_states(states)
    return append_hypergraph_state(
        circuit,
        data_vector,
        states,
        ones_count
    )
