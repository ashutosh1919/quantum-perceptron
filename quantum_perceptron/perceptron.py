from typing import Dict
from qiskit import QuantumCircuit, Aer, execute
from quantum_perceptron.utils import (
    assert_negative,
    assert_bits,
    create_hypergraph_state,
    get_vector_from_int
)


class Perceptron:
    def __init__(self,
                 num_qubits: int,
                 weight: int = 1,
                 input: int = 1):
        """
        This class creates a quantum perceptron instance which has
        capability calculate input * weight. Note that we are not applying
        any non-linearity. Our perceptron design is as per
        https://arxiv.org/pdf/1811.02266.pdf

        Args:
          num_qubits: `int` denoting number of qubits in perceptron
          weight: `int` denoting the weight of the perceptron.
          input: `int` denoting the data to input to the perceptron.
        """
        self.num_qubits = num_qubits
        assert self.num_qubits > 0, "Number qubits must be positive"
        assert_negative(weight)
        self.weight = weight
        assert_negative(input)
        self.input = input
        assert_bits(self.weight, self.num_qubits)
        assert_bits(self.input, self.num_qubits)
        self.build_flag = False
        self.build_circuit()

    def Ui(self):
        """
        Sub-circuit to transform input data.
        """
        if not self.build_flag:
            raise RuntimeError("Ui() cannot be called independently.")

        Ui = QuantumCircuit(self.num_qubits)

        # Applying hadamard to first num_qubits
        for q in range(self.num_qubits):
            Ui.h(q)

        # Extracting vectors for input
        input_vector = get_vector_from_int(self.input, self.num_qubits)

        # Applying hypergraph state corresponding to input.
        Ui = create_hypergraph_state(Ui,
                                     input_vector,
                                     self.num_qubits)
        Ui = Ui.to_gate()
        Ui.name = "U_i"
        return Ui

    def Uw(self):
        """
        Sub-circuit to transform weight data.
        """
        if not self.build_flag:
            raise RuntimeError("Ui() cannot be called independently.")

        Uw = QuantumCircuit(self.num_qubits)

        # Extracting vectors for weight
        input_vector = get_vector_from_int(self.weight, self.num_qubits)

        # Applying hypergraph state corresponding to weight.
        Uw = create_hypergraph_state(Uw,
                                     input_vector,
                                     self.num_qubits)

        # Applying hadamard to first num_qubits
        for q in range(self.num_qubits):
            Uw.h(q)

        # Applying X gate to first num_qubits
        for q in range(self.num_qubits):
            Uw.x(q)
        Uw = Uw.to_gate()
        Uw.name = "U_w"
        return Uw

    def build_circuit(self):
        """
        Build quantum circuit corresponding to single perceptron combining
        input data and weight of the perceptron.
        """
        # Creating circuit with num_qubits + 1 (ancilla) qubit.
        self.circuit = QuantumCircuit(1 + self.num_qubits, 1)

        def toggle_build_flag():
            """
            Toggle the build circuit flag. Used to monitor Ui and Uf circuits
            to ensure that those functions are not called seperately but from
            the `build_circuit()` function.
            """
            self.build_flag = not self.build_flag

        # Append Ui for processing input
        toggle_build_flag()
        # self.Ui()
        self.circuit.append(
            self.Ui(),
            list(range(self.num_qubits))
        )
        toggle_build_flag()

        # Append Uf for processing input
        toggle_build_flag()
        self.circuit.append(
            self.Uw(),
            list(range(self.num_qubits))
        )
        toggle_build_flag()

        # Toffoli gate at the end with target as ancilla qubit
        self.circuit.mcx(
            control_qubits=list(range(self.num_qubits)),
            target_qubit=self.num_qubits
        )

        # Measure the last qubit.
        self.circuit.measure(self.num_qubits, 0)

    def measure_circuit(self, num_iters: int = 1000) -> Dict[str, int]:
        """
        Measure the perceptron and get the counts of the final results.

        Args:
          num_iters: `int` denoting number of iterations to execute circuit.

        Returns: `dict` containing the measurement frequencies.
        """
        if not hasattr(self, 'circuit'):
            raise RuntimeError("The circuit hasn't yet built.",
                               "Please call build_circuit() first.")
        backend = Aer.get_backend('qasm_simulator')

        # Execute the circuit
        job = execute(self.circuit, backend, shots=num_iters)

        # Get result and counts
        result = job.result()
        counts = result.get_counts(self.circuit)
        return dict(counts)

    def save_circuit_image(self,
                           file_path: str,
                           output_format: str = "mpl"):
        """
        Save circuit to the image file.
        """
        if not hasattr(self, 'circuit'):
            raise RuntimeError("The circuit hasn't yet built.",
                               "Please call build_circuit() first.")
        self.circuit.draw(output=output_format, filename=file_path)
