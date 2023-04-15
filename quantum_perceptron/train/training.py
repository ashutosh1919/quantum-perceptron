import os
from typing import List
import numpy as np
from tqdm import tqdm
import wandb
from quantum_perceptron.utils import (
    get_vector_from_int,
    get_int_from_vector,
    calculate_succ_probability
)
from quantum_perceptron import Perceptron


class PerceptronTrainer:
    def __init__(self,
                 num_qubits: int,
                 fixed_weight: int,
                 dataset_path: str,
                 threshold: float = 0.5,
                 num_runs: int = 8192,
                 learning_rate_pos: float = 0.5,
                 learning_rate_neg: float = 0.5):
        """
        This class is used to train the perceptron.

        Args:
          num_qubits: `int` representing number of qubits.
          fixed_weight: `int` representing the fixed weight value.
          dataset_path: `str` representing the path to the dataset.
          threshold: `float` representing the threshold value.
          num_runs: `int` representing number of runs.
          learning_rate_pos: `float` representing the learning rate for positve
            samples.
          learning_rate_neg: `float` representing the learning rate for
            negativesamples.
        """
        self.num_qubits = num_qubits
        self.fixed_weight = fixed_weight
        assert os.path.exists(dataset_path), "Dataset path does not exist"
        self.data = self.read_dataset(dataset_path)
        self.threshold = threshold
        self.num_runs = num_runs
        self.learning_rate_pos = learning_rate_pos
        self.learning_rate_neg = learning_rate_neg
        self.perceptron = Perceptron(num_qubits)
        self.accumulate_loss: List[float] = []
        self.num_steps = 0

        # Initializing random weight for the training
        self.weight_variable = np.random.randint(
            np.power(2, np.power(2, num_qubits)))

        wandb.init(
            project="quantum-perceptron",
            config={
                "learning_rate_pos": learning_rate_pos,
                "learning_rate_neg": learning_rate_neg,
                "fixed_weight": fixed_weight,
                "num_qubits": num_qubits,
                "dataset": dataset_path,
                "num_runs": num_runs,
                "threshold": threshold
            }
        )

    def read_dataset(self, filepath: str) -> np.ndarray:
        """
        Read dataset from file.
        """
        return np.loadtxt(filepath, dtype=np.int64, delimiter=',')

    def invert_non_matching_bits(self, input: int):
        """
        Invert non-matching positions in vector.
        """
        input_vector = get_vector_from_int(input, self.num_qubits)
        weight_vector = get_vector_from_int(self.weight_variable,
                                            self.num_qubits)
        non_match_ids = np.where(input_vector != weight_vector)[0]
        num_select = int(np.ceil(len(non_match_ids) * self.learning_rate_pos))
        selected_ids = np.random.choice(non_match_ids,
                                        num_select,
                                        replace=False)
        for id in selected_ids:
            weight_vector[id] *= -1
        self.weight_variable = get_int_from_vector(weight_vector,
                                                   self.num_qubits)

    def invert_matching_bits(self, input: int):
        """
        Invert matching positions in vector.
        """
        input_vector = get_vector_from_int(input, self.num_qubits)
        weight_vector = get_vector_from_int(self.weight_variable,
                                            self.num_qubits)
        match_ids = np.where(input_vector == weight_vector)[0]
        num_select = int(np.ceil(len(match_ids) * self.learning_rate_neg))
        selected_ids = np.random.choice(match_ids,
                                        num_select,
                                        replace=False)
        for id in selected_ids:
            weight_vector[id] *= -1
        self.weight_variable = get_int_from_vector(weight_vector,
                                                   self.num_qubits)

    def calc_loss(self):
        """
        Note that we will only use this loss to generate the plot
        and not for training the perceptron.
        """
        self.perceptron.input = self.weight_variable
        self.perceptron.weight = self.fixed_weight
        self.perceptron.build_circuit()
        loss = calculate_succ_probability(
            self.perceptron.measure_circuit(self.num_runs))
        return loss

    def train_step(self, input: int, label: int):
        """
        Training step for a single sample.
        """
        self.perceptron.input = input
        self.perceptron.weight = self.weight_variable
        self.perceptron.build_circuit()
        prob = calculate_succ_probability(
            self.perceptron.measure_circuit(self.num_runs))
        loss = self.calc_loss()
        self.accumulate_loss.append(loss)
        self.num_steps += 1
        if int(loss) == 1:
            print("Training converged at step: {}".format(self.num_steps))
            return True
        if prob > self.threshold:
            pred = 1
        else:
            pred = 0
        if label == 1 and pred == 0:
            self.invert_non_matching_bits(input)
            wandb.log({"probability": loss, "weight": self.weight_variable})
        elif label == 0 and pred == 1:
            self.invert_matching_bits(input)
            wandb.log({"probability": loss, "weight": self.weight_variable})
        return False

    def train_epoch(self, epoch: int):
        """
        Train the epoch.
        """
        for i in tqdm(range(self.data.shape[0])):
            input = self.data[i, 0]
            label = self.data[i, 1]
            converged = self.train_step(input, label)
            if converged:
                return True
        return False

    def train(self, num_epochs: int):
        """
        Train the perceptron.
        """
        for i in range(num_epochs):
            converged = self.train_epoch(i)
            if converged:
                break
