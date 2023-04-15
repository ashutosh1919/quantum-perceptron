import numpy as np
from tqdm import tqdm
from quantum_perceptron.utils import (
    calculate_succ_probability
)
from quantum_perceptron.perceptron import Perceptron


def generate_training_samples(data: np.ndarray,
                              num_positive_samples: int,
                              num_negative_samples: int) -> np.ndarray:
    """
    From the entire dataset, generate training samples.
    """
    pos_inds = np.where(data[:, 1] == 1)[0]
    neg_inds = np.where(data[:, 1] == 0)[0]

    if len(pos_inds) < num_positive_samples:
        num_positive_samples = len(pos_inds)
    if len(neg_inds) < num_negative_samples:
        num_negative_samples = len(neg_inds)

    sampled_neg_inds = np.random.choice(neg_inds,
                                        num_negative_samples,
                                        replace=False)
    sampled_pos_inds = np.random.choice(pos_inds,
                                        num_positive_samples,
                                        replace=False)

    new_data = np.vstack((data[sampled_pos_inds], data[sampled_neg_inds]))
    np.random.shuffle(new_data)
    return new_data


def generate_dataset(num_qubits: int = 4,
                     fixed_weight: int = 626,
                     dir_path: str = './data/',
                     threshold: float = 0.5,
                     num_runs: int = 8192,
                     create_training_samples: bool = True,
                     num_pos_train_samples: int = 50,
                     num_neg_train_samples: int = 3000):
    """
    Generate training dataset with fixed weight.

    Args:
        num_qubits: `int` representing number of qubits.
        fixed_weight: `int` representing the fixed weight value.
        dir_path: `str` representing the directory path.
    """
    num_samples = np.power(2, np.power(2, num_qubits))
    data = np.empty([num_samples, 2], dtype=np.int64)
    p = Perceptron(num_qubits, fixed_weight, 0)

    for i in tqdm(range(num_samples)):
        p.input = i
        p.build_circuit()
        prob = calculate_succ_probability(p.measure_circuit(num_runs))
        if prob > threshold:
            label = 1
        else:
            label = 0
        data[i][0] = i
        data[i][1] = label

    print("Number of positive samples: {}".format(
        np.sum(data[:, 1] == 1)
    ))
    print("Number of negative samples: {}".format(
        np.sum(data[:, 1] == 0)
    ))

    filename = 'sample_space_qubits_{}_fweight_{}.txt'.format(
        num_qubits, fixed_weight
    )
    np.savetxt(dir_path + filename, data, fmt='%i,%i', delimiter=',')
    print('Saved data to {}'.format(dir_path + filename))

    if create_training_samples:
        train_data = generate_training_samples(
            data, num_pos_train_samples, num_neg_train_samples
        )
        train_filename = 'train_space_qubits_{}_fweight_{}.txt'.format(
            num_qubits, fixed_weight
        )
        np.savetxt(dir_path + train_filename,
                   train_data,
                   fmt='%i,%i',
                   delimiter=',')
        print('Saved training data to {}'.format(dir_path + train_filename))
