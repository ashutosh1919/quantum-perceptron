import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
from quantum_perceptron import Perceptron

matplotlib.pyplot.switch_backend('Agg')


def run_perceptron(
        num_qubits: int,
        input_value: int,
        weight_value: int,
        num_iters: int
):
    p = Perceptron(num_qubits, weight_value, input_value)
    counts = p.measure_circuit(num_iters)
    prob_1 = counts.get('1', 0) / num_iters
    freq_hist = plt.figure()
    plt.bar(counts.keys(), counts.values(), width=0.5)
    for i, v in enumerate(list(counts.values())):
        plt.text(i, v+10, v)
    plt.xlabel('Measured State')
    plt.ylabel('Frequency of Measured State')
    plt.tight_layout()
    return prob_1, freq_hist


app_inputs = [
    gr.Slider(1, 9, value=2, step=1, label="Number of Qubits"),
    gr.Number(value=12, label="Input Value", precision=0),
    gr.Number(value=13, label="Weight Value", precision=0),
    gr.Number(value=1000,
              label="Number of Measurement Iterations",
              precision=0),
]

app_outputs = [
    gr.Number(precision=2, label="Probability of Firing Perceptron"),
    gr.Plot(label="Distribution of Measurement Frequencies")
]

demo = gr.Interface(
    fn=run_perceptron,
    inputs=app_inputs,
    outputs=app_outputs,
    title="Simulate Quantum Perceptron",
)
demo.launch()
