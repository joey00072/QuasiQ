from quasiq import Circuit
import numpy as np

# Define global shots
SHOTS = 2**12

def cu1(circuit, lam, control_qubit, target_qubit):
    """
    Decompose CU1(λ) into CNOT and Rz gates.
    """
    circuit.cx(control_qubit, target_qubit)
    circuit.rz(lam, target_qubit)
    circuit.cx(control_qubit, target_qubit)


def swap(circuit, qubit1, qubit2):
    """
    Decompose SWAP gate into three CNOT gates.
    """
    circuit.cx(qubit1, qubit2)
    circuit.cx(qubit2, qubit1)
    circuit.cx(qubit1, qubit2)


def apply_qft(circuit, qubits, swap_qubits=True):
    """
    Apply Quantum Fourier Transform to the specified qubits.
    """
    n = len(qubits)
    for i in range(n):
        circuit.h(qubits[i])
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            cu1(circuit, angle, qubits[j], qubits[i])
    if swap_qubits:
        for i in range(n // 2):
            swap(circuit, qubits[i], qubits[n - i - 1])


def apply_iqft(circuit, qubits, swap_qubits=True):
    """
    Apply Inverse Quantum Fourier Transform to the specified qubits.
    """
    n = len(qubits)
    if swap_qubits:
        for i in range(n // 2):
            swap(circuit, qubits[i], qubits[n - i - 1])
    for i in reversed(range(n)):
        for j in range(i + 1, n):
            angle = -np.pi / (2 ** (j - i))
            cu1(circuit, angle, qubits[j], qubits[i])
        circuit.h(qubits[i])


def quantum_fourier_transform(input_state, swap_qubits=True):
    """
    Perform QFT on the input state.
    """
    n = len(input_state)
    circuit = Circuit(n, n)

    # State Preparation
    for i, state in enumerate(input_state):
        if state == "1":
            circuit.x(i)
        elif state == "+":
            circuit.h(i)
        elif state == "p":
            circuit.rx(np.pi / 3, i)
        elif state != "0":
            raise ValueError(f"Unsupported state '{state}' for qubit {i}")

    # Apply QFT
    apply_qft(circuit, list(range(n)), swap_qubits)

    # Measurement
    for i in range(n):
        circuit.measure(i, i)

    # Execute and Return Results
    circuit.print_circuit()
    return circuit.execute(shots=SHOTS, visualize=True)


def quantum_inverse_fourier_transform(input_state, swap_qubits=True):
    """
    Perform IQFT on the input state.
    """
    n = len(input_state)
    circuit = Circuit(n, n)

    # State Preparation
    for i, state in enumerate(input_state):
        if state == "1":
            circuit.x(i)
        elif state == "+":
            circuit.h(i)
        elif state == "p":
            circuit.rx(np.pi / 3, i)
        elif state != "0":
            raise ValueError(f"Unsupported state '{state}' for qubit {i}")

    # Apply IQFT
    apply_iqft(circuit, list(range(n)), swap_qubits)

    # Measurement
    for i in range(n):
        circuit.measure(i, i)

    # Execute and Return Results
    circuit.print_circuit()
    return circuit.execute(shots=SHOTS, visualize=True)


def analyze_results(results, num_qubits):
    """
    Analyze and print measurement probabilities.
    """
    if isinstance(results, dict):
        total = sum(results.values())
        probabilities = {k: v / total for k, v in results.items()}
        for state, prob in sorted(probabilities.items()):
            print(f"|{state}⟩: {prob:.4f}")
    elif isinstance(results, np.ndarray):
        counts = np.bincount(
            results[:, :num_qubits].dot(1 << np.arange(num_qubits)[::-1]),
            minlength=2**num_qubits
        )
        total = counts.sum()
        for i in range(2**num_qubits):
            state = format(i, f'0{num_qubits}b')
            prob = counts[i] / total
            print(f"|{state}⟩: {prob:.4f}")
    else:
        print("Unsupported results format.")


def main():
    # Define input states excluding examples 5 and 7
    input_states = [
        ["0", "0"],
        ["0", "1"],
        ["1", "0"],
        ["1", "1"],
        ["+", "+"],
        ["p", "1"],
        ["1", "+"],
        ["+", "p"]
    ]

    # Select a state to test (avoiding examples 5 and 7)
    selected_state = ["+", "p"]  # Example 10
    num_qubits = len(selected_state)

    print(f"\n--- Quantum Fourier Transform (QFT) on {selected_state} ---")
    results_qft = quantum_fourier_transform(selected_state)
    print("\nQFT Measurement Probabilities:")
    analyze_results(results_qft, num_qubits)

    print(f"\n--- Inverse Quantum Fourier Transform (IQFT) on {selected_state} ---")
    results_iqft = quantum_inverse_fourier_transform(selected_state)
    print("\nIQFT Measurement Probabilities:")
    analyze_results(results_iqft, num_qubits)


if __name__ == "__main__":
    main()
