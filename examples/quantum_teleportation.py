from quasiq import Circuit
import numpy as np

def quantum_teleportation(state_to_teleport):
    # Create a circuit with 3 qubits
    circuit = Circuit(3,3) # init in zero states 

    if state_to_teleport == '1':
        circuit.x(0)
    elif state_to_teleport == '+':
        circuit.h(0)
    elif state_to_teleport == 'p':
        circuit.rx(np.pi/3, 0)  # State with 80% probability of measuring |0>
    
    # Create Bell pair between qubits 1 and 2
    circuit.h(1)
    circuit.cx(1, 2)

    # # Perform Bell state measurement on qubits 0 and 1
    circuit.cx(0, 1)
    circuit.h(0)

    # # Measure qubits 0 and 1
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    # # Apply corrections on qubit 2 based on measurement results
    circuit.cx(1, 2)
    circuit.cz(0, 2)

    # # Measure the final state of qubit 2
    circuit.measure(2, 2)
    circuit.print_circuit()

    # Execute the circuit
    results = circuit.execute(shots=1420, visualize=True)
    
    return results


if __name__ == "__main__":

    # Test the teleportation for |0⟩, |1⟩, |+⟩, and |p⟩ states
    # print("Teleporting |0⟩ state:")
    # results_0 = quantum_teleportation('0')
    # print("\nTeleporting |1⟩ state:")
    # results_1 = quantum_teleportation('1')
    # print("\nTeleporting |+⟩ state:")
    # results_plus = quantum_teleportation('+')
    print("\nTeleporting |p⟩ state (75% |0⟩, 25% |1⟩):")
    results_p = quantum_teleportation('p')

    # Analyze results
    def analyze_results(results):
        counts = np.bincount(results[:, 2], minlength=2)
        total = np.sum(counts)
        prob_0 = counts[0] / total
        prob_1 = counts[1] / total
        print(f"Probability of measuring |0⟩: {prob_0:.2f}")
        print(f"Probability of measuring |1⟩: {prob_1:.2f}")
    
    
    print("\n\n")
    print("=" * 40)
    print("Analysis of Results:")
    print("-" * 40)
    states = [
        # ("|0⟩", results_0, (1.0, 0.0)),
        # ("|1⟩", results_1, (0.0, 1.0)),
        # ("|+⟩", results_plus, (0.5, 0.5)),
        ("|p⟩", results_p, (0.75, 0.25))
    ]

    for state, results, expected in states:
        print(f"\nFor {state} state:")
        print(f"Expected: |0⟩ with probability {expected[0]:.2f}, |1⟩ with probability {expected[1]:.2f}")
        print("Actual:")
        analyze_results(results)
        print("-" * 40)
