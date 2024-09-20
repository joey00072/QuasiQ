import pytest
import numpy as np
from quasiq import Circuit

def quantum_teleportation(state_to_teleport):
    circuit = Circuit(3, 3)
    
    # Prepare the state to teleport
    if state_to_teleport == '1':
        circuit.x(0)
    
    # Create Bell pair
    circuit.h(1)
    circuit.cx(1, 2)
    
    # Perform Bell state measurement
    circuit.cx(0, 1)
    circuit.h(0)
    
    # Measure qubits 0 and 1
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    
    # Apply corrections on qubit 2
    circuit.cx(1, 2)
    circuit.cz(0, 2)
    
    # Measure the final state of qubit 2
    circuit.measure(2, 2)
    
    return circuit.execute(shots=1000)

def test_quantum_teleportation():
    # Test teleportation of |0⟩ state
    results_0 = quantum_teleportation('0')
    counts_0 = np.bincount(results_0[:, 2])
    prob_0_for_0 = counts_0[0] / np.sum(counts_0)
    
    # Test teleportation of |1⟩ state
    results_1 = quantum_teleportation('1')
    counts_1 = np.bincount(results_1[:, 2])
    prob_1_for_1 = counts_1[1] / np.sum(counts_1)
    
    # Assert that the teleportation works with high probability
    assert prob_0_for_0 > 0.9, f"Teleportation of |0⟩ failed. Probability of measuring |0⟩: {prob_0_for_0}"
    assert prob_1_for_1 > 0.9, f"Teleportation of |1⟩ failed. Probability of measuring |1⟩: {prob_1_for_1}"

if __name__ == "__main__":
    pytest.main()
