from quasiq import Circuit
import numpy as np
def test_circuit():
    circuit = Circuit(1)
    circuit.ry(np.arccos(1/np.sqrt(2)), 0)
    circuit.measure(0, 0)
    results = circuit.execute(shots=1, visualize=True)
    
    counts = np.bincount(results[:, 0])
    print(counts)
    
if __name__ == "__main__":
    test_circuit()