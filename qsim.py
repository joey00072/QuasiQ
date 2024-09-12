import numpy as np

DEBUG = False

dtype = np.complex128  # double precision is needed

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

I = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=dtype)
H = np.array([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2)

# densitymatrix class for multi-qubit systems
class DensityMatrix:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        dim = 2**n_qubits  # dimension of the density matrix for n_qubits
        pure_state = np.zeros((dim, dim), dtype=dtype)
        pure_state[0, 0] = 1  # initialize the state to |000...0> (ground state)
        self.state = pure_state

    def apply_gate(self, gate, target_qubit):
        # the gate matrix is expanded to act on the full system, then applied as: 
        # U * ρ * U† where U† is the hermitian conjugate (complex conjugate transpose) of U
        full_gate = self._expand_gate(gate, target_qubit)
        self.state = full_gate @ self.state @ full_gate.conj().T

    def apply_controlled_gate(self, control_qubit, target_qubit, gate):
        # applies a controlled gate (e.g., cnot) between two qubits
        # the controlled gate is expanded to act on the full system, then applied as:
        # C_U * ρ * C_U† where C_U is the controlled gate operation
        full_controlled_gate = self._expand_controlled_gate(control_qubit, target_qubit, gate)
        self.state = full_controlled_gate @ self.state @ full_controlled_gate.conj().T  # complex conjugate transpose is the dagger †

    def _expand_gate(self, gate, target_qubit):
        """
        expands a single-qubit gate to act on the entire multi-qubit system.

        uses the kronecker product to construct the gate matrix for the full system:
        if the target_qubit is the ith qubit, the matrix is:
        I ⊗ ... ⊗ I ⊗ G ⊗ I ⊗ ... ⊗ I
        where G is the gate applied to the target qubit
        """
        result = 1  # start with scalar 1 for kronecker product
        for i in range(self.n_qubits):
            if i == target_qubit:
                result = np.kron(result, gate)  
            else:
                result = np.kron(result, I) 
        return result

    def _expand_controlled_gate(self, control_qubit, target_qubit, gate):
        """
        expands a controlled gate to act on the entire multi-qubit system.

        constructs the controlled gate matrix by applying the gate to the target qubit
        only if the control qubit is in state |1>. 
        """
        dim = 2**self.n_qubits  # dimension of the density matrix
        full_gate = np.eye(dim, dtype=dtype)  # start with an identity matrix of the system's full dimension
        
        for i in range(dim):
            # check if the control qubit is in state |1>
            if (i >> (self.n_qubits - 1 - control_qubit)) & 1:
                # if control qubit is |1>, apply the gate to the target qubit
                i_flipped = i ^ (1 << (self.n_qubits - 1 - target_qubit))  # flip target qubit
                full_gate[i, i] = gate[0, 0]
                full_gate[i, i_flipped] = gate[0, 1]
                full_gate[i_flipped, i] = gate[1, 0]
                full_gate[i_flipped, i_flipped] = gate[1, 1]
        
        return full_gate

    def measure(self, qubit):
        """
        measures a single qubit and collapses the density matrix accordingly.

        the measurement collapses the quantum state into one of the basis states based on the measured result.
        probabilities of measuring 0 or 1 for the specified qubit are calculated, and the state is updated:
        after measurement, the state collapses to reflect the observed outcome.
        """
        # calculate probabilities for each outcome
        dim = 2**self.n_qubits
        probabilities = np.zeros(2, dtype=dtype)
        
        for i in range(dim):
            if (i >> (self.n_qubits - 1 - qubit)) & 1:
                probabilities[1] += self.state[i, i].real
            else:
                probabilities[0] += self.state[i, i].real

        # normalize probabilities and ensure they are real numbers
        probabilities = np.real(probabilities)
        probabilities /= np.sum(probabilities)
        
        # measurement based on the computed probabilities
        result = int(np.random.choice([0, 1], p=probabilities))
        debug_print(f"measured qubit {qubit}: {result}")

        # collapse the state based on the measurement result
        new_state = np.zeros_like(self.state)
        for i in range(dim):
            if ((i >> (self.n_qubits - 1 - qubit)) & 1) == result:
                new_state[i, i] = self.state[i, i]

        # normalize 
        if np.sum(new_state) > 0:
            self.state = new_state / np.sum(new_state)  
        else:
            self.state = new_state
        
        return result

    def __repr__(self):
        return f"DensityMatrix(n_qubits={self.n_qubits})\nState:\n{self.state}"

if __name__ == "__main__":
    print("all qubits should collapse to the same state, either 0s or 1s")

    shots = 5
    for i in range(shots):
        dm = DensityMatrix(3)
        
        dm.apply_gate(H, 0)
        
        dm.apply_controlled_gate(0, 1, X)
        dm.apply_controlled_gate(1, 2, X)
        
        # print("state after gates:")
        # print(dm)
        
        results = [dm.measure(i) for i in range(3)]
        print(f"{results=}")
    print("yay quantum entanglement!")
