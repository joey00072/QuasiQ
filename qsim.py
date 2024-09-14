import numpy as np

DEBUG = False

dtype = np.complex128  # double precision is needed

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# identity matrix
I = np.eye(2)

# pauli-x gate (not gate): σₓ = [[0, 1], [1, 0]]
X = np.array([[0, 1], [1, 0]], dtype=dtype)

# pauli-y gate: σy = [[0, -1j], [1j, 0]]
Y = np.array([[0, -1j], [1j, 0]], dtype=dtype)

# pauli-z gate: σz = [[1, 0], [0, -1]]
Z = np.array([[1, 0], [0, -1]], dtype=dtype)

# hadamard gate: H = (1/√2) * [[1, 1], [1, -1]]
H = np.array([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2)

# phase gate (s gate): S = [[1, 0], [0, i]]
S = np.array([[1, 0], [0, 1j]], dtype=dtype)

# pi/8 gate (t gate): T = [[1, 0], [0, exp(i*pi/4)]]
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=dtype)

# swap gate
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype)

# √not gate (square root of x)
SQRTX = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=dtype) / 2

# rotation gates
def RX(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=dtype)

def RY(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=dtype)

def RZ(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=dtype)

class DensityMatrix:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        dim = 2**n_qubits  # dimension of the density matrix for n_qubits
        pure_state = np.zeros((dim, dim), dtype=dtype)
        pure_state[0, 0] = 1  # initialize the state to |000...0> (ground state)
        self.state = pure_state

    def apply_gate(self, gate, target_qubit):
        """
        the gate matrix is expanded to act on the full system, then applied as: 
        ρ' = U * ρ * U† where U† is the hermitian conjugate (complex conjugate transpose) of U
        """
        full_gate = self._expand_gate(gate, target_qubit)
        self.state = full_gate @ self.state @ full_gate.conj().T

    def apply_controlled_gate(self, control_qubit, target_qubit, gate):
        """
        applies a controlled gate (e.g., cnot) between two qubits 
        the controlled gate is expanded to act on the full system, then applied as: 
        ρ' = C_U * ρ * C_U† where C_U is the controlled gate operation
        """
        full_controlled_gate = self._expand_controlled_gate(control_qubit, target_qubit, gate)
        self.state = full_controlled_gate @ self.state @ full_controlled_gate.conj().T  # complex conjugate transpose is the dagger †

    def _expand_gate(self, gate, target_qubit):
        """
        expands a single-qubit gate to act on the entire multi-qubit system.

        uses the kronecker product to construct the gate matrix for the full system:
        if the target_qubit is the ith qubit, the matrix is:
        U = I ⊗ ... ⊗ I ⊗ G ⊗ I ⊗ ... ⊗ I
        where G is the gate applied to the target qubit and ⊗ is the kronecker product
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
        
        for a 2-qubit system, the controlled-U gate has the form:
        C_U = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U
        where |0⟩⟨0| and |1⟩⟨1| are projectors onto the control qubit states.
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

        for a density matrix ρ, the probability of measuring |0⟩ on qubit i is:
        P(0) = Tr(P₀ᵢρ), where P₀ᵢ is the projector onto |0⟩ for qubit i.
        similarly, P(1) = Tr(P₁ᵢρ) for |1⟩.

        after measurement, the state collapses to:
        ρ' = (PᵢρPᵢ) / Tr(PᵢρPᵢ), where Pᵢ is the projector for the measured outcome.
        """
        assert 0 <= qubit < self.n_qubits, f"Invalid qubit index {qubit}. Must be between 0 and {self.n_qubits - 1}."

        dim = 2**self.n_qubits
        P_0 = np.zeros_like(self.state, dtype=dtype)
        P_1 = np.zeros_like(self.state, dtype=dtype)

        # calculate probabilities for measuring |0⟩ and |1⟩ on the qubit
        probabilities = np.zeros(2, dtype=dtype)

        for i in range(dim):
            if (i >> (self.n_qubits - 1 - qubit)) & 1:
                probabilities[1] += self.state[i, i].real
            else:
                probabilities[0] += self.state[i, i].real

        # normalize the probabilities and ensure they are real numbers
        probabilities = np.real(probabilities)
        probabilities /= np.sum(probabilities)  # ensure normalization

        # assert that probabilities sum to 1
        assert np.isclose(np.sum(probabilities), 1.0), f"probabilities do not sum to 1: {probabilities}"

        # perform the measurement based on the computed probabilities
        result = int(np.random.choice([0, 1], p=probabilities))
        debug_print(f"measured qubit {qubit}: {result}")

        # construct projectors p_0 (for |0⟩) and p_1 (for |1⟩)
        for i in range(dim):
            if (i >> (self.n_qubits - 1 - qubit)) & 1:
                P_1[i, i] = 1  # projector for |1⟩
            else:
                P_0[i, i] = 1  # projector for |0⟩

        # now we apply the projector to the density matrix which will give us the collapsed state
        # ie. new state after measurement
        if result == 0:
            new_state = P_0 @ self.state @ P_0  # p_0 projects onto |0⟩
        else:
            new_state = P_1 @ self.state @ P_1  # p_1 projects onto |1⟩

        normalization_factor = np.trace(new_state)
        assert normalization_factor > 0, "normalization factor must be positive."
        self.state = new_state / normalization_factor  # normalize the collapsed state

        return result


    def __repr__(self):
        return f"DensityMatrix(n_qubits={self.n_qubits})\nState:\n{self.state}"

if __name__ == "__main__":
    print("all qubits should collapse to the same state, either 0s or 1s")

    shots = 5
    for i in range(shots):
        dm = DensityMatrix(3)
        
        dm.apply_gate(H, 0)  # apply hadamard to qubit 0: |0⟩ → (|0⟩ + |1⟩)/√2
        
        dm.apply_controlled_gate(0, 1, X)  # cnot with control 0, target 1
        dm.apply_controlled_gate(1, 2, X)  # cnot with control 1, target 2
        
        # the state after gates is the ghz state: (|000⟩ + |111⟩)/√2
        
        # print("state after gates:")
        # print(dm)
        
        results = [dm.measure(i) for i in range(3)]
        print(f"{results=}")
    print("yay quantum entanglement!")
