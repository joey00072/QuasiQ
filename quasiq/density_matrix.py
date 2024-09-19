import numpy as np

from quasiq.utils import tensor_product, debug_print

I = np.eye(2, dtype=dtype)

class DensityMatrix:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        dim = 2**n_qubits  # dimension of the density matrix for n_qubits
        pure_state = np.zeros((dim, dim), dtype=dtype)
        pure_state[0, 0] = 1  # initialize the state to |000...0> (ground state)
        self.state = pure_state

    # TODO: merge apply_gate and apply_controlled_gate
    def apply_gate(self, gate, target_qubit):
        """
        the gate matrix is expanded to act on the full system, then applied as: 
        ρ' = U * ρ * U† where U† is the hermitian conjugate (complex conjugate transpose) of U
        """
        full_gate = self.expand_gate_tensor_product(gate, target_qubit)
        self.state = full_gate @ self.state @ full_gate.conj().T

        debug_print(f"=========Single Gate=========")
        debug_print("\n",self.state/self.state.max(), 1/self.state.max())
        debug_print(f"==================")

    def apply_controlled_gate(self, control_qubits, target_qubit, gate):
        """
        applies a controlled gate (e.g., cnot) between two qubits 
        the controlled gate is expanded to act on the full system, then applied as: 
        ρ' = C_U * ρ * C_U† where C_U is the controlled gate operation
        """
        if isinstance(control_qubits, int):
            control_qubits = [control_qubits]  # make it a list if a single control qubit is passed

        full_controlled_gate = self.expand_gate_tensor_product(gate, target_qubit, control_qubits)
        self.state = full_controlled_gate @ self.state @ full_controlled_gate.conj().T

        debug_print(f"=========Controlled Gate=========")
        debug_print("\n",self.state/self.state.max(), 1/self.state.max())
        debug_print(f"==================")


    def expand_gate_tensor_product(
        self,
        gate: np.ndarray,
        target_qubit: int,
        control_bits: list[int] = []
    ) -> np.ndarray:
        
        num_qubits: int = self.n_qubits

        
        assert 0 <= target_qubit < num_qubits, f"target qubit index {target_qubit} out of range."
        assert target_qubit not in control_bits, f"target qubit {target_qubit} cannot be a control qubit."
        assert all(0 <= cb < num_qubits for cb in control_bits), f"control qubit index {control_bits} out of range."

        P1 = np.array([[0, 0], [0, 1]], dtype=dtype)  # |1⟩⟨1|
        I = np.eye(2, dtype=dtype)  
        
        # Pcontrol = I ⊗ I ⊗ P1 ⊗ P1 ⊗ I (for num_qubits = 5, control bit = (2,3))
        operators = [
            P1 if q in control_bits else I
            for q in range(num_qubits)
        ]
        P_control = tensor_product(operators) 

        # U = I ⊗ I ⊗ G ⊗ I ⊗ I (for num_qubits = 5, target_qubit = 2)
        gate_operators = [
            gate if q == target_qubit else I
            for q in range(num_qubits)
        ]    
        U = tensor_product(gate_operators)

        I_full = np.eye(2**num_qubits, dtype=dtype)

        debug_print(f"expanded gate shapes: P_control={P_control.shape}, U={U.shape}, I_full={I_full.shape}")
        
        # when there are no control qubits, P_control => I, so this reduces to U 
        C_U = P_control @ U + (I_full - P_control)

        return C_U

    def measure(self, qubit):
        """
        Measures a single qubit and collapses the density matrix accordingly using tensor products.

        Args:
            qubit (int): The index of the qubit to measure.

        Returns:
            int: The measurement result (0 or 1).
        """
        assert 0 <= qubit < self.n_qubits, f"Invalid qubit index {qubit}. Must be between 0 and {self.n_qubits - 1}."

        dim = 2**self.n_qubits

        # Define projectors |0><0| and |1><1| for a single qubit
        P0 = np.array([[1, 0], [0, 0]], dtype=dtype)
        P1 = np.array([[0, 0], [0, 1]], dtype=dtype)

        # Construct the full projectors using tensor products
        operators_P0 = [
            P0 if q == qubit else I
            for q in range(self.n_qubits)
        ]
        operators_P1 = [
            P1 if q == qubit else I
            for q in range(self.n_qubits)
        ]

        P0_full = tensor_product(operators_P0)
        P1_full = tensor_product(operators_P1)

        # Calculate probabilities
        P_0 = np.trace(P0_full @ self.state).real
        P_1 = np.trace(P1_full @ self.state).real

        # Normalize probabilities to avoid numerical issues
        total_prob = P_0 + P_1
        P_0 /= total_prob
        P_1 /= total_prob

        # Perform the measurement based on the computed probabilities
        result = int(np.random.choice([0, 1], p=[P_0, P_1]))
        debug_print(f"Measured qubit {qubit}: {result}")

        # Collapse the state
        if result == 0:
            new_state = P0_full @ self.state @ P0_full
        else:
            new_state = P1_full @ self.state @ P1_full

        # Normalize the collapsed state
        normalization_factor = np.trace(new_state)
        assert normalization_factor > 0, "Normalization factor must be positive."
        self.state = new_state / normalization_factor

        debug_print(f"=========Measure=========")
        debug_print("\n", self.state.real / self.state.max().real, 1 / self.state.max().real)
        debug_print(f"==================")

        return result

    def __repr__(self):
        return f"DensityMatrix(n_qubits={self.n_qubits})\nState:\n{self.state}"
    
    
if __name__ == "__main__":
    shots = 10

    for i in range(shots):
        dm = DensityMatrix(3)
        from quasiq.gates import *
        
        dm.apply_gate(H, 0)  # apply hadamard to qubit 0: |0⟩ → (|0⟩ + |1⟩)/√2
        dm.apply_gate(H, 1)
        
        # dm.apply_controlled_gate(0, 1, X)  # cnot with control 0, target 1
        # dm.apply_controlled_gate(1, 2, X)  # cnot with control 1, target 2
        
        # cxx
        dm.apply_controlled_gate([0, 1], 2, X)
        # print("state after gates:")
        # print(dm)
        
        results = [dm.measure(i) for i in range(3)]
        print(f"{results=}")
    print("yay quantum entanglement!")
