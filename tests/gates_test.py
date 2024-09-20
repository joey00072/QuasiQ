import numpy as np
import pytest
from quasiq.gates import H, X, Y, Z, CNOT, CY, CZ, S, T, I
from quasiq.density_matrix import DensityMatrix

def basis_density_matrix(num_qubits, basis_state):
    dim = 2 ** num_qubits
    rho = np.zeros((dim, dim), dtype=np.complex128)
    rho[basis_state, basis_state] = 1.0
    return rho

def test_H_gate():
    dm = DensityMatrix(1)
    dm.apply_gate(H, 0)
    expected_state = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    assert np.allclose(dm.state, expected_state)

def test_X_gate():
    dm = DensityMatrix(1)
    dm.apply_gate(X, 0)
    expected_state = basis_density_matrix(1, 1)
    assert np.allclose(dm.state, expected_state)

def test_Y_gate():
    dm = DensityMatrix(1)
    dm.apply_gate(Y, 0)
    expected_state = basis_density_matrix(1, 1)
    assert np.allclose(dm.state, expected_state)

def test_Z_gate():
    dm = DensityMatrix(1)
    dm.apply_gate(Z, 0)
    expected_state = basis_density_matrix(1, 0)
    assert np.allclose(dm.state, expected_state)

def test_CNOT_gate():
    dm = DensityMatrix(2)
    dm.apply_gate(H, 0)
    dm.apply_controlled_gate([0], 1, X)
    expected_state = np.array([
        [0.5, 0, 0, 0.5],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0.5, 0, 0, 0.5]
    ], dtype=np.complex128)
    assert np.allclose(dm.state, expected_state)

def test_S_gate():
    dm = DensityMatrix(1)
    dm.apply_gate(H, 0)
    dm.apply_gate(S, 0)
    expected_state = np.array([
        [0.5, -0.5j],
        [0.5j, 0.5]
    ], dtype=np.complex128)
    assert np.allclose(dm.state, expected_state)

def test_T_gate():
    dm = DensityMatrix(1)
    dm.apply_gate(H, 0)
    dm.apply_gate(T, 0)
    
    # Numerical approximation of e^{iπ/4} / 2 ≈ 0.3536 + 0.3536j
    expected_state = np.array([
        [0.5, 0.35355339 - 0.35355339j],
        [0.35355339 + 0.35355339j, 0.5]
    ], dtype=np.complex128)
    
    assert np.allclose(dm.state, expected_state), f"Expected {expected_state}, but got {dm.state}"

def test_I_gate():
    dm = DensityMatrix(1)
    initial_state = dm.state.copy()
    dm.apply_gate(I, 0)
    assert np.allclose(dm.state, initial_state)

def test_multiple_gates():
    dm = DensityMatrix(2)
    dm.apply_gate(H, 0)
    dm.apply_controlled_gate([0], 1, X)
    dm.apply_gate(T, 1)
    
    # Calculate the phase introduced by the T gate
    phase = np.exp(1j * np.pi / 4)  # e^{iπ/4}
    
    # Expected density matrix after H, CNOT, and T
    expected_state = np.array([
        [0.5, 0, 0, np.exp(-1j * np.pi / 4) / 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [np.exp(1j * np.pi / 4) / 2, 0, 0, 0.5]
    ], dtype=np.complex128)
    
    assert np.allclose(dm.state, expected_state), f"Expected {expected_state}, but got {dm.state}"


if __name__ == "__main__":
    pytest.main()
