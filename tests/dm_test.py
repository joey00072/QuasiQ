from quasiq.density_matrix import DensityMatrix

import numpy as np


def test_density_matrix():
    dm = DensityMatrix(2)
    assert dm.n_qubits == 2
    assert dm.state.shape == (4, 4)
    assert dm.state.dtype == np.complex128

    # test the density matrix for a single qubit in the |0> state
    dm = DensityMatrix(1)
    dm.state = np.array([[1, 0], [0, 0]])
    assert np.allclose(dm.state, np.array([[1, 0], [0, 0]]))


    nqubits = 10
    dm = DensityMatrix(nqubits)
    s = np.zeros((2**nqubits, 2**nqubits), dtype=np.complex128)
    s[0,0] = 1
    dm.state = s
    assert np.allclose(dm.state, s)


