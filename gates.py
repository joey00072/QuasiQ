from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from fractions import Fraction

dtype = np.complex128  # double precision is needed

# identity gate: leaves the quantum state unchanged
I = np.eye(2, dtype=dtype)

# pauli-x gate: flips the state of a qubit (not gate)
X = np.array([[0, 1],
              [1, 0]], dtype=dtype)

# pauli-y gate: rotates the qubit state around the y-axis of the bloch sphere
Y = np.array([[0, -1j],
              [1j, 0]], dtype=dtype)

# pauli-z gate: flips the phase of the qubit
Z = np.array([[1, 0],
              [0, -1]], dtype=dtype)

# hadamard gate: creates an equal superposition of |0⟩ and |1⟩ states
H = np.array([[1, 1],
              [1, -1]], dtype=dtype) / np.sqrt(2)

# s gate (phase gate): rotates the qubit state by 90 degrees around the z-axis
S = np.array([[1, 0],
              [0, 1j]], dtype=dtype)

# t gate: rotates the qubit state by 45 degrees around the z-axis
T = np.array([[1, 0],
              [0, np.exp(1j * np.pi / 4)]], dtype=dtype)

# swap gate: exchanges the states of two qubits
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=dtype)

# square root of x gate: rotates the qubit state by 90 degrees around the x-axis
SQRTX = np.array([[1 + 1j, 1 - 1j],
                  [1 - 1j, 1 + 1j]], dtype=dtype) / 2

# parameterized rotation gates
def RX(theta):
    """rotation around x-axis by angle theta"""
    return np.array(
        [[np.cos(theta / 2), -1j * np.sin(theta / 2)],
         [-1j * np.sin(theta / 2), np.cos(theta / 2)]],
        dtype=dtype,
    )

def RY(theta):
    """rotation around y-axis by angle theta"""
    return np.array(
        [[np.cos(theta / 2), -np.sin(theta / 2)],
         [np.sin(theta / 2), np.cos(theta / 2)]],
        dtype=dtype,
    )

def RZ(theta):
    """rotation around z-axis by angle theta"""
    return np.array(
        [[np.exp(-1j * theta / 2), 0],
         [0, np.exp(1j * theta / 2)]],
        dtype=dtype
    )

# universal single-qubit gates
def U1(lambda_):
    """single-qubit rotation about the z axis"""
    return np.array(
        [[1, 0],
         [0, np.exp(1j * lambda_)]],
        dtype=dtype
    )

def U2(phi, lambda_):
    """single-qubit rotation about the x+z axis"""
    return np.array(
        [[1, -np.exp(1j * lambda_)],
         [np.exp(1j * phi), np.exp(1j * (phi + lambda_))]],
        dtype=dtype
    ) / np.sqrt(2)

def U3(theta, phi, lambda_):
    """general single-qubit rotation"""
    return np.array(
        [[np.cos(theta/2), -np.exp(1j * lambda_) * np.sin(theta/2)],
         [np.exp(1j * phi) * np.sin(theta/2), np.exp(1j * (phi + lambda_)) * np.cos(theta/2)]],
        dtype=dtype
    )

# controlled-not (cnot) gate: flips the target qubit if the control qubit is |1⟩
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=dtype)

# toffoli (ccnot) gate: flips the target qubit if both control qubits are |1⟩
TOFFOLI = np.eye(8, dtype=dtype)
TOFFOLI[6:8, 6:8] = X

# controlled-z (cz) gate: applies a phase flip on the target qubit if the control qubit is |1⟩
CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]], dtype=dtype)

# phase gate (p): introduces a phase shift
def P(theta):
    """introduces a phase shift of e^(i*theta) to the |1⟩ state"""
    return np.array([[1, 0],
                     [0, np.exp(1j * theta)]], dtype=dtype)

# controlled-y (cy) gate: applies pauli-y on the target qubit if the control qubit is |1⟩
CY = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, -1j],
               [0, 0, 1j, 0]], dtype=dtype)

# controlled-h (ch) gate: applies hadamard on the target qubit if the control qubit is |1⟩
CH = np.eye(4, dtype=dtype)
CH[2:4, 2:4] = H
