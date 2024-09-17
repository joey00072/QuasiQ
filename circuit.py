from __future__ import annotations


import numpy as np 

from gates import *
from density_matrix import DensityMatrix

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Instruction:
    name: str
    symbol: str
    qubits: List[int]
    params: Optional[List[float]] = None
    gate: Optional[np.ndarray] = None


@dataclass
class Circuit:
    num_qubits: int
    num_cbits: int = 1
    instructions: List[Instruction]|None = None
    density_matrix: DensityMatrix = None

    def __post_init__(self):
        self.instructions = []
        self.classical_bits = [[] for _ in range(self.num_cbits)]
        self.density_matrix = DensityMatrix(self.num_qubits)

    def _reset(self):
        self.classical_bits = [[] for _ in range(self.num_cbits)]
        self.density_matrix = DensityMatrix(self.num_qubits)

    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)

    def __str__(self):
        return str(self.instructions)

    def __repr__(self):
        return f"Circuit(num_qubits={self.num_qubits}, num_cbits={self.num_cbits}, instructions={self.instructions})"

    def h(self, qubit: int):
        self.add_instruction(Instruction(name="Hadamard", symbol="H", qubits=[qubit], gate=H))

    def x(self, qubit: int):
        self.add_instruction(Instruction(name="Pauli-X", symbol="X", qubits=[qubit], gate=X))

    def y(self, qubit: int):
        self.add_instruction(Instruction(name="Pauli-Y", symbol="Y", qubits=[qubit], gate=Y))

    def z(self, qubit: int):
        self.add_instruction(Instruction(name="Pauli-Z", symbol="Z", qubits=[qubit], gate=Z))

    def s(self, qubit: int):
        self.add_instruction(Instruction(name="Phase", symbol="S", qubits=[qubit], gate=S))

    def sdg(self, qubit: int):
        self.add_instruction(Instruction(name="Phase-Dagger", symbol="S†", qubits=[qubit], gate=np.conj(S.T)))

    def t(self, qubit: int):
        self.add_instruction(Instruction(name="T-Gate", symbol="T", qubits=[qubit], gate=T))

    def tdg(self, qubit: int):
        self.add_instruction(Instruction(name="T-Gate-Dagger", symbol="T†", qubits=[qubit], gate=np.conj(T.T)))

    def rx(self, theta: float, qubit: int):
        self.add_instruction(Instruction(name="Rotation-X", symbol="RX", qubits=[qubit], params=[theta], gate=RX(theta)))

    def ry(self, theta: float, qubit: int):
        self.add_instruction(Instruction(name="Rotation-Y", symbol="RY", qubits=[qubit], params=[theta], gate=RY(theta)))

    def rz(self, theta: float, qubit: int):
        self.add_instruction(Instruction(name="Rotation-Z", symbol="RZ", qubits=[qubit], params=[theta], gate=RZ(theta)))

    def u1(self, lambda_: float, qubit: int):
        self.add_instruction(Instruction(name="U1-Gate", symbol="U1", qubits=[qubit], params=[lambda_], gate=U1(lambda_)))

    def u2(self, phi: float, lambda_: float, qubit: int):
        self.add_instruction(Instruction(name="U2-Gate", symbol="U2", qubits=[qubit], params=[phi, lambda_], gate=U2(phi, lambda_)))

    def u3(self, theta: float, phi: float, lambda_: float, qubit: int):
        self.add_instruction(Instruction(name="U3-Gate", symbol="U3", qubits=[qubit], params=[theta, phi, lambda_], gate=U3(theta, phi, lambda_)))

    def cx(self, control: int, target: int):
        self.add_instruction(Instruction(name="Controlled-X", symbol="CX", qubits=[control, target], gate=X))

    def cz(self, control: int, target: int):
        self.add_instruction(Instruction(name="Controlled-Z", symbol="CZ", qubits=[control, target], gate=CZ))

    def ccx(self, control1: int, control2: int, target: int):
        self.add_instruction(Instruction(name="Toffoli", symbol="CCX", qubits=[control1, control2, target], gate=TOFFOLI))

    def measure(self, qubit: int, cbit: int):
        self.add_instruction(Instruction(name="Measurement", symbol="M", qubits=[qubit], params=[cbit]))

    def barrier(self):
        self.add_instruction(Instruction(name="Barrier", symbol="|", qubits=[]))

    def p(self, theta: float, qubit: int):
        self.add_instruction(Instruction(name="Phase", symbol="P", qubits=[qubit], params=[theta], gate=P(theta)))

    def cy(self, control: int, target: int):
        self.add_instruction(Instruction(name="Controlled-Y", symbol="CY", qubits=[control, target], gate=CY))

    def ch(self, control: int, target: int):
        self.add_instruction(Instruction(name="Controlled-H", symbol="CH", qubits=[control, target], gate=CH))

    def sqrtx(self, qubit: int):
        self.add_instruction(Instruction(name="Square-Root-X", symbol="√X", qubits=[qubit], gate=SQRTX))



    def run(self):
        for instruction in self.instructions:
            if instruction.gate is not None:
                if len(instruction.qubits) == 2:
                    # print(f"{instruction.name} {instruction.qubits[:-1]} {instruction.qubits[-1]}")
                    self.density_matrix.apply_controlled_gate(
                        control_qubits=instruction.qubits[:-1],
                        target_qubit=instruction.qubits[-1],
                        gate=instruction.gate)
                else:
                    self.density_matrix.apply_gate(instruction.gate, instruction.qubits[0])
            elif instruction.name == "Measurement":
                output = self.density_matrix.measure(instruction.qubits[0])
                self.classical_bits[instruction.params[0]].append(output)
        return self.density_matrix, self.classical_bits
    
    def execute(self, shots: int):
        results = []
        for qubit in range(self.num_qubits):
                self.measure(qubit, 0)

        for _ in range(shots):
            self._reset()
            self.run()
            results.append((self.classical_bits[0]))
        print(results)
        return np.array(results)
        


def bell_state():
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    results = circuit.execute(shots=10)
    print(results)

bell_state()
