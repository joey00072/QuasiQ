from __future__ import annotations


import numpy as np

from quasiq.gates import *
from quasiq.density_matrix import DensityMatrix
from quasiq.utils import debug_print

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Instruction:
    name: str
    symbol: str
    qubits: List[int]
    params: Optional[List[float]] = None
    gate: Optional[np.ndarray] = None


class Symbol:
    BOX = '■'
    OPEN_BOX = '□'
    
@dataclass
class Circuit:
    num_qubits: int
    num_cbits: int = None
    instructions: List[Instruction] = None
    density_matrix: DensityMatrix = None

    def __post_init__(self):
        self.num_cbits = self.num_cbits or self.num_qubits 
        self.instructions = []
        self.quantums_bits = [[] for _ in range(self.num_cbits)]
        self.classical_bits = [[] for _ in range(self.num_cbits)]
        self.density_matrix = DensityMatrix(self.num_qubits)

    def _reset(self):
        self.quantums_bits = [[] for _ in range(self.num_qubits)]
        self.classical_bits = [[] for _ in range(self.num_cbits)]
        self.density_matrix = DensityMatrix(self.num_qubits)

    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)

    def __str__(self):
        return str(self.instructions)

    def __repr__(self):
        return f"Circuit(num_qubits={self.num_qubits}, num_cbits={self.num_cbits}, instructions={self.instructions})"

    def h(self, qubit: int):
        self.add_instruction(
            Instruction(name="Hadamard", symbol="H", qubits=[qubit], gate=H)
        )

    def x(self, qubit: int):
        self.add_instruction(
            Instruction(name="Pauli-X", symbol="X", qubits=[qubit], gate=X)
        )

    def y(self, qubit: int):
        self.add_instruction(
            Instruction(name="Pauli-Y", symbol="Y", qubits=[qubit], gate=Y)
        )

    def z(self, qubit: int):
        self.add_instruction(
            Instruction(name="Pauli-Z", symbol="Z", qubits=[qubit], gate=Z)
        )

    def s(self, qubit: int):
        self.add_instruction(
            Instruction(name="Phase", symbol="S", qubits=[qubit], gate=S)
        )

    def sdg(self, qubit: int):
        self.add_instruction(
            Instruction(
                name="Phase-Dagger", symbol="S†", qubits=[qubit], gate=np.conj(S.T)
            )
        )

    def t(self, qubit: int):
        self.add_instruction(
            Instruction(name="T-Gate", symbol="T", qubits=[qubit], gate=T)
        )

    def tdg(self, qubit: int):
        self.add_instruction(
            Instruction(
                name="T-Gate-Dagger", symbol="T†", qubits=[qubit], gate=np.conj(T.T)
            )
        )

    def rx(self, theta: float, qubit: int):
        self.add_instruction(
            Instruction(
                name="Rotation-X",
                symbol="RX",
                qubits=[qubit],
                params=[theta],
                gate=RX(theta),
            )
        )

    def ry(self, theta: float, qubit: int):
        self.add_instruction(
            Instruction(
                name="Rotation-Y",
                symbol="RY",
                qubits=[qubit],
                params=[theta],
                gate=RY(theta),
            )
        )

    def rz(self, theta: float, qubit: int):
        self.add_instruction(
            Instruction(
                name="Rotation-Z",
                symbol="RZ",
                qubits=[qubit],
                params=[theta],
                gate=RZ(theta),
            )
        )

    def u1(self, lambda_: float, qubit: int):
        self.add_instruction(
            Instruction(
                name="U1-Gate",
                symbol="U1",
                qubits=[qubit],
                params=[lambda_],
                gate=U1(lambda_),
            )
        )

    def u2(self, phi: float, lambda_: float, qubit: int):
        self.add_instruction(
            Instruction(
                name="U2-Gate",
                symbol="U2",
                qubits=[qubit],
                params=[phi, lambda_],
                gate=U2(phi, lambda_),
            )
        )

    def u3(self, theta: float, phi: float, lambda_: float, qubit: int):
        self.add_instruction(
            Instruction(
                name="U3-Gate",
                symbol="U3",
                qubits=[qubit],
                params=[theta, phi, lambda_],
                gate=U3(theta, phi, lambda_),
            )
        )

    def cx(self, control: int, target: int):
        self.add_instruction(
            Instruction(
                name="Controlled-X", symbol="CX", qubits=[control, target], gate=X
            )
        )

    def cz(self, control: int, target: int):
        self.add_instruction(
            Instruction(
                name="Controlled-Z", symbol="CZ", qubits=[control, target], gate=Z
            )
        )

    def ccx(self, control1: int, control2: int, target: int):
        self.add_instruction(
            Instruction(
                name="Toffoli",
                symbol="CCX",
                qubits=[control1, control2, target],
                gate=TOFFOLI,
            )
        )

    def measure(self, qubit: int, cbit: int):
        self.add_instruction(
            Instruction(name="Measurement", symbol="M", qubits=[qubit], params=[cbit])
        )

    def barrier(self):
        self.add_instruction(Instruction(name="Barrier", symbol="|", qubits=[]))

    def p(self, theta: float, qubit: int):
        self.add_instruction(
            Instruction(
                name="Phase", symbol="P", qubits=[qubit], params=[theta], gate=P(theta)
            )
        )

    def cy(self, control: int, target: int):
        self.add_instruction(
            Instruction(
                name="Controlled-Y", symbol="CY", qubits=[control, target], gate=CY
            )
        )

    def ch(self, control: int, target: int):
        self.add_instruction(
            Instruction(
                name="Controlled-H", symbol="CH", qubits=[control, target], gate=CH
            )
        )

    def sqrtx(self, qubit: int):
        self.add_instruction(
            Instruction(name="Square-Root-X", symbol="√X", qubits=[qubit], gate=SQRTX)
        )

    def run(self):

        lst = [0 for _ in range(self.num_qubits)]
        for instruction in self.instructions:
            debug_print(f"Executing: {instruction.name}, qubits: {instruction.qubits}, params: {instruction.params}")
            if instruction.name == "Measurement":
                lst[instruction.qubits[0]] = 1

        for idx, q in enumerate(lst):
            if q != 1:
                print(f"Measuring qubit {idx}")
                self.measure(idx, 0)

        debug_print("Starting circuit execution...")

        for instruction in self.instructions:
            if instruction.name == "Measurement":
                if self.quantums_bits[instruction.qubits[0]] == []:
                    output = self.density_matrix.measure(instruction.qubits[0])
                    self.quantums_bits[instruction.qubits[0]].append(output)
                    self.classical_bits[instruction.params[0]].append(output)
                else:
                    output = self.quantums_bits[instruction.qubits[0]][-1]
                    self.classical_bits[instruction.params[0]].append(output)
                debug_print(f"Qubit {instruction.qubits[0]} measured as {output}")

            elif instruction.gate is not None:
                debug_print(f"Applying gate: {instruction.name}, qubits: {instruction.qubits}, gate shape: {instruction.gate.shape}")
                if len(instruction.qubits) == 1:
                    self.density_matrix.apply_gate(
                        instruction.gate, instruction.qubits[0]
                    )
                else:
                    control_qubits = instruction.qubits[:-1]
                    target_qubit = instruction.qubits[-1]
                    if any((len(self.quantums_bits[cidx])) for cidx in control_qubits):
                        if len(control_qubits) == 1:
                            if self.quantums_bits[control_qubits[0]][-1] == 1:
                                self.density_matrix.apply_gate(instruction.gate, target_qubit)
                        elif len(control_qubits) == 2:
                            if self.quantums_bits[control_qubits[0]][-1] == 1 and self.quantums_bits[control_qubits[1]][-1] == 1:
                                self.density_matrix.apply_gate(instruction.gate, target_qubit)
                        elif len(control_qubits) == 3:
                            if self.quantums_bits[control_qubits[0]][-1] == 1 and self.quantums_bits[control_qubits[1]][-1] == 1 and self.quantums_bits[control_qubits[2]][-1] == 1:
                                self.density_matrix.apply_gate(instruction.gate, target_qubit)
                        debug_print(f"Applying controlled gate based on measured qubits")
                    else:
                        debug_print("Applying Controlled Quantum Gate")
                        self.density_matrix.apply_controlled_gate(
                            control_qubits=control_qubits,
                            target_qubit=target_qubit,
                            gate=instruction.gate,
                        )
                
        debug_print("Circuit execution completed")
        return self.density_matrix, self.classical_bits
    

    def print_state_probabilities(self, state_probabilities):
        for i, (state, probability) in enumerate(state_probabilities.items(), start=1):
            bar_length = int(probability * 20)
            bar = Symbol.BOX * bar_length + Symbol.OPEN_BOX * (20 - bar_length)
            percentage = probability * 100
            print(f"{i:2d}  |{state}⟩  {bar}  {percentage:.1f}% chance")


    def execute(self, shots: int=1, visualize: bool = False):
        results = []
        for _ in range(shots):
            self._reset()
            self.run()
            results.append([self.quantums_bits[qubit][-1] for qubit in range(self.num_qubits)])
            
        if visualize:
            debug_print("Visualizing results...")
            d = {}
            for result in results:
                s = "".join([str(i) for i in result])
                d[s] = d.get(s,0) + 1
            for k,v in d.items():
                d[k] = v/shots
            self.print_state_probabilities(d)
                
        return np.array(results)
    



if __name__ == "__main__":
    def bell_state():
        """
        simplest quantum entanglement
        
        https://cnot.io/quantum_computing/circuit_examples.html

        if first qbit is measured in |0⟩, second qbit will be |0⟩
        if first qbit is measured in |1⟩, second qbit will be |1⟩
        both qbits are maximally entangled
        so you wil either get 00 or 11 (but not 01 or 10)
        """
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        results = circuit.execute(shots=13)
        print(results)


    bell_state()



