from __future__ import annotations


import numpy as np

from quasiq.gates import *
from quasiq.density_matrix import DensityMatrix
from quasiq.utils import debug_print
import math
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
    BOX_TOP_LEFT = '┌─'
    BOX_BOTTOM_LEFT = '└─'
    BOX_TOP_RIGHT = '─┐'
    BOX_BOTTOM_RIGHT = '─┘'
    MULTI_REG_TOP_LEFT = '╭─'
    MULTI_REG_BOTTOM_LEFT = '╰─'
    MULTI_REG_TOP_RIGHT = '─╮'
    MULTI_REG_BOTTOM_RIGHT = '─╯'
    VERTICAL_LINE = '│'
    HORIZONTAL_LINE = '─'
    LEFT_T = '┤'
    RIGHT_T = '├'
    TOP_T = '┴'
    BOTTOM_T = '┬'
    BULLET = '●'
    OPEN_BULLET = '○'
    BOX = '■'
    OPEN_BOX = '□'
    DOUBLE_VERTICAL_LINE = '║'
    DOUBLE_TOP = '╥'
    PI = 'π'
    
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

    def print_circuit(self):
        print_circuit(self)

    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)

    def __str__(self):
        return str(self.instructions)

    def __repr__(self):
        return PrintCircuit(self).print()
    
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
    


class PrintCircuit:
    def __init__(self, circuit: Circuit):
        self.lines = [[] for _ in range(circuit.num_qubits * 3 + 4)]
        self.top_padding = 3
        self.left_margin = "    "
        self.circuit = circuit

        self.top_lines = self.lines[0:self.top_padding]
        self.body_lines = self.lines[self.top_padding:]

        self.m = 1

    def add_left_padding(self):
        for ln in self.top_lines:
            ln.append(self.left_margin[1:])
        for idx, ln in enumerate(self.body_lines):
            ln.append(f"q_{idx // 3}:" if idx % 3 == 1 else self.left_margin)
        return self.lines    


    def add_space_padding(self: PrintCircuit):
        for idx,ln in enumerate(self.top_lines):
            ln.append(" ")

        for idx, ln in enumerate(self.body_lines):
            if idx % 3 == 1:
                ln.append(Symbol.HORIZONTAL_LINE)
            else:
                ln.append(" ")


    def add_barrier(self, instruction: Instruction, idx: int):
        
        for idx,ln in enumerate(self.top_lines):
            ln.append(" ")
        for idx,ln in enumerate(self.body_lines):
            ln.append("|")

    


    def add_single_qubit_gate(self, instruction: Instruction, idx: int,single=True):

        lines = [[] for _ in range(len(self.body_lines))]

        target_qbits = [instruction.qubits[-1]] if single else instruction.qubits
        # print(target_qbits)

        for idx,ln in enumerate(lines):
            qbit = idx//3
            if qbit in target_qbits:
                if idx % 3 == 0:
                    ln.append(f"{Symbol.BOX_TOP_LEFT}")
                elif idx % 3 == 1:
                    ln.append(f"{Symbol.LEFT_T}")
                elif idx % 3 == 2:
                    ln.append(f"{Symbol.BOX_BOTTOM_LEFT}")
            elif idx % 3 == 1:
                ln.append(f"{Symbol.HORIZONTAL_LINE}")
            else:
                ln.append(" ")

        inst = f" {instruction.symbol} "

        if instruction.symbol.lower()[0] == "c":
            inst = f" {instruction.symbol[-1].upper()} "
            
        if "Rotation" in instruction.name:
            frac = Fraction(instruction.params[0]/math.pi).limit_denominator()
            nu = frac.numerator
            de = frac.denominator
            arg_val = ""
            if nu>100 or de>100:
                arg_val = f"{instruction.params[0]:.4f}"
            else:
                if nu==1:
                    nu = ""
                arg_val = f"{nu}{Symbol.PI}/{de}"
            inst = f" {instruction.symbol}({arg_val})"
            
            if instruction.params[0] == 0:
                inst = f" {instruction.symbol}(0)"

        iln = len(inst) 

        for idx,ln in enumerate(lines):
            qbit = idx//3
            if qbit in target_qbits:
                if idx % 3 == 0:
                    ln.append(f"{Symbol.HORIZONTAL_LINE*(iln-2)}")
                elif idx % 3 == 1:
                    ln.append(inst)
                elif idx % 3 == 2:
                    ln.append(f"{Symbol.HORIZONTAL_LINE*(iln-2)}")
            elif idx % 3 == 1:
                ln.append(f"{Symbol.HORIZONTAL_LINE*iln}")
            else:
                ln.append(" "*iln)

        for idx,ln in enumerate(lines):
            qbit = idx//3
            if qbit in target_qbits:
                if idx % 3 == 0:
                    ln.append(f"{Symbol.BOX_TOP_RIGHT}")
                elif idx % 3 == 1:
                    ln.append(f"{Symbol.RIGHT_T}")
                elif idx % 3 == 2:
                    ln.append(f"{Symbol.BOX_BOTTOM_RIGHT}")
            elif idx % 3 == 1:
                ln.append(f"{Symbol.HORIZONTAL_LINE}")
            else:
                ln.append(" ")


        for bln,ln in zip(self.body_lines, lines):
            bln.append("".join(ln))


        for idx,ln in enumerate(self.top_lines):
            if idx%3 == 1:
                ln.append(f"{('m'+str(+self.m)).center((iln+2))}")
                self.m += 1
            else:
                ln.append(" "*(iln+2))

        return lines

    def add_multi_qubit_gate(self, instruction: Instruction, idx: int):
        lines = []
        for idx,ln in enumerate(self.add_single_qubit_gate(instruction, idx)):
            lines.append([c for c in ln])
        # return

        start = instruction.qubits[-1]
        
        
        for target in instruction.qubits[:-1]:
            n = start*3
            # print(start,target)
            if target < start:
            
                box_idx = len(lines[n])//2
                tlst = [s for s in lines[n][box_idx]]
                lines[n][box_idx] = ("" if len(lines[n][box_idx])==1 else Symbol.HORIZONTAL_LINE)+Symbol.TOP_T
                n-=1
                while n-1!= (target*3):
                    # print(f"{lines[n][box_idx] =}")
                    # if len(lines[n][box_idx]) != 1:
                    
                    slist = [s for s in lines[n][box_idx]]
                    if slist[len(slist)//2] == Symbol.BOX:
                        n-=1
                        continue
                    slist[len(slist)//2] = Symbol.VERTICAL_LINE
                    lines[n][box_idx] = "".join(slist)

                    n -=1
                if len(lines[n][box_idx]) != 1:
                    slist = [s for s in lines[n][box_idx]]
                    slist[len(slist)//2] = Symbol.BOX
                    lines[n][box_idx] = "".join(slist)
                else:
                    lines[n][box_idx] = Symbol.BOX

            if target > start:
                n = start*3 +2
                box_idx = len(lines[n])//2 
                lines[n][box_idx] = Symbol.BOTTOM_T
                n+=1
                while n-1!= (target*3):
                    # print(f"{lines[n][box_idx] =}")
                    # if len(lines[n][box_idx]) != 1:
                    slist = [s for s in lines[n][box_idx]]
                    if slist[len(slist)//2] == Symbol.BOX:
                        n+=1
                        continue
                    slist[len(slist)//2] = Symbol.VERTICAL_LINE
                    lines[n][box_idx] = "".join(slist)

                    n +=1
                if len(lines[n][box_idx]) != 1:
                    slist = [s for s in lines[n][box_idx]]
                    slist[len(slist)//2] = Symbol.BOX
                    lines[n][box_idx] = "".join(slist)
                else:
                    lines[n][box_idx] = Symbol.BOX
            
        
        for idx,ln in enumerate(self.body_lines):
            ln[-1] = "".join(lines[idx])
    

    def add_measurement(self, instruction: Instruction, idx: int):
        lines = []
        for idx,ln in enumerate(self.add_single_qubit_gate(instruction, idx)):
            lines.append([c for c in ln])
        start = instruction.qubits[-1] *3 +2
        
        
        flg = False
        lidx=1
        for idx,line in enumerate(lines):
            # print(f"{line=}")
            slist = [s for s in line[lidx]]
            if idx==start:
                slist[len(slist)//2] = Symbol.DOUBLE_TOP
            if idx>(start):
                slist[len(slist)//2] = Symbol.DOUBLE_VERTICAL_LINE
            if idx==len(lines)-1:
                slist[len(slist)//2] = str(instruction.params[-1])
            line[lidx] = "".join(slist)
            
            

        for idx,ln in enumerate(self.body_lines):
            ln[-1] = "".join(lines[idx])

    def add_instruction(self, instruction: Instruction, idx: int):
        
        if instruction.name == "Barrier":
            return self.add_barrier(instruction, idx)
        
        if instruction.name == "Measurement":
            return self.add_measurement(instruction, idx)
        
        if instruction.name == "SWAP-Gate":
            return self.add_single_qubit_gate(instruction, idx,single=False)
        
        if instruction.qubits and len(instruction.qubits) == 1:
            self.add_single_qubit_gate(instruction, idx)
        else:
            self.add_multi_qubit_gate(instruction, idx)




def print_circuit(circuit: Circuit):
    ctx = PrintCircuit(circuit)

    ctx.add_left_padding()
    ctx.add_space_padding()
    ctx.add_space_padding()


    for idx, instruction in enumerate(circuit.instructions):
        ctx.add_space_padding()
        ctx.add_instruction(instruction, idx)
        ctx.add_space_padding()

    for ln in ctx.lines:
        print("".join(ln).replace(" ", " "))


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
        circuit.print_circuit()
        results = circuit.execute(shots=13)
        
        # print(results)


    bell_state()



