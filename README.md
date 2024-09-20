# QuasiQ

QuasiQ is a simple quantum computer simulator

```css
                                                                                
       ┌────────┐                       ┌───┐  ┌───┐                             
q_0:───┤ RX(π/3)├──────────────────■────┤ H ├──┤ M ├──────────────────■──────────
       └────────┘                  │    └───┘  └─╥─┘                  │          
                   ┌───┐         ┌─┴─┐           ║    ┌───┐           │          
q_1:───────────────┤ H ├────■────┤ X ├───────────║────┤ M ├────■──────│──────────
                   └───┘    │    └───┘           ║    └─╥─┘    │      │          
                          ┌─┴─┐                  ║      ║    ┌─┴─┐  ┌─┴─┐  ┌───┐ 
q_2:──────────────────────┤ X ├──────────────────║──────║────┤ X ├──┤ Z ├──┤ M ├─
                          └───┘                  ║      ║    └───┘  └───┘  └─╥─┘ 
                                                 0      1                    2   
```

Quantum Teleportation Example

```python

from quasiq import Circuit
import numpy as np


circuit = Circuit(3,3) # init in zero states 

circuit.rx(np.pi/3, 0)  # State with 75% probability of measuring |0>

# Create Bell pair between qubits 1 and 2
circuit.h(1)
circuit.cx(1, 2)

# # Perform Bell state measurement on qubits 0 and 1
circuit.cx(0, 1)
circuit.h(0)

# # Measure qubits 0 and 1
circuit.measure(0, 0)
circuit.measure(1, 1)

# # Apply corrections on qubit 2 based on measurement results
circuit.cx(1, 2)
circuit.cz(0, 2)

# # Measure the final state of qubit 2
circuit.measure(2, 2)
circuit.print_circuit()

# Execute the circuit
results = circuit.execute(shots=420)

ones = sum([r[2] for r in results])
print(f"Probability of measuring |0⟩: {1 - ones / len(results):.2f}")
print(f"Probability of measuring |1⟩: {ones / len(results):.2f}")

```


## Installation

```bash
git clone https://github.com/joey00072/quasiq.git
cd quasiq
pip install -e .
```


## Features

- Multi-qubit system simulation
- Basic quantum gates (X, H)
- Controlled gates (e.g., CNOT)
- Qubit measurement
- Density matrix representation

## Checklist

- [x] Implement basic quantum gates (X, H)
- [x] Implement controlled gates (CNOT)
- [x] Create DensityMatrix class for multi-qubit systems
- [x] Implement qubit measurement
- [x] Add example GHZ state creation and measurement
- [x] Implement additional quantum gates (Y, Z, S, T, etc.)
- [x] Create quantum circuit class
- [x] quantum teleportation example
- [ ] superdense coding example
- [ ] Add visualization tools for quantum states
- [ ] impliment basic algorithms (ghz, teleportation, etc.)
- [ ] Noise simulation
- [ ] QASM (OpenQASM) support
- [ ] tests :?
- [ ] Add error handling and input validation
- [ ] Implement state vector representation alongside density matrices


## Contributing

Contributions are welcome! Be nice and keep it clean.
