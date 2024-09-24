from quasiq import Circuit


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
    results = circuit.execute(shots=10, visualize=True)
    # print(results)


bell_state()
