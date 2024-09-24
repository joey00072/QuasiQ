from quasiq import Circuit


# Balanced oracle for Deutsch-Jozsa problem (example)
def balanced(circuit):
    # Apply CNOT gates to simulate a balanced function
    for i in range(circuit.num_qubits - 1):
        circuit.cx(i, circuit.num_qubits - 1)


# Constant oracle for Deutsch-Jozsa problem (example)
def constant(circuit):
    # Do nothing for the constant function (or apply identity)
    pass


# Deutsch-Jozsa algorithm implementation
def deutsch_jozsa(oracle, num_qubits):
    circuit = Circuit(num_qubits + 1)  # Add one auxiliary qubit

    # Initialize the auxiliary qubit in the |1⟩ state
    circuit.x(num_qubits)

    # Apply Hadamard gates to all qubits
    for qubit in range(num_qubits + 1):
        circuit.h(qubit)

    # Apply the oracle
    oracle(circuit)

    # Apply Hadamard gates to the input qubits (not the auxiliary qubit)
    for qubit in range(num_qubits):
        circuit.h(qubit)

    # Measure all input qubits
    for qubit in range(num_qubits):
        circuit.measure(qubit, qubit)

    return circuit


if __name__ == "__main__":
    print("-" * 50)
    print("Testing balanced oracle: Qubits should measure a non-zero state")
    num_qubits = 3  # Example with 3 input qubits
    c1 = deutsch_jozsa(balanced, num_qubits)
    results = c1.execute(100, visualize=True)
    print(
        f"Function is balanced with {sum([r[0] for r in results])}% probability measured"
    )

    print()
    print("-" * 50)
    print("Testing constant oracle: Qubits should measure zero state")
    c2 = deutsch_jozsa(constant, num_qubits)
    results = c2.execute(100, visualize=True)
    print(
        f"Function is constant with {100 - sum([r[0] for r in results])}% probability measured"
    )
