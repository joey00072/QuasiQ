from quasiq import Circuit



def balanced(circuit):
    circuit.cx(0,1)

def constant(circuit):
    ...



def deutsch(oracle):
    circuit = Circuit(2)

    circuit.x(1)

    circuit.h(0)
    circuit.h(1)


    oracle(circuit)

    circuit.h(0)


    circuit.measure(0,0)

    return circuit


if __name__ == "__main__":

    print("-"*50)
    print("Qubit 0 should be |1⟩, eg (|10⟩,|11⟩)")
    c1 = deutsch(balanced)
    results = c1.execute(100,visualize=True)
    print(f"function is balaced {sum([r[0] for r in results])}% probablity measued")

    print()
    print("-"*50)
    print("Qubit 0 should be |0⟩,, eg (|00⟩,|01⟩)")
    c1 = deutsch(constant)
    results = c1.execute(100,visualize=True)
    print(f"function is constant {100 - sum([r[0] for r in results])}% probablity measued")

