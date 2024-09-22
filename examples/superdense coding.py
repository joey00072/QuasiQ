from quasiq import Circuit



def alice(circuit:Circuit,message:str):

    """
    we are passing circuit here but irl alice will recieve single qubit
    and will apply the gates based x and z gate based on the message

    and send her qubit to bob
    """

    if message[0] == '1':
        circuit.z(0)

    if message[1] == '1':
        circuit.x(0)
   

    return circuit



def bob_decode(circuit:Circuit):

    """
    bob will gate two qubits one at the start
    and 2nd sent by alice with dense
    """

    circuit.cx(0, 1)

    circuit.h(0)

    circuit.measure(0, 0)

    
    return circuit  
    
def superdense_coding(message:str):

    circuit = Circuit(num_qubits=2)

    circuit.h(0)

    circuit.cx(0, 1)

    circuit = alice(circuit,message)

    circuit = bob_decode(circuit)
   
    circuit.print_circuit()

    result = circuit.execute(shots=10,visualize=True)

    return result

if __name__ == "__main__":

    message = "00"
    result = superdense_coding(message)

    
    
