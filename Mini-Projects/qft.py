# Import Required Packages/Libraries
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ
from qiskit.visualization import plot_bloch_multivector

# Request number of qubits
n = int(input("How many qubits would you like to generate circuits for? "))



''' FIRST ATTEMPT '''
# Swap all registers (necessary following the bulk of the QFT)
def swapALL(circuit, n):
    # Swap registers in pairs until the registers have been flipped
    for j in range(n // 2):
        circuit.swap(j, n - j - 1)

    return circuit

# Construct and return a circuit that will apply the 
# Quantum Fourier transform to some input state over n qubits
def qft(n):
    # Initialize circuit of n qubits
    circuit = QuantumCircuit(n)

    # For the jth qubit, apply a Hadamard followed by n - j - 1 controlled rotation gates
    for j in range(n):
        circuit.h(j)
        k = np.arange(2, n - j + 1)
        for m in range(j + 1, n):
            circuit.cp(2.0 * np.pi / np.power(2.0, k[m - j - 1]), m, j)

    # Swap registers to get outputs in the Fourier basis
    circuit = swapALL(circuit, n)

    # Display the final circuit for reference
    circuit.draw(output='mpl')
    plt.show()

qft(n)



''' QISKIT TEXTBOOK SOLUTION '''
def qft_rotations(circuit, n):
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(np.pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)

qc = QuantumCircuit(n)
qft_rotations(qc, n)
swapALL(qc, n)
qc.draw(output='mpl')
plt.show()



''' SECOND ATTEMPT '''
def qft_recursive(circuit, n, top=0):
    if n == 0:
        return circuit

    k = np.arange(2, n + 1)
    circuit.h(top)

    for j in range(n - 1):
        circuit.cp(2.0 * np.pi / np.power(2.0, k[j]), j + top + 1, top)
    
    top += 1
    n -= 1
    qft_recursive(circuit, n, top)

# Initialize circuit of n qubits
circuit = QuantumCircuit(n)

# Construct the Quantum Fourier Transform of n qubits
qft_recursive(circuit, n)

# Swap registers to get outputs in the Fourier basis
swapALL(circuit, n)

# Display the final circuit for reference
circuit.draw(output='mpl')
plt.show()



''' FINAL TEST '''
finalCircuit = QuantumCircuit(3)
finalCircuit.x(0)
finalCircuit.x(2)

sv_sim = Aer.get_backend("statevector_simulator")
qobj = assemble(finalCircuit)
statevector = sv_sim.run(qobj).result().get_statevector()
plot_bloch_multivector(statevector)
plt.show()

qft_recursive(finalCircuit, 3)

qobj = assemble(finalCircuit)
statevector = sv_sim.run(qobj).result().get_statevector()
plot_bloch_multivector(statevector)
plt.show()