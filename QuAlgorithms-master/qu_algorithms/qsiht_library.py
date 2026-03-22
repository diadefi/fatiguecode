#
# General Fast Path for QsiHT (Quantum Signal-Induced Heap Transform)
#
# Alexis A. Gomez (0009-0003-0592-8553)
# alexis.gomez@utsa.edu
#
# Artyom M. Grigoryan (0000-0001-6683-0064)
# artyom.grigoryan@utsa.edu
#
# University of Texas at San Antonio
# Electrical and Computer Engineering Department
#
# March 17, 2025
#
# Paper must be reference when using this code.
# Grigoryan, A.M.; Gomez, A.; Espinoza, I.; Agaian, S.S. Signal-Induced Heap Transform-Based QR-Decomposition and Quantum Circuit for Implementing 3-Qubit Operations. Information 2025, 16, 466. https://doi.org/10.3390/info16060466
# 

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
import numpy as np


## Auxiliary Functions

def generate_binary_values(bits):
    # Calculate the maximum value based on the number of bits
    max_value = (1 << bits) - 1

    # Generate binary strings starting from the highest to the lowest
    binary_values = [bin(i)[2:].zfill(bits) for i in range(max_value, -1, -1)]

    return binary_values


# # Example usage:
# bits = 2
# binary_values = generate_binary_values(bits)
# print(binary_values)


def generate_and_shift_array(size, shift):
    # Generate the initial array
    array = list(range(size - 1, -1, -1))

    # Perform the shift (rotate the array)
    shift = shift % size  # This ensures shift stays within the bounds of the array
    shifted_array = array[-shift:] + array[:-shift]

    return shifted_array


# # Example usage:
# size = 4
# shift = 0
# result = generate_and_shift_array(size, shift)
# print(result)

## QsiHT-FastPath General Code

def generalFastPath(qubits, angles=None):
    qc = QuantumCircuit(qubits, 0)

    num_rotation_gates = []
    binary_values = generate_binary_values(qubits - 1)
    binary_values = np.flip(binary_values)

    for i in range(qubits): num_rotation_gates.append(pow(2, i))

    k = np.sum(num_rotation_gates)
    theta = [0]*(k+1)

    if angles is not None:
        # Create a dictionary mapping each parameter to its fixed value
        angles = np.flip(angles)
        theta = angles

    for i in range(0, qubits):
        qc1 = QuantumCircuit(qubits, 0)
        # Reverse the loop to apply the angles in reverse order
        k = sum(num_rotation_gates[:i + 1]) - 1  # Start at the correct index for the given qubit
        for j in range(1, num_rotation_gates[i] + 1):
            # Apply the control gate with the correct theta[k] and control state
            qc1.append(RYGate(-2 * theta[k]).control(qubits - 1, ctrl_state=binary_values[j - 1]),
                       generate_and_shift_array(qubits, i))
            k -= 1  # Decrement k to ensure we're applying in reverse order

        qc.append(qc1.inverse(), range(qubits))

    return qc.reverse_bits()