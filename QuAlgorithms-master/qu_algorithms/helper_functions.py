#
# Helper Functions for Testing and Validation
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

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import array_to_latex
from qiskit.quantum_info import Statevector

draw_settings = {'output': "mpl", 'style': "bw", 'scale': 1.5, 'fold': 12}


def DrawCircuit(circuit, name='circuit1', getArray=False, savefig=False):
    circuitDrawn = circuit.draw(**draw_settings)
    if savefig:
        circuitDrawn.savefig(str(name) + '.svg')
    if getArray:
        array = array_to_latex(Operator(circuit.reverse_bits()))
        return circuitDrawn, array

    return circuitDrawn


def removeReset(circuit):
    totalgates = sum(circuit.count_ops().values())

    nq = len(circuit.qubits)
    qc2 = QuantumCircuit(nq)

    for i in range(0, totalgates):

        gate1 = circuit[i]

        if gate1.operation.name not in {'reset'}:
            qc2.append(gate1)

    return qc2


def TranspileCircuit(circuit, target_basis=None):
    if target_basis is None:
        target_basis = ['rx', 'ry', 'rz', 'h', 'cx']

    from qiskit import transpile

    decomposed = transpile(circuit,
                           basis_gates=target_basis,
                           optimization_level=3)

    return decomposed


def GetUnitaryFromQC(circuit):
    totalgates = sum(circuit.count_ops().values())

    nq = len(circuit.qubits)
    qc2 = QuantumCircuit(nq)

    for i in range(0, totalgates):

        gate1 = circuit[i]

        if gate1.operation.name not in {'barrier', 'cx', 'reset'}:
            qc2.append(gate1)

        else:
            # print("Little Endian")
            display(qc2.draw("mpl", style="bw", scale=1.5))
            # display(array_to_latex(Operator(qc2)))
            qc2.draw("mpl", style="bw", scale=1.5).savefig('circuitdecom' + str(i) + '.svg')

            print("Big Endian")
            # display(qc2.reverse_bits().draw("mpl", style="bw", scale = 1.5))
            display(array_to_latex(Operator(qc2.reverse_bits())))

            nq = len(circuit.qubits)
            qc2 = QuantumCircuit(nq)


def GetInstructionsFromCircuit(circuit):
    for _instruction in circuit.data:
        if len(_instruction[0].params) > 0:
            print('\nInstruction:', _instruction[0].name)
            print('Params:', [str(_param) for _param in _instruction[0].params])


## Calculation Methods

def GetRMSE(stats, true_results_list, shots, ExportPrint=False):
    errors = []

    for set in stats:
        sum = 0
        i = 0
        for bit in set.values():
            sum += np.square(true_results_list[i] - bit)
            i += 1

        errors.append(np.sqrt(1 / i * sum))
    if ExportPrint:
        for shots, error in zip(shots, errors):
            print(f'RMSE for {shots} Shots:  {error}\n')

    return errors


def GetTheoreticalProb(circuit, reversebits=False):

    if reversebits:
        circuit = circuit.reverse_bits()

    def list_to_binary_dict(lst):
        num_bits = len(bin(len(lst) - 1)[2:])  # Determine the number of bits needed
        return {f"{i:0{num_bits}b}": item for i, item in enumerate(lst)}

    true_results = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    true_results = np.square( np.abs(true_results) )
    true_results = list_to_binary_dict(true_results)

    return true_results


def GetTheoreticalSV(circuit, reversebits=False):
    if reversebits:
        circuit = circuit.reverse_bits()

    def list_to_binary_dict(lst):
        num_bits = len(bin(len(lst) - 1)[2:])  # Determine the number of bits needed
        return {f"{i:0{num_bits}b}": item for i, item in enumerate(lst)}

    true_results = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    true_results = np.abs(true_results)
    true_results = list_to_binary_dict(true_results)

    return true_results

def getProbAndAmp():
    f = [1, 2, 3, 4, 5, 6, 7, 8] / np.linalg.norm([1, 2, 3, 4, 5, 6, 7, 8])
    fhat = np.fft.fft(f, 8) / np.sqrt(8)
    amplitude = np.round(np.conj(fhat), 4)
    print(amplitude)
    prob = np.absolute(amplitude)
    prob = np.round(np.power(prob, 2), 4)
    print(prob)


def runSampler(circuit, shots, true_results=None):
    from qiskit.primitives import Sampler
    from qiskit.visualization import plot_distribution

    results = []
    stats = []
    stats_histo = []
    legend = ['Theoretical']

    sampler = Sampler()
    for shot in shots:
        res = sampler.run(circuit, shots=shot).result()
        distri = res.quasi_dists[0]
        stats.append(distri)
        stats_histo.append(distri.binary_probabilities())
        results.append(res)
        legend.append(str(shot) + ' Shots')

    if true_results is not None:
        hist = plot_distribution([true_results] + stats_histo,
                                 legend=legend,
                                 title="Probabilities of States Per Different Runs", figsize=(40, 20), number_to_keep=2)
        return stats, hist

    return stats


def GetMRSE(stats, true_results_list, shots, probabilities=True, ExportPrint=False, scientific=False):
    errors = []
    size = len(true_results_list)
    for set in stats:
        set = dict(set)
        b = np.zeros(size)
        for index, val in set.items():
            b[index] = val
        if probabilities:
            tot_sum = np.square(np.abs(true_results_list) - b)
        else:
            tot_sum = np.square(np.sqrt(np.abs(true_results_list)) - np.sqrt(b))
        tot_sum = sum(tot_sum)
        errors.append(np.sqrt(tot_sum) / size)
    if ExportPrint:
        for shots, error in zip(shots, errors):
            if scientific:
                print(f'MRSE for {shots} Shots: {"{:.4e}".format(error)}')
            else:
                print(f'MRSE for {shots} Shots: {error}')

    return errors

def printErrors(qc, qc_shots=[10**3,10**4,10**5,10**6], plot=True, plot_prob=False):

    # Measure all qubits
    qc.measure_all()

    # Get the theoretical statevector and probabilities
    qc_true_res = GetTheoreticalProb(qc)

    # Run the Simulation
    qc_stats, qc_hist = runSampler(qc, qc_shots, qc_true_res)

    # Print Amplitude from Theoretical Probabilities
    print("Theoretical Amplitudes: [ ", end='')
    theoretical_amplitudes = []
    for prob in list(GetTheoreticalSV(qc).values()):

        prob_print = "{:.4e} ".format(prob)
        print(prob_print, end='')

        if plot_prob:
            theoretical_amplitudes.append( np.pow(prob,2) )
        else:
            theoretical_amplitudes.append(prob)

    print(end=']')

    # Print Amplitudes from Shot Probabilities
    print("\nAmplitudes for various shots:")
    shot_amplitudes = []
    for probs in qc_stats:
        probs = np.sqrt(list(probs.values()))
        sci_prob = []
        sci_prob2 = []
        for prob in probs:
            if plot_prob:
                sci_prob2.append( np.pow(prob,2) )
            else:
                sci_prob2.append(prob)
            prob = "{:.4e}".format(prob)
            sci_prob.append(prob)
        shot_amplitudes.append(sci_prob2)
        print(sci_prob)

    # Compute the MRSE of the Simulations
    GetMRSE(qc_stats, list(qc_true_res.values()), qc_shots, probabilities=False, ExportPrint=True, scientific=True)

    print("Theoretical Probabilities: [ ", end='')
    for prob in list(qc_true_res.values()):
        prob = "{:.4e} ".format(prob)
        print(prob, end='')
    print(end=']')

    print("\nProbabilities for various shots:")
    for probs in qc_stats:
        probs = list(probs.values())
        sci_prob = []
        for prob in probs:
            prob = "{:.4e}".format(prob)
            sci_prob.append(prob)
        print(sci_prob)

    # Compute the MRSE of the Simulations
    GetMRSE(qc_stats, list(qc_true_res.values()), qc_shots, ExportPrint=True, scientific=True)
    print()

    if plot:
        num_states = len(theoretical_amplitudes)

        # Find the max length
        max_len = max(len(sublist) for sublist in shot_amplitudes)

        # Pad each list with zeros
        shot_amplitudes = [sublist + [0.0] * (max_len - len(sublist)) for sublist in shot_amplitudes]

        shot_min = np.min( np.array( shot_amplitudes ) )
        shot_max = np.max( np.array( shot_amplitudes ) )

        x = [format(i, f'0{int(np.log2(num_states))}b') for i in range(num_states)]

        # Plot theoretical amplitudes
        plt.figure()
        plt.plot(x, theoretical_amplitudes, marker='o', linestyle='-', label='Theoretical', linewidth=2)

        # Define different markers and linestyles
        markers = ['o', 's', 'D', '^', 'v', '*', 'P', 'X']
        linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (1, 1))]

        # Plot sampled amplitudes for each shot count
        for idx, amps in enumerate(shot_amplitudes):
            marker = markers[idx % len(markers)]
            linestyle = linestyles[idx % len(linestyles)]
            plt.plot(x, amps, marker=marker, linestyle=linestyle, label=f'{qc_shots[idx]:,} shots')

        plt.xlabel('Quantum States')
        if plot_prob:
            plt.ylabel('Probabilities')
            plt.title('Comparison of Theoretical and Sampled Probabilities')
        else:
            plt.ylabel('Amplitude')
            plt.title('Comparison of Theoretical and Sampled Amplitudes')
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
        plt.tight_layout()
        plt.grid(False)
        plt.yticks(np.linspace(shot_min, shot_max, 5))

        plt.show()