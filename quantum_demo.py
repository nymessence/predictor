#!/usr/bin/env python3
"""
Demo script for quantum hex AI simulation
This demonstrates the quantum functionality without requiring API credentials
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np


def create_hex_lattice_demo():
    """Demonstrate the hex lattice quantum circuit creation"""
    print("Creating 19-qubit 7-hex lattice quantum circuit...")
    
    # Define the 19-qubit hexagonal lattice structure
    qc = QuantumCircuit(19)
    
    # Add initial Hadamard gates to create superposition
    for i in range(19):
        qc.h(i)
    
    # Define hexagonal connections (simulating 7 hexagons with shared vertices)
    connections = [
        # Central hexagon connections
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
        # Adjacent hexagon connections
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
        # Additional connections forming the 7-hex structure
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 6),
        # Outer connections
        (6, 12), (7, 13), (8, 14), (9, 15), (10, 16), (11, 17),
        # Final connections
        (12, 18), (13, 18), (14, 18), (15, 18), (16, 18), (17, 18)
    ]
    
    print(f"Circuit created with {qc.num_qubits} qubits and {len(connections)} connections")
    print("Applying alternating CNOT and Hadamard gates...")
    
    # Apply alternating pattern: Hadamard on even layers, CNOT on odd layers
    for d in range(3):  # Just 3 layers for demo
        if d % 2 == 0:
            # Apply Hadamard gates
            for i in range(qc.num_qubits):
                qc.h(i)
        else:
            # Apply CNOT gates along connections
            for control, target in connections:
                if control < qc.num_qubits and target < qc.num_qubits:
                    qc.cx(control, target)
    
    # Add measurement to classical registers
    cr = ClassicalRegister(19)
    qc.add_register(cr)
    qc.measure(range(19), range(19))
    
    print(f"Circuit depth: {qc.depth()}")
    print(f"Total operations: {qc.size()}")
    
    # Simulate the circuit
    print("Simulating quantum circuit...")
    simulator = AerSimulator()
    transpiled_circuit = transpile(qc, simulator)
    result = simulator.run(transpiled_circuit, shots=10).result()
    counts = result.get_counts(qc)
    
    print("Quantum simulation results:")
    for state, count in list(counts.items())[:5]:  # Show first 5 results
        prob = count / sum(counts.values())
        print(f"  State: {state}, Count: {count}, Probability: {prob:.4f}")
    
    return counts


if __name__ == "__main__":
    print("Quantum Hex Lattice Demo")
    print("=" * 40)
    results = create_hex_lattice_demo()
    print(f"\nTotal unique quantum states measured: {len(results)}")
    print("Demo completed successfully!")