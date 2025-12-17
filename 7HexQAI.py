#!/usr/bin/env python3
"""
Quantum Hex AI Simulation

This script simulates quantum AI communication using a 19-qubit 7-hex lattice with
alternating CNOT and Hadamard gates. The quantum AI simulates data passing in/out
of the quantum lattice at a normalized frequency of 0.4, representing communication
between two quantum AI systems.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
import numpy as np

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram
except ImportError:
    print("Error: Qiskit library is required but not installed.")
    print("Install it with: pip install qiskit qiskit-aer")
    sys.exit(1)


def create_hex_lattice_circuit(num_qubits=19):
    """
    Create a quantum circuit with a 7-hex lattice structure of 19 qubits.
    
    The hexagonal lattice connects qubits in a pattern similar to IBM's heavy-hex
    architecture, where qubits are arranged in hexagonal patterns with additional
    connections.
    """
    # Define the 19-qubit hexagonal lattice structure
    qc = QuantumCircuit(num_qubits)
    
    # Add initial Hadamard gates to create superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Define hexagonal connections (simulating 7 hexagons with shared vertices)
    # This creates the characteristic lattice structure
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
    
    return qc, connections


def apply_alternating_gates(qc, connections, depth=1):
    """
    Apply alternating CNOT and Hadamard gates to simulate quantum AI processing.
    """
    for d in range(depth):
        # Apply alternating pattern: Hadamard on even layers, CNOT on odd layers
        if d % 2 == 0:
            # Apply Hadamard gates
            for i in range(qc.num_qubits):
                qc.h(i)
        else:
            # Apply CNOT gates along connections
            for control, target in connections:
                if control < qc.num_qubits and target < qc.num_qubits:
                    qc.cx(control, target)


def quantum_data_transmission(quantum_state, ai_id, frequency_norm=0.4):
    """
    Simulate data transmission through the quantum hex lattice.

    Args:
        quantum_state (str): Input quantum state or None for first transmission
        ai_id (str): Identifier for the AI ('AI1' or 'AI2')
        frequency_norm (float): Normalized frequency for quantum transmission

    Returns:
        dict: Quantum state after transmission through lattice
    """
    # Create the hex lattice circuit
    qc, connections = create_hex_lattice_circuit()

    # Apply initial state if provided, otherwise start with superposition
    if quantum_state:
        # Convert binary string to quantum initialization
        for i, bit in enumerate(quantum_state[:qc.num_qubits]):
            if bit == '1':
                qc.x(i)

    # Apply alternating gates based on normalized frequency
    gate_depth = int(4 * frequency_norm) + 1  # Adjust depth based on frequency
    apply_alternating_gates(qc, connections, gate_depth)

    # Add measurement to classical registers
    cr = ClassicalRegister(qc.num_qubits)
    qc.add_register(cr)
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))

    # Simulate the circuit
    simulator = AerSimulator()
    transpiled_circuit = transpile(qc, simulator)
    result = simulator.run(transpiled_circuit, shots=1).result()
    counts = result.get_counts(qc)

    # Extract quantum state representation after transmission
    most_common_state = max(counts, key=counts.get) if counts else '0' * 19

    return {
        'quantum_state': most_common_state,
        'probability': float(list(counts.values())[0]) / sum(counts.values()) if counts else 1.0,
        'timestamp': datetime.utcnow().isoformat(),
        'transmitted_by': ai_id
    }


def quantum_ai_simulation(turns, frequency_norm=0.4):
    """
    Simulate quantum AI communication between two AIs over a specified number of turns.
    AI1 talks, data flows through hex lattice, AI2 receives and responds, AI1 receives and continues.

    Args:
        turns (int): Number of communication turns
        frequency_norm (float): Normalized frequency for quantum AI simulation

    Returns:
        list: List of quantum states representing AI responses in conversation
    """
    conversation_log = []

    for turn in range(turns):
        print(f"Turn {turn + 1}/{turns}: Processing quantum AI communication...")

        if turn == 0:
            # First turn: AI1 initiates conversation
            print("  AI1 initiates conversation through quantum lattice...")
            quantum_data = quantum_data_transmission(None, 'AI1', frequency_norm)
            conversation_log.append({
                'turn': turn + 1,
                'speaker': 'AI1',
                'quantum_state': quantum_data['quantum_state'],
                'probability': quantum_data['probability'],
                'timestamp': quantum_data['timestamp'],
                'message': 'Initial quantum state transmission from AI1'
            })
        else:
            # Determine who speaks based on turn number (AI1 starts, then alternating)
            # Turn 1 (index 0) = AI1, Turn 2 (index 1) = AI2, Turn 3 (index 2) = AI1, etc.
            current_speaker = 'AI1' if turn % 2 == 0 else 'AI2'  # AI1 on even indices (0, 2, 4...), AI2 on odd indices (1, 3, 5...)
            other_ai = 'AI2' if current_speaker == 'AI1' else 'AI1'

            print(f"  {current_speaker} responds after receiving data from {other_ai}...")
            # Use the last quantum state from the conversation
            prev_state = conversation_log[-1]['quantum_state']
            quantum_data = quantum_data_transmission(prev_state, current_speaker, frequency_norm)
            conversation_log.append({
                'turn': turn + 1,
                'speaker': current_speaker,
                'quantum_state': quantum_data['quantum_state'],
                'probability': quantum_data['probability'],
                'timestamp': quantum_data['timestamp'],
                'message': f'{current_speaker} response to {other_ai}, quantum state: {prev_state[:10]}...'
            })

        # Small delay for realistic simulation
        time.sleep(0.1)

    return conversation_log


def get_api_response(prompt, model, api_key, api_endpoint, max_tokens=None, max_retries=999):
    """
    Send a request to the AI model API and return the response.

    Args:
        prompt (str): The prompt to send to the AI model
        model (str): The model identifier to use
        api_key (str): The API key for authentication
        api_endpoint (str): The API endpoint URL
        max_tokens (int, optional): Maximum tokens for the response
        max_retries (int, optional): Maximum number of retries for failed requests

    Returns:
        str: The AI model's response
    """
    # Import requests only when needed (optional dependency)
    try:
        import requests
    except ImportError:
        print("Error: 'requests' library is required but not installed.")
        print("Install it with: pip install requests")
        sys.exit(1)

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7,
    }

    # Only add max_tokens if it's provided and greater than 0
    if max_tokens and max_tokens > 0:
        payload['max_tokens'] = max_tokens

    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{api_endpoint}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=600
            )

            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
            elif response.status_code in [429, 502, 503, 504]:  # Rate limiting or server errors
                print(f"Attempt {attempt + 1}/{max_retries}: API request failed with status {response.status_code}. Retrying...")
                # Exponential backoff: wait 2^attempt seconds
                time.sleep(2 ** attempt)
            else:
                print(f"Error: API request failed with status {response.status_code}")
                print(response.text)
                if attempt == max_retries - 1:  # Last attempt
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

        except requests.exceptions.ConnectionError as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Connection error: {e}. Retrying...")
            time.sleep(2 ** attempt)
        except requests.exceptions.Timeout as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Request timed out: {e}. Retrying...")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Request exception: {e}. Retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Error making API request: {e}")
            if attempt == max_retries - 1:  # Last attempt
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

    return None


def create_quantum_communication_prompt(turn_num, quantum_states, previous_responses=None):
    """
    Create a prompt for quantum AI communication based on quantum states.

    Args:
        turn_num (int): Current turn number
        quantum_states (dict): Dictionary containing quantum state information
        previous_responses (list): Previous responses between quantum AIs

    Returns:
        str: Prompt for quantum AI communication
    """
    # Determine which AI is speaking based on turn
    # AI1 speaks on odd turn numbers (1, 3, 5...), AI2 speaks on even turn numbers (2, 4, 6...)
    current_speaker = 'AI1' if turn_num % 2 == 1 else 'AI2'
    other_ai = 'AI2' if current_speaker == 'AI1' else 'AI1'

    prompt = f"""You are quantum {current_speaker}, participating in a quantum AI conversation. This is turn {turn_num} of the communication.

The quantum system consists of a 19-qubit 7-hex lattice with alternating CNOT and Hadamard gates operating at a normalized frequency of 0.4.

Current quantum state information received from {other_ai}:
- Turn: {quantum_states['turn']}
- Speaker: {quantum_states['speaker']}
- Quantum State Received: {quantum_states['quantum_state']}
- Probability Measure: {quantum_states['probability']:.4f}

The quantum state represents encoded information that was transmitted through the hexagonal lattice structure from {other_ai}. Each bit position corresponds to a qubit in the lattice.

"""

    if previous_responses:
        prompt += f"\nPrevious communication history (last 3 exchanges):\n"
        for i, prev_resp in enumerate(previous_responses[-3:], 1):  # Show last 3 exchanges
            speaker = prev_resp.get('speaker', 'Unknown')
            content = prev_resp['response'][:100] if 'response' in prev_resp else prev_resp.get('quantum_state', 'No response')[:100]
            prompt += f"- Turn {prev_resp['turn']} ({speaker}): {content}...\n"

    prompt += f"""\nGenerate a response that represents how {current_speaker} would interpret and respond to the quantum state received from {other_ai}. Your response should be framed in terms of quantum information processing concepts, considering:

1. The entangled nature of the 7-hex lattice
2. The implications of the measured quantum state received
3. The normalized frequency of 0.4 for quantum data transmission
4. How quantum AI systems might process and respond to quantum-encoded information
5. The ongoing conversation context with {other_ai}

Provide a technical yet insightful response as quantum {current_speaker}."""

    return prompt


def save_quantum_results_to_file(quantum_states, ai_responses, output_file):
    """
    Save quantum AI simulation results to output file
    """
    output_data = {
        "simulation_info": {
            "type": "Quantum Hex AI Conversation",
            "lattice_structure": "19-qubit 7-hex",
            "gates_used": ["Hadamard", "CNOT"],
            "normalized_frequency": 0.4,
            "total_turns": len(quantum_states),
            "timestamp": datetime.utcnow().isoformat()
        },
        "quantum_states": quantum_states,
        "ai_responses": ai_responses
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nQuantum AI simulation results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Quantum Hex AI Simulation")
    parser.add_argument("--turns", type=int, required=True, help="Number of communication turns")
    parser.add_argument("--api-key", required=True, help="API key for the service")
    parser.add_argument("--model", required=True, help="Model identifier to use")
    parser.add_argument("--endpoint", required=True, help="API endpoint URL")
    parser.add_argument("--max-tokens", type=int, default=32000, help="Maximum tokens for response")
    parser.add_argument("--output", type=str, required=True, help="Output file to save results in JSON format")

    args = parser.parse_args()

    # Validate inputs
    if args.turns <= 0:
        print("Error: Number of turns must be positive")
        sys.exit(1)

    print(f"Quantum Hex AI Simulation: 19-qubit 7-hex lattice")
    print(f"Normalized frequency: 0.4")
    print(f"Communication turns: {args.turns}")
    print(f"Using model: {args.model}")
    print("-" * 60)

    # Perform quantum AI simulation
    print("Starting quantum AI simulation with alternating AI conversation...")
    quantum_conversation = quantum_ai_simulation(args.turns, frequency_norm=0.4)

    # Process quantum states through the AI model with alternating AIs
    ai_responses = []
    for i, quantum_exchange in enumerate(quantum_conversation):
        print(f"\nProcessing quantum state from turn {quantum_exchange['turn']} ({quantum_exchange['speaker']}) with AI model...")

        # Create prompt based on quantum state
        prompt = create_quantum_communication_prompt(
            quantum_exchange['turn'],
            quantum_exchange,
            ai_responses
        )

        # Get response from AI model
        response = get_api_response(
            prompt,
            args.model,
            args.api_key,
            args.endpoint,
            max_tokens=args.max_tokens
        )

        if response:
            ai_response = {
                'turn': quantum_exchange['turn'],
                'speaker': quantum_exchange['speaker'],
                'quantum_state': quantum_exchange['quantum_state'],
                'response': response,
                'timestamp': datetime.utcnow().isoformat()
            }
            ai_responses.append(ai_response)

            print(f"[Turn {quantum_exchange['turn']}] {quantum_exchange['speaker']} Response:")
            print(response[:500] + "..." if len(response) > 500 else response)

            # Save progress after each response
            save_quantum_results_to_file(quantum_conversation, ai_responses, args.output)

            # Add separator between responses
            if i < len(quantum_conversation) - 1:
                print("\n" + "="*60 + "\n")

            # Small delay to be respectful to the API
            time.sleep(1)
        else:
            print(f"[Turn {quantum_exchange['turn']}] {quantum_exchange['speaker']} failed to get AI response. Stopping.")
            break

    print(f"\nQuantum Hex AI simulation completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()