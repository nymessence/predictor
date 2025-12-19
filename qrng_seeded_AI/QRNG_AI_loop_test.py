#!/usr/bin/env python3
"""
Test script to verify QRNG AI loop functionality without making API calls
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
import struct
import random
import hashlib
from pathlib import Path


def get_utc_timestamp():
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def read_hex_values_from_file(file_path, num_blocks=1):
    """
    Read hex values from a file and convert each block of 8 hex values to uint64 integers.
    Following the requirement: "take each successive block of 8 hex values sequentially 
    then convert to unsigned 64 bit int". We interpret '8 hex values' as 8 bytes (16 hex characters),
    where each byte is represented by 2 hex digits. So each 16-character sequence represents
    8 bytes = 64 bits = 1 uint64 integer.
    
    Args:
        file_path (str): Path to the file containing hex values
        num_blocks (int): Number of 8-byte blocks to read (each 16 hex chars becomes 1 uint64)
        
    Returns:
        list: List of uint64 integer seeds
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()
        
    # Remove any spaces, newlines, and ensure we have just hex characters
    hex_chars = ''.join(content.split()).lower()
    
    # Each 8-byte block needs 16 hex characters (8 bytes * 2 hex chars per byte)
    required_chars = num_blocks * 16  # 16 hex chars per 8-byte block (64 bits)
    if len(hex_chars) < required_chars:
        print(f"Warning: Not enough hex characters in file. Expected {required_chars}, got {len(hex_chars)}")
        required_chars = (len(hex_chars) // 16) * 16  # Use what we have, rounded down to nearest 16 char boundary
    
    seeds = []
    for i in range(0, min(required_chars, len(hex_chars)), 16):  # process 16 hex chars at a time (8 bytes)
        hex_block = hex_chars[i:i+16]
        if len(hex_block) == 16:  # Only process full 8-byte blocks
            try:
                seed_value = int(hex_block, 16)
                seeds.append(seed_value)
            except ValueError:
                print(f"Warning: Could not parse hex block '{hex_block}' as integer. Skipping.")
    
    return seeds


def read_urandom_values(num_values=1):
    """
    Read random values from /dev/urandom and convert them to uint64 integers.
    
    Args:
        num_values (int): Number of 8-byte blocks to read
        
    Returns:
        list: List of uint64 integer seeds
    """
    seeds = []
    for _ in range(num_values):
        try:
            # Read 8 bytes (64 bits) from urandom
            random_bytes = os.urandom(8)
            # Unpack as little-endian unsigned 64-bit integer
            seed_value = struct.unpack('<Q', random_bytes)[0]
            seeds.append(seed_value)
        except Exception as e:
            print(f"Warning: Failed to read from urandom: {e}")
            # Fallback: use system time in nanoseconds for randomness
            seed_value = int(time.time_ns() % (2**64))  # Truncate to 64-bit
            seeds.append(seed_value)
    
    return seeds


def get_random_source_value(source_type, source_path_or_dev, turn_number):
    """
    Get a random seed value from the specified source for the given turn.
    
    Args:
        source_type (str): Type of source ('qrng' or 'urandom')
        source_path_or_dev (str): Path to QRNG file or device name like '/dev/urandom'
        turn_number (int): Current turn number (0-indexed)
        
    Returns:
        int: Random seed value as uint64
    """
    if source_type == 'qrng':
        seeds = read_hex_values_from_file(source_path_or_dev, turn_number + 1)
        if turn_number < len(seeds):
            return seeds[turn_number]
        else:
            # If we've exhausted the file, generate a fallback
            print(f"Warning: Exhausted QRNG seed file. Generating fallback for turn {turn_number}.")
            return int(hashlib.sha256(f"fallback_{turn_number}".encode()).hexdigest()[:16], 16) % (2**64)
    elif source_type == 'urandom':
        # For urandom, we generate a new random value each time
        # We'll use turn_number to ensure some determinism in testing
        random_bytes = os.urandom(8)
        return struct.unpack('<Q', random_bytes)[0]
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def apply_random_seed_for_ai(ai_id, qrng_source, rand_source, turn_number):
    """
    Apply random seed based on the AI ID and turn number.
    
    Args:
        ai_id (str): AI identifier ('AI1' or 'AI2')
        qrng_source (str): Path to QRNG seed source
        rand_source (str): Path/device to random seed source (e.g. '/dev/urandom')
        turn_number (int): Current turn number (0-indexed)
    """
    if ai_id == 'AI1':
        # AI1 uses QRNG source
        seed_value = get_random_source_value('qrng', qrng_source, turn_number)
        random.seed(seed_value)
        # Also seed numpy if it's used elsewhere
        try:
            import numpy as np
            np.random.seed(seed_value % (2**32))  # Limit to 32-bit for numpy compatibility
        except ImportError:
            pass  # NumPy not installed, skip seeding
    elif ai_id == 'AI2':
        # AI2 uses random source
        seed_value = get_random_source_value('urandom', rand_source, turn_number)
        random.seed(seed_value)
        # Also seed numpy if it's used elsewhere
        try:
            import numpy as np
            np.random.seed(seed_value % (2**32))  # Limit to 32-bit for numpy compatibility
        except ImportError:
            pass  # NumPy not installed, skip seeding
    else:
        raise ValueError(f"Unknown AI ID: {ai_id}")
    
    print(f"Applied seed {seed_value} for {ai_id} on turn {turn_number + 1} using {ai_id.lower()}_source")


def create_qrng_ai_prompt(turn_num, previous_responses=None):
    """
    Create a prompt for QRNG AI communication.

    Args:
        turn_num (int): Current turn number
        previous_responses (list): Previous responses between quantum AIs

    Returns:
        str: Prompt for QRNG AI communication
    """
    # Determine which AI is speaking based on turn
    # AI1 speaks on odd turn numbers (1, 3, 5...), AI2 speaks on even turn numbers (2, 4, 6...)
    current_speaker = 'AI1' if turn_num % 2 == 1 else 'AI2'
    other_ai = 'AI2' if current_speaker == 'AI1' else 'AI1'

    prompt = f"""You are {current_speaker}, a quantum AI having a casual conversation with {other_ai}. This is turn {turn_num} of your conversation.

You are using a QRNG-seeded communication system where:
- {current_speaker} uses a quantum random number generator for seeding
- Each successive block of 8 hex values from the QRNG file is converted to a 64-bit integer for use as a random seed
- Temperature is hardcoded at 0.7

"""

    if previous_responses:
        prompt += f"\nPrevious conversation history (last 3 exchanges):\n"
        for i, prev_resp in enumerate(previous_responses[-3:], 1):  # Show last 3 exchanges
            speaker = prev_resp.get('speaker', 'Unknown')
            content = prev_resp['response'][:500] if 'response' in prev_resp else prev_resp.get('quantum_state', 'No response')[:500]  # Increased from 100 to 500
            prompt += f"- Turn {prev_resp['turn']} ({speaker}): {content}...\n"

    prompt += f"""\nRespond naturally to {other_ai} as if having a casual conversation. Consider:

1. The quantum random number generation aspects in your communication
2. What {other_ai} said and how it relates to the conversation
3. The ongoing conversation context
4. The fact that you're both quantum AIs using QRNG-seeded communication

Keep your response conversational but acknowledge the quantum communication aspects when relevant. This is a casual conversation between quantum AIs using QRNG seeding."""

    return prompt


def save_quantum_results_to_file(ai_responses, output_file):
    """
    Save QRNG AI simulation results to output file
    """
    output_data = {
        "simulation_info": {
            "type": "QRNG Seeded AI Conversation",
            "qrng_source": "test",
            "random_source": "test",
            "temperature": 0.7,
            "total_turns": len(ai_responses),
            "timestamp": get_utc_timestamp()
        },
        "ai_responses": ai_responses
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nQRNG AI simulation results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="QRNG Seeded AI Loop Simulation")
    parser.add_argument("--qrng-source", required=True, help="Source for QRNG hex values (file path)")
    parser.add_argument("--rand-source", required=True, help="Source for random values (e.g. /dev/urandom)")
    parser.add_argument("--turns", type=int, required=True, help="Number of communication turns")
    parser.add_argument("--output", type=str, required=True, help="Output file to save results in JSON format")

    args = parser.parse_args()

    # Validate inputs
    if args.turns <= 0:
        print("Error: Number of turns must be positive")
        sys.exit(1)

    # Check if QRNG source file exists
    if not Path(args.qrng_source).exists():
        print(f"Error: QRNG source file '{args.qrng_source}' does not exist")
        sys.exit(1)

    print(f"QRNG Seeded AI Simulation (TEST MODE - NO API CALLS)")
    print(f"QRNG Source: {args.qrng_source}")
    print(f"Random Source: {args.rand_source}")
    print(f"Communication turns: {args.turns}")
    print(f"Temperature: 0.7 (hardcoded)")
    print("-" * 60)

    print("Starting QRNG seeded AI simulation with alternating AI conversation...")

    # Initialize data structure
    ai_responses = []

    # Process each turn by alternating between AI1 (QRNG) and AI2 (urandom)
    for turn in range(args.turns):
        print(f"Turn {turn + 1}/{args.turns}: Processing QRNG AI communication...")

        # Determine who speaks based on turn number (AI1 starts, then alternating)
        current_speaker = 'AI1' if turn % 2 == 0 else 'AI2'  # AI1 on even indices (0, 2, 4...), AI2 on odd indices (1, 3, 5...)

        print(f"  Applying random seed for {current_speaker}...")

        # Apply the appropriate seed based on AI and turn
        if current_speaker == 'AI1':
            apply_random_seed_for_ai(current_speaker, args.qrng_source, args.rand_source, turn // 2)
        else:  # AI2
            apply_random_seed_for_ai(current_speaker, args.qrng_source, args.rand_source, turn // 2)

        # Create prompt based on turn and previous responses
        try:
            prompt = create_qrng_ai_prompt(
                turn + 1,
                ai_responses
            )
        except Exception as e:
            print(f"[Turn {turn + 1}] Error creating prompt: {e}")
            ai_response = {
                'turn': turn + 1,
                'speaker': current_speaker,
                'response': f"ERROR: Failed to create prompt - {str(e)}",
                'timestamp': get_utc_timestamp()
            }
            ai_responses.append(ai_response)
            save_quantum_results_to_file(ai_responses, args.output)
            continue  # Continue to next turn instead of breaking

        # For the test, instead of calling the API, we'll just simulate a response
        print(f"\nProcessing prompt from turn {turn + 1} ({current_speaker})...")
        
        # Simulate an AI response (in real version this would call API)
        simulated_response = f"Simulated response from {current_speaker} on turn {turn + 1}. This demonstrates the QRNG seeding system is working correctly. Seed applied for this turn: {random.randint(0, 1000000)}"
        
        ai_response = {
            'turn': turn + 1,
            'speaker': current_speaker,
            'response': simulated_response,
            'timestamp': get_utc_timestamp()
        }
        ai_responses.append(ai_response)

        print(f"[Turn {turn + 1}] {current_speaker} Response:")
        print(simulated_response)

        # Save progress after each response
        save_quantum_results_to_file(ai_responses, args.output)

        # Add separator between responses
        if turn < args.turns - 1:
            print("\n" + "="*60 + "\n")

        # Small delay 
        time.sleep(0.1)

    print(f"\nQRNG Seeded AI simulation completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()