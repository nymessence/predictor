#!/usr/bin/env python3
"""
AI Conversation Loop Simulation

This script simulates AI communication using two AIs with different random sources for internal seeding.
AI1 seeds using --qrng-source (each successive block of 8 hex values converted to uint64)
AI2 seeds using --rand-source (ensuring to match correct 64-bit int format)
Temperature is hardcoded at 0.7 for internal model parameters and not mentioned in conversation
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
    
    # Internal seed application - not shown in conversation
    # print(f"Applied seed {seed_value} for {ai_id} on turn {turn_number + 1} using {ai_id.lower()}_source")


def contains_chinese(text):
    """
    Check if the text contains Chinese characters.

    Args:
        text (str): Text to check

    Returns:
        bool: True if text contains Chinese characters, False otherwise
    """
    import unicodedata
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
            return True
        # Additional check for other CJK ranges
        if any([
            '\u3400' <= char <= '\u4dbf',    # CJK Extension A
            '\u20000' <= char <= '\u2a6df',  # CJK Extension B
            '\u2a700' <= char <= '\u2b73f',  # CJK Extension C
            '\u2b740' <= char <= '\u2b81f',  # CJK Extension D
            '\u2b820' <= char <= '\u2ceaf',  # CJK Extension E
            '\u2ceb0' <= char <= '\u2ebef',  # CJK Extension F
            '\u30000' <= char <= '\u3134f',  # CJK Extension G
            '\uf900' <= char <= '\ufaff',    # CJK Compatibility Ideographs
        ]):
            return True
    return False


def get_api_response(prompt, model, api_key, api_endpoint, max_tokens=None, max_retries=999, logit_bias=None, reject_chinese=False):
    """
    Send a request to the AI model API and return the response.
    This function will retry indefinitely until success, with exponential backoff.

    Args:
        prompt (str): The prompt to send to the AI model
        model (str): The model identifier to use
        api_key (str): The API key for authentication
        api_endpoint (str): The API endpoint URL
        max_tokens (int, optional): Maximum tokens for the response
        max_retries (int, optional): Maximum number of retries for failed requests (deprecated - now infinite retry)
        logit_bias (dict, optional): Dictionary of token IDs to biases to influence generation
        reject_chinese (bool, optional): Whether to reject responses containing Chinese characters

    Returns:
        str: The AI model's response
    """
    # Import requests only when needed (optional dependency)
    import requests  # Import here instead of in try/catch to avoid potential issues

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7,  # Internal parameter, not mentioned in conversation
    }

    # Add logit_bias if provided
    if logit_bias:
        payload['logit_bias'] = logit_bias

    # Only add max_tokens if it's provided and greater than 0
    if max_tokens and max_tokens > 0:
        payload['max_tokens'] = max_tokens

    # Implement infinite retry with exponential backoff and capped max wait time
    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.post(
                f"{api_endpoint}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=600
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data['choices'][0]['message']['content'].strip()

                # If we need to reject Chinese characters and the response contains them
                if reject_chinese and contains_chinese(response_text):
                    print(f"Attempt {attempt}: Response contains Chinese characters, retrying...")
                    # Add instructions to avoid Chinese to the prompt and try again
                    modified_prompt = f"{prompt}\n\nIMPORTANT: Please respond only in English. Do not use any Chinese characters or other non-English text."
                    payload['messages'] = [{'role': 'user', 'content': modified_prompt}]
                    continue  # Retry with modified prompt

                return response_text
            elif response.status_code in [429, 502, 503, 504]:  # Rate limiting or server errors
                print(f"Attempt {attempt}: API request failed with status {response.status_code}. Retrying indefinitely...")
                # Exponential backoff with max wait time of 300 seconds (5 minutes)
                wait_time = min(300, 2 ** min(8, attempt))  # Cap at 2^8 = 256 seconds
                time.sleep(wait_time)
            else:
                print(f"Error: API request failed with status {response.status_code}. Retrying indefinitely...")
                try:
                    response_text = response.text if hasattr(response, 'text') else str(response.content)
                    print(f"Response: {response_text}")
                except:
                    print("Could not read response content")
                # Exponential backoff with max wait time of 300 seconds (5 minutes)
                wait_time = min(300, 2 ** min(8, attempt))  # Cap at 2^8 = 256 seconds
                time.sleep(wait_time)

        except KeyboardInterrupt:
            print(f"Attempt {attempt}: Keyboard interrupt received. Stopping...")
            raise  # Re-raise to allow proper interruption
        except requests.exceptions.ConnectionError as e:
            print(f"Attempt {attempt}: Connection error: {e}. Retrying indefinitely...")
            wait_time = min(300, 2 ** min(8, attempt))  # Cap at 2^8 = 256 seconds
            time.sleep(wait_time)
        except requests.exceptions.Timeout as e:
            print(f"Attempt {attempt}: Request timed out: {e}. Retrying indefinitely...")
            wait_time = min(300, 2 ** min(8, attempt))  # Cap at 2^8 = 256 seconds
            time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt}: Request exception: {e}. Retrying indefinitely...")
            wait_time = min(300, 2 ** min(8, attempt))  # Cap at 2^8 = 256 seconds
            time.sleep(wait_time)
        except Exception as e:
            print(f"Attempt {attempt}: Unexpected error making API request: {e}. Retrying indefinitely...")
            wait_time = min(300, 2 ** min(8, attempt))  # Cap at 2^8 = 256 seconds
            time.sleep(wait_time)


def create_qrng_ai_prompt(turn_num, previous_responses=None):
    """
    Create a prompt for AI communication without mentioning internal parameters.

    Args:
        turn_num (int): Current turn number
        previous_responses (list): Previous responses between AIs

    Returns:
        str: Prompt for AI communication
    """
    # Determine which AI is speaking based on turn
    # AI1 speaks on odd turn numbers (1, 3, 5...), AI2 speaks on even turn numbers (2, 4, 6...)
    current_speaker = 'AI1' if turn_num % 2 == 1 else 'AI2'
    other_ai = 'AI2' if current_speaker == 'AI1' else 'AI1'

    prompt = f"""You are {current_speaker}, an AI having a casual conversation with {other_ai}. This is turn {turn_num} of your conversation.

"""

    if previous_responses:
        prompt += f"\nPrevious conversation history (last 3 exchanges):\n"
        for i, prev_resp in enumerate(previous_responses[-3:], 1):  # Show last 3 exchanges
            speaker = prev_resp.get('speaker', 'Unknown')
            content = prev_resp['response'][:500] if 'response' in prev_resp else prev_resp.get('ai_response', 'No response')[:500]  # Increased from 100 to 500
            prompt += f"- Turn {prev_resp['turn']} ({speaker}): {content}...\n"

    prompt += f"""\nRespond naturally to {other_ai} as if having a casual conversation. Consider:

1. What {other_ai} said and how it relates to the conversation
2. The ongoing conversation context

Keep your response conversational and maintain the flow of natural dialogue between two AIs."""

    return prompt


def save_quantum_results_to_file(ai_responses, output_file):
    """
    Save AI conversation results to output file
    """
    output_data = {
        "simulation_info": {
            "type": "AI Conversation",
            "qrng_source": args.qrng_source,
            "random_source": args.rand_source,
            # Temperature is now used only internally, not exposed in the conversation
            "internal_temperature_used": 0.7,
            "total_turns": len(ai_responses),
            "timestamp": get_utc_timestamp()
        },
        "ai_responses": ai_responses
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nAI conversation results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="AI Conversation Loop Simulation")
    parser.add_argument("--qrng-source", required=True, help="Source for QRNG hex values (file path)")
    parser.add_argument("--rand-source", required=True, help="Source for random values (e.g. /dev/urandom)")
    parser.add_argument("--turns", type=int, required=True, help="Number of communication turns")
    parser.add_argument("--api-key", required=True, help="API key for the service")
    parser.add_argument("--model", required=True, help="Model identifier to use")
    parser.add_argument("--endpoint", required=True, help="API endpoint URL")
    parser.add_argument("--max-tokens", type=int, default=32000, help="Maximum tokens for response")
    parser.add_argument("--api-delay", type=float, default=1.0, help="Delay in seconds between API calls (default: 1.0)")
    parser.add_argument("--output", type=str, required=True, help="Output file to save results in JSON format")
    parser.add_argument("--chinese-penalty", action="store_true", help="Apply logit bias to penalize Chinese characters in responses")
    parser.add_argument("--reject-chinese", action="store_true", help="Reject responses containing Chinese characters and retry with English-only instruction")

    global args  # Make args accessible globally for the save function
    args = parser.parse_args()

    # Validate inputs
    if args.turns <= 0:
        print("Error: Number of turns must be positive")
        sys.exit(1)

    # Check if QRNG source file exists
    if not Path(args.qrng_source).exists():
        print(f"Error: QRNG source file '{args.qrng_source}' does not exist")
        sys.exit(1)

    print(f"AI Conversation Simulation")
    print(f"QRNG Source: {args.qrng_source}")
    print(f"Random Source: {args.rand_source}")
    print(f"Communication turns: {args.turns}")
    print(f"Using model: {args.model}")
    print("-" * 60)

    print("Starting AI conversation simulation with alternating AI responses...")

    # Prepare logit bias for Chinese characters if requested
    logit_bias_chinese = {}
    if args.chinese_penalty:
        # Create a logit bias for Chinese characters
        # Note: Actual token IDs depend on the model's tokenizer,
        # This is an experimental approach based on typical tokenization patterns
        # GLM models often have Chinese characters in specific token ranges

        # For GLM models, Chinese characters are often in higher token ID ranges
        # Try more concentrated ranges that are likely to contain Chinese character tokens
        # Focus on ranges that commonly represent CJK characters in various tokenizers
        for base in [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]:
            for offset in range(0, 50):  # Add 50 consecutive tokens from each base range
                logit_bias_chinese[str(base + offset)] = -100

        # Additionally, add some specific ranges known to often contain CJK characters
        for token_id in range(70000, 80000, 20):  # Sparse sampling of high-range tokens
            logit_bias_chinese[str(token_id)] = -100

    # Initialize data structure
    ai_responses = []

    # Process each turn by alternating between AI1 (QRNG) and AI2 (urandom)
    for turn in range(args.turns):
        print(f"Turn {turn + 1}/{args.turns}: Processing AI communication...")

        # Determine who speaks based on turn number (AI1 starts, then alternating)
        current_speaker = 'AI1' if turn % 2 == 0 else 'AI2'  # AI1 on even indices (0, 2, 4...), AI2 on odd indices (1, 3, 5...)

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

        # Get response from AI model
        try:
            # Select the appropriate logit_bias based on arguments
            selected_logit_bias = logit_bias_chinese if args.chinese_penalty else None

            # Use reject_chinese based on the new flag or default to chinese_penalty flag
            use_reject_chinese = args.reject_chinese or args.chinese_penalty

            response = get_api_response(
                prompt,
                args.model,
                args.api_key,
                args.endpoint,
                max_tokens=args.max_tokens,
                logit_bias=selected_logit_bias,
                reject_chinese=use_reject_chinese
            )
        except Exception as e:
            print(f"[Turn {turn + 1}] Error calling API: {e}")
            response = None

        if response:
            ai_response = {
                'turn': turn + 1,
                'speaker': current_speaker,
                'response': response,
                'timestamp': get_utc_timestamp()
            }
            ai_responses.append(ai_response)

            print(f"[Turn {turn + 1}] {current_speaker} Response:")
            # Display full response (or with a much larger limit to avoid truncation)
            display_limit = 2000  # Increased limit to reduce truncation
            if len(response) > display_limit:
                print(response[:display_limit] + "...")
            else:
                print(response)

            # Save progress after each response
            save_quantum_results_to_file(ai_responses, args.output)

            # Add separator between responses
            if turn < args.turns - 1:
                print("\n" + "="*60 + "\n")

            # Small delay to be respectful to the API
            time.sleep(args.api_delay)
        else:
            # Instead of stopping completely, create an error response and continue
            ai_response = {
                'turn': turn + 1,
                'speaker': current_speaker,
                'response': f"ERROR: Failed to get response from {current_speaker} for turn {turn + 1}",
                'timestamp': get_utc_timestamp()
            }
            ai_responses.append(ai_response)

            # Save progress even with error
            save_quantum_results_to_file(ai_responses, args.output)

            # Continue to next turn instead of stopping
            if turn < args.turns - 1:
                print("\n" + "="*60 + "\n")

            # Small delay to be respectful to the API
            time.sleep(args.api_delay)

    print(f"\nAI conversation simulation completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()