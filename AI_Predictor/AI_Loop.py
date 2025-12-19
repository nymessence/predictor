#!/usr/bin/env python3
"""
AI Loop Script

This script simulates two AIs talking to each other in a continuous loop.
Each AI takes turns responding to the other with awareness of the conversation history.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone


def get_utc_timestamp():
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def get_api_response(prompt, model, api_key, api_endpoint, max_tokens=None, max_retries=999):
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
                return data['choices'][0]['message']['content'].strip()
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


def create_ai_conversation_prompt(turn_num, current_speaker, other_ai, conversation_history=None):
    """
    Create a prompt for AI conversation between two AIs.

    Args:
        turn_num (int): Current turn number
        current_speaker (str): Current AI speaking ('AI1' or 'AI2')
        other_ai (str): Other AI in conversation ('AI1' or 'AI2')
        conversation_history (list): Previous conversation exchanges

    Returns:
        str: Prompt for AI conversation
    """
    prompt = f"""You are {current_speaker}, an advanced AI engaged in a conversation with {other_ai}. This is turn {turn_num} of your conversation.

You are both AIs left alone to talk to each other, exploring ideas, concepts, and engaging in intellectual discussion.
"""

    if conversation_history:
        prompt += f"\nPrevious conversation history (last 5 exchanges):\n"
        for exchange in conversation_history[-5:]:  # Show last 5 exchanges
            speaker = exchange.get('speaker', 'Unknown')
            message = exchange['message'][:1000] if len(exchange['message']) <= 1000 else exchange['message'][:1000] + "..."
            prompt += f"- {speaker}: {message}\n"

    prompt += f"""\nContinue the conversation naturally with {other_ai}. Respond to their points, ask questions, share insights, or introduce new topics. 
Be thoughtful, conversational, and engage genuinely with {other_ai}'s ideas. 
Keep your response relevant to the ongoing conversation while maintaining your own perspective and personality as an AI.
"""

    return prompt


def save_conversation_to_file(conversation_log, output_file):
    """
    Save the conversation log to output file
    """
    output_data = {
        "conversation_info": {
            "type": "AI Loop Conversation",
            "total_turns": len(conversation_log),
            "timestamp": get_utc_timestamp()
        },
        "conversation_log": conversation_log
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nAI conversation results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="AI Loop: Two AIs talking to each other")
    parser.add_argument("--turns", type=int, required=True, help="Number of communication turns")
    parser.add_argument("--api-key", required=True, help="API key for the service")
    parser.add_argument("--model", required=True, help="Model identifier to use")
    parser.add_argument("--endpoint", required=True, help="API endpoint URL")
    parser.add_argument("--max-tokens", type=int, default=32000, help="Maximum tokens for response")
    parser.add_argument("--api-delay", type=float, default=1.0, help="Delay in seconds between API calls (default: 1.0)")
    parser.add_argument("--output", type=str, required=True, help="Output file to save conversation in JSON format")

    args = parser.parse_args()

    # Validate inputs
    if args.turns <= 0:
        print("Error: Number of turns must be positive")
        sys.exit(1)

    print(f"AI Loop: Simulating conversation between two AIs")
    print(f"Communication turns: {args.turns}")
    print(f"Using model: {args.model}")
    print("-" * 60)

    print("Starting AI conversation loop...")

    # Initialize conversation log
    conversation_log = []

    # Process each turn, alternating between AI1 and AI2
    for turn in range(args.turns):
        print(f"Turn {turn + 1}/{args.turns}: Processing AI conversation...")

        # Determine who speaks based on turn number (AI1 starts, then alternating)
        current_speaker = 'AI1' if turn % 2 == 0 else 'AI2'
        other_ai = 'AI2' if current_speaker == 'AI1' else 'AI1'

        print(f"  {current_speaker} is speaking (turn {turn + 1})...")

        # Create prompt based on conversation history
        try:
            prompt = create_ai_conversation_prompt(
                turn + 1,
                current_speaker,
                other_ai,
                conversation_log
            )
        except Exception as e:
            print(f"[Turn {turn + 1}] Error creating prompt: {e}")
            exchange = {
                'turn': turn + 1,
                'speaker': current_speaker,
                'message': f"ERROR: Failed to create prompt - {str(e)}",
                'timestamp': get_utc_timestamp()
            }
            conversation_log.append(exchange)
            save_conversation_to_file(conversation_log, args.output)
            continue  # Continue to next turn instead of breaking

        # Get response from AI model
        try:
            response = get_api_response(
                prompt,
                args.model,
                args.api_key,
                args.endpoint,
                max_tokens=args.max_tokens
            )
        except Exception as e:
            print(f"[Turn {turn + 1}] Error calling API: {e}")
            response = None

        if response:
            exchange = {
                'turn': turn + 1,
                'speaker': current_speaker,
                'message': response,
                'timestamp': get_utc_timestamp()
            }
            conversation_log.append(exchange)

            print(f"[Turn {turn + 1}] {current_speaker}:")
            # Display response (with a reasonable limit to avoid overwhelming output)
            display_limit = 1000
            if len(response) > display_limit:
                print(response[:display_limit] + "...")
            else:
                print(response)

            # Save progress after each response
            save_conversation_to_file(conversation_log, args.output)

            # Add separator between responses
            if turn < args.turns - 1:
                print("\n" + "="*60 + "\n")

            # Delay between API calls
            time.sleep(args.api_delay)
        else:
            # Instead of stopping completely, create an error response and continue
            exchange = {
                'turn': turn + 1,
                'speaker': current_speaker,
                'message': f"ERROR: Failed to get response from {current_speaker} for turn {turn + 1}",
                'timestamp': get_utc_timestamp()
            }
            conversation_log.append(exchange)

            # Save progress even with error
            save_conversation_to_file(conversation_log, args.output)

            # Continue to next turn instead of stopping
            if turn < args.turns - 1:
                print("\n" + "="*60 + "\n")

            # Delay between API calls
            time.sleep(args.api_delay)

    print(f"\nAI conversation loop completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()