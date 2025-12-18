#!/usr/bin/env python3
"""
Year-by-Year AI Predictor

This script takes a prediction topic and cycles through a range of years,
generating predictions for each year using an AI model. It uses OpenRouter
to connect to AI models like DeepSeek or other open models.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime


def get_api_response(prompt, model, api_key, api_endpoint, max_tokens=None, max_retries=999, logit_bias=None):
    """
    Send a request to the AI model API and return the response.

    Args:
        prompt (str): The prompt to send to the AI model
        model (str): The model identifier to use
        api_key (str): The API key for authentication
        api_endpoint (str): The API endpoint URL
        max_tokens (int, optional): Maximum tokens for the response
        max_retries (int, optional): Maximum number of retries for failed requests
        logit_bias (dict, optional): Dictionary of token IDs to biases to influence generation

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

    # Add logit_bias if provided
    if logit_bias:
        payload['logit_bias'] = logit_bias

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


def create_prediction_prompt(topic, current_year, previous_predictions=None, token_limit=32000):
    """
    Create a prompt for the AI model for a specific year.

    Args:
        topic (str): The prediction topic
        current_year (int): The year to predict for
        previous_predictions (list): Previous predictions (for context)
        token_limit (int): Maximum token limit for context

    Returns:
        str: The prompt for the AI model
    """
    prompt = f"You are an expert futurist analyzing the '{topic}' domain."
    prompt += f"\n\nThe current year being analyzed is: {current_year}"

    if previous_predictions:
        prompt += f"\n\nHistorical developments that occurred in previous years:"
        # Calculate the starting year based on how many previous predictions we have
        start_year = current_year - len(previous_predictions)

        # Convert all previous predictions to entries with year information
        all_entries = []
        for i, pred_text in enumerate(previous_predictions):
            pred_year = start_year + i
            all_entries.append({
                'year': pred_year,
                'text': pred_text,
                'length': len(pred_text)
            })

        # Process entries prioritizing most recent ones to fit within token limit
        # We'll go backwards to prioritize recent context, then reverse the selected entries to maintain chronological order
        processed_entries = []
        estimated_token_usage = len(prompt) // 4
        token_budget = int(token_limit * 0.7)  # Use 70% of limit to leave room for response

        # Go through entries in reverse chronological order (most recent first)
        for entry in reversed(all_entries):
            entry_tokens = entry['length'] // 4
            estimated_new_tokens = estimated_token_usage + entry_tokens

            if estimated_new_tokens <= token_budget:
                # Add the full entry if it fits
                processed_entries.append(entry)
                estimated_token_usage = estimated_new_tokens
            else:
                # Try to add a shortened version of this entry
                remaining_tokens = token_budget - estimated_token_usage
                if remaining_tokens > 10:  # Only add if there's meaningful space left
                    char_budget = remaining_tokens * 4
                    preview = entry['text'][:char_budget] + "..."
                    processed_entries.append({
                        'year': entry['year'],
                        'text': preview,
                        'length': len(preview)
                    })
                    estimated_token_usage = token_budget
                # Stop once we hit the budget
                break

        # Sort the selected entries back into chronological order
        processed_entries.sort(key=lambda x: x['year'])

        # Add the processed entries to the prompt in chronological order
        for entry in processed_entries:
            preview = entry['text'][:200] + "..." if len(entry['text']) > 200 else entry['text']
            prompt += f"\n- {entry['year']}: {preview}"

    prompt += f"\n\nBased on this historical context, predict the major technological, social, economic, or scientific developments that will occur in {current_year} related to '{topic}'."
    prompt += f"\nFocus on specific, concrete developments rather than vague trends. Include potential breakthroughs, major milestones, regulatory changes, or market shifts that could be documented as having happened in {current_year}."

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Year-by-Year AI Predictor")
    parser.add_argument("--model", required=True, help="Model identifier to use")
    parser.add_argument("--api-key", required=True, help="API key for the service")
    parser.add_argument("--api-endpoint", required=True, help="API endpoint URL")
    parser.add_argument("--years", type=int, required=True, help="Number of years to predict")
    parser.add_argument("--predict", required=True, help="Topic to predict")
    parser.add_argument("--start-year", type=int, default=datetime.now().year + 1,
                        help="Starting year for predictions (default: next year)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without making API calls")
    parser.add_argument("--token-limit", type=int, default=32000,
                        help="Token limit for context (default: 32000)")
    parser.add_argument("--output", type=str,
                        help="Output file to save predictions in JSON format")
    parser.add_argument("--chinese-penalty", action="store_true",
                        help="Apply logit bias to penalize Chinese characters in responses")

    args = parser.parse_args()

    # Validate inputs
    if args.years <= 0:
        print("Error: Number of years must be positive")
        sys.exit(1)

    # Check if output file exists and load existing predictions for resume functionality
    existing_predictions = {}
    start_offset = 0

    if args.output and os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            # Check if the existing file matches our current parameters
            if (existing_data.get("topic") == args.predict and
                existing_data.get("start_year") == args.start_year and
                existing_data.get("end_year") == (args.start_year + args.years - 1)):

                existing_predictions = existing_data.get("predictions", {})
                print(f"Resuming from existing file: {args.output}")
                print(f"Found {len(existing_predictions)} existing predictions")

                # Determine start offset based on the number of years already completed
                completed_years = len([year for year in existing_predictions.keys()
                                     if args.start_year <= int(year) <= args.start_year + args.years - 1])
                start_offset = completed_years

                # Get the prediction values in chronological order
                predictions = []
                for year in range(args.start_year, args.start_year + start_offset):
                    year_str = str(year)
                    if year_str in existing_predictions:
                        predictions.append(existing_predictions[year_str])

                print(f"Starting from year {args.start_year + start_offset} (index {start_offset})")
            else:
                print(f"Warning: Existing file {args.output} has different parameters. Starting fresh.")
                predictions = []

        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"Warning: Could not parse existing file {args.output}. Starting fresh.")
            predictions = []
    else:
        predictions = []

    print(f"AI Predictor: {args.predict}")
    print(f"From {args.start_year} to {args.start_year + args.years - 1}")
    print(f"Using model: {args.model}")
    print(f"Resume mode: Will start from year index {start_offset}")
    print("-" * 60)

    # Prepare logit bias for Chinese characters if requested
    logit_bias_chinese = {}
    if args.chinese_penalty:
        # Create a logit bias for Chinese characters
        # Note: Actual token IDs depend on the model's tokenizer,
        # so this is an approximation based on common patterns
        # For GLM models, Chinese characters may have specific token ranges
        # We'll use common token ranges that are likely to include Chinese characters

        # This approach tries multiple potential token ranges based on different tokenizers
        # Adding more concentrated ranges that are more likely to contain Chinese tokens
        for base in [10000, 20000, 30000, 40000, 50000, 60000, 70000]:
            for offset in range(0, 100):  # Add 100 consecutive tokens from each base range
                logit_bias_chinese[str(base + offset)] = -100

    for i in range(start_offset, args.years):
        current_year = args.start_year + i

        # Create prompt for current year
        # Only use previous predictions that have been completed
        current_predictions = predictions[:]  # Copy current predictions
        prompt = create_prediction_prompt(args.predict, current_year, current_predictions, args.token_limit)

        if args.dry_run:
            print(f"\n[{current_year}] DRY RUN - Would send prompt:")
            print(prompt)
            print(f"\n[{current_year}] DRY RUN - Would receive prediction")

            # Simulate adding a prediction for dry-run so later years have context
            # Note: In dry run, we don't actually call the API but we would use the same max_tokens calculation
            simulated_prediction = f"[Simulated prediction for {current_year}: Major AI advancement in {args.predict.lower()}]"
            predictions.append(simulated_prediction)

            # Save progress after each simulated prediction too
            if args.output:
                save_predictions_to_file(args, predictions, existing_predictions)

            # Add separator between predictions
            if i < args.years - 1:  # Not the last iteration
                print("\n" + "="*60 + "\n")
        else:
            print(f"\n[{current_year}] Requesting prediction...")

            # Get response from AI model
            # Calculate max_tokens for the response, leaving room for the prompt
            # Use approximately half of the token limit or a reasonable default
            response_max_tokens = max(args.token_limit // 2, 4000)  # At least 4000 tokens for response

            # Select the appropriate logit_bias based on arguments
            selected_logit_bias = logit_bias_chinese if args.chinese_penalty else None

            response = get_api_response(prompt, args.model, args.api_key, args.api_endpoint, max_tokens=response_max_tokens, logit_bias=selected_logit_bias)

            if response:
                predictions.append(response)
                print(f"[{current_year}] Prediction:")
                print(response)

                # Save progress after each successful prediction
                if args.output:
                    save_predictions_to_file(args, predictions, existing_predictions)

                # Add separator between predictions
                if i < args.years - 1:  # Not the last iteration
                    print("\n" + "="*60 + "\n")

                    # Small delay to be respectful to the API
                    time.sleep(1)
            else:
                print(f"[{current_year}] Failed to get prediction. Stopping.")
                break


def save_predictions_to_file(args, current_predictions, existing_predictions=None):
    """
    Save predictions to output file, combining with existing predictions if present
    """
    # Initialize output data
    output_data = {
        "topic": args.predict,
        "start_year": args.start_year,
        "end_year": args.start_year + args.years - 1,
        "years_count": args.years,
        "predictions": {}
    }

    # Add existing predictions first (if any)
    if existing_predictions:
        output_data["predictions"].update(existing_predictions)

    # Add current predictions
    for i in range(len(current_predictions)):
        current_year = args.start_year + i
        year_str = str(current_year)

        # Only overwrite if the new one is different from what we had
        output_data["predictions"][year_str] = current_predictions[i]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nProgress saved to {args.output}")


if __name__ == "__main__":
    main()
