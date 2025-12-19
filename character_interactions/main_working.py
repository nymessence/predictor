#!/usr/bin/env python3
"""
Dynamic AI Character Conversation System - Anti-Repetition Enhanced Edition
Main execution module
"""

import os
import sys
import json
import time
import argparse
import re

# Check for required imports
print("🔧 Checking imports...")
try:
    from openai import OpenAI
    print("✓ OpenAI imported successfully")
except ImportError as e:
    print(f"❌ Failed to import OpenAI: {e}")
    print("Run: pip install openai")
    sys.exit(1)

# Import local modules
from config import *
# FIX 1: Corrected spelling to match your file: charachter_loader.py
from character_loader import load_character_generic
from response_generator import generate_response_adaptive
# FIX 2: Corrected spelling to match your file: repitition_detector.py
from repitition_detector import detect_repetition_patterns
from scenario_adapter import adapt_character_message
from scenario_progression import ScenarioProgressor, check_scenario_progression


def validate_api_key(args):
    """Validate API key is set"""
    # Check command line API key first, then check for OpenRouter, then use default
    if args.api_key:
        api_key = args.api_key
    elif args.api_endpoint and "openrouter" in args.api_endpoint.lower():
        # If using OpenRouter endpoint, try to get OpenRouter API key
        api_key = args.api_key if args.api_key else os.environ.get("OPENROUTER_API_KEY", API_KEY)
        if not api_key:
            print("❌ ERROR: OPENROUTER_API_KEY environment variable not set for OpenRouter endpoint")
            print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
            print("Or use --api-key parameter")
            sys.exit(1)
    else:
        # Use default API key for non-OpenRouter endpoints
        api_key = args.api_key if args.api_key else API_KEY
        if not api_key:
            print("❌ ERROR: AICHAT_API_KEY environment variable not set")
            print("Set it with: export AICHAT_API_KEY='your-key-here'")
            print("Or use --api-key parameter")
            sys.exit(1)
    return api_key


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Dynamic AI Character Conversation System - Anti-Repetition Enhanced'
    )
    parser.add_argument('characters', nargs='+', 
                        help='Character JSON files (minimum 2, supports more for multi-character scenarios)')
    parser.add_argument('-o', '--output', help='Output JSON file')
    parser.add_argument('-t', '--max-turns', type=int, default=MAX_TURNS,
                        help=f'Maximum conversation turns (default: {MAX_TURNS})')
    parser.add_argument('-d', '--delay', type=int, default=DELAY_SECONDS,
                        help=f'Delay between turns in seconds (default: {DELAY_SECONDS})')
    parser.add_argument('-s', '--similarity', type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                        help='Similarity threshold for repetition detection (0.0-1.0)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-environmental', action='store_true',
                        help='Disable environmental triggers')
    parser.add_argument('--emergency-threshold', type=float, default=EMERGENCY_REPETITION_THRESHOLD,
                        help='Threshold for emergency protocols (0.0-1.0)')
    parser.add_argument('--critical-threshold', type=float, default=CRITICAL_REPETITION_THRESHOLD,
                        help='Threshold for critical emergency protocols (0.0-1.0)')
    # New API configuration arguments
    parser.add_argument('--api-endpoint', default=BASE_URL,
                        help=f'API endpoint URL (default: {BASE_URL})')
    parser.add_argument('--model', default=MODEL_NAME,
                        help=f'Model name (default: {MODEL_NAME})')
    parser.add_argument('--api-key', help='API key (overrides environment variable)')
    # Custom scenario argument
    parser.add_argument('--scenario', type=str, default=None,
                        help='Custom scenario description to guide the conversation and trigger relevant lorebook entries')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a saved conversation file')
    parser.add_argument('--start-turn', type=int, default=1,
                        help='Starting turn number when resuming (default: 1)')
    return parser.parse_args()


def validate_character_files(args):
    """Validate character files exist"""
    for char_file in args.characters:
        if not os.path.exists(char_file):
            print(f"❌ ERROR: File not found: {char_file}")
            print("\nLooking for available character files in current directory...")
            available_files = [f for f in os.listdir('.') 
                              if f.endswith('.json') and not f.startswith('.')]
            if available_files:
                print("Available files:")
                for f in available_files[:10]:
                    print(f"  - {f}")
                if len(available_files) > 10:
                    print(f"  ... and {len(available_files)-10} more files")
            sys.exit(1)
    print(f"✓ Files validated: {len(args.characters)} characters loaded\n")


def load_conversation_from_file(resume_file: str) -> list:
    """Load conversation history from a JSON file"""
    try:
        with open(resume_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert old format to new format if necessary
        history = []
        for item in data:
            if isinstance(item, dict):
                if 'name' in item and 'content' in item:
                    history.append({
                        'name': item['name'],
                        'content': item['content']
                    })
                elif 'character' in item and 'message' in item:
                    history.append({
                        'name': item['character'],
                        'content': item['message']
                    })
        return history
    except Exception as e:
        print(f"❌ Error loading conversation from {resume_file}: {e}")
        return []


def setup_output_file(character_names: list, output_arg: str = None) -> str:
    """Setup output filename"""
    if output_arg:
        output_file = output_arg
    else:
        # Create a filename with all character names
        clean_names = [re.sub(r'[<>:"/\\|?* ]', '_', name) for name in character_names]
        output_file = f"{'_&_'.join(clean_names)}_conversation.json"

    # Sanitize filename
    output_file = re.sub(r'[<>:"/\\|?* ]', '_', output_file)
    return output_file


def save_conversation(history: list, output_file: str):
    """Save conversation history to JSON file"""
    print(f"\n💾 Saving conversation...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([{
            'turn': i+1,
            'name': h['name'],
            'content': h['content']
        } for i, h in enumerate(history) if isinstance(h, dict) and h.get('name') and h.get('content')], 
        f, indent=4, ensure_ascii=False)
    print(f"✅ Conversation saved to {output_file}")
    print(f"📊 Total turns: {len(history)}")


def print_final_analysis(history: list, similarity_threshold: float):
    """Print final repetition analysis"""
    final_repetition = detect_repetition_patterns(history, similarity_threshold)
    print("\n📈 Final repetition analysis:")
    print(f"  Overall repetition score: {final_repetition.get('repetition_score', 0.0):.2f}")
    if final_repetition.get('issues'):
        print(f"  Issues detected: {', '.join(final_repetition['issues'])}")
    if final_repetition.get('blocked_patterns'):
        print(f"  Blocked patterns: {len(final_repetition['blocked_patterns'])} patterns prevented")


def main():
    """Main execution function"""
    try:
        print("🚀 Starting Enhanced Character Conversation System (Anti-Repetition Edition)...")
        print("=" * 80)
        
        # Parse arguments first
        args = parse_arguments()
        print(f"✓ Arguments parsed")
        print(f"  Characters: {args.characters}")
        print(f"  Max turns: {args.max_turns}")
        print(f"  Delay: {args.delay}s")
        print(f"  Similarity threshold: {args.similarity:.2f}")
        print(f"  API Endpoint: {args.api_endpoint}")
        print(f"  Model: {args.model}")
        if args.scenario:
            print(f"  Scenario: {args.scenario}")
        print()
        
        # Validate API key
        api_key = validate_api_key(args)
        print("✓ API key validated\n")

        # Update global config
        import config
        config.DELAY_SECONDS = args.delay
        config.EMERGENCY_REPETITION_THRESHOLD = args.emergency_threshold
        config.CRITICAL_REPETITION_THRESHOLD = args.critical_threshold
        config.BASE_URL = args.api_endpoint
        config.MODEL_NAME = args.model
        config.API_KEY = api_key

        # If using OpenRouter, also update the OPENROUTER_API_KEY in config if available
        if "openrouter" in args.api_endpoint.lower():
            if not hasattr(config, 'OPENROUTER_API_KEY') or config.OPENROUTER_API_KEY != api_key:
                config.OPENROUTER_API_KEY = api_key
        
        # Validate files
        validate_character_files(args)

        # Load characters
        print("📖 Loading characters (adapting to ANY format)...")
        characters = []
        for char_file in args.characters:
            char = load_character_generic(char_file)
            characters.append(char)
            print()

        # Validate minimum characters
        if len(characters) < 2:
            print("❌ ERROR: At least 2 characters are required")
            sys.exit(1)

        # Print conversation setup
        print("=" * 80)
        print(f"📋 CONVERSATION SETUP")
        for i, char in enumerate(characters):
            print(f"  Character {chr(65+i)}: {char['name']}")
            print(f"    Private Agenda: {char['private_agenda']}")
            print(f"    Voice: {char['voice_analysis']['formality']}, {char['voice_analysis']['style']}")
            if i < len(characters) - 1:
                print()

        if args.scenario:
            print(f"\n  Custom Scenario: {args.scenario}")
        print("=" * 80 + "\n")

        # Setup output
        character_names = [char['name'] for char in characters]
        output_file = setup_output_file(character_names, args.output)
        print(f"💾 Output: {output_file}\n")

        # Initialize conversation
        if args.resume and os.path.exists(args.resume):
            print(f"🔄 Loading conversation from {args.resume}...")
            history = load_conversation_from_file(args.resume)
            turn_offset = len(history) if history else 0
            print(f"✅ Loaded {turn_offset} turns from saved conversation")

            # Adjust start turn if specified
            start_turn = max(args.start_turn, turn_offset + 1)
            print(f"📊 Starting from turn {start_turn}")
        else:
            # Initialize conversation normally
            adapted_greeting = characters[0]['greeting']
            # Adapt the first character's greeting if a custom scenario is provided
            if args.scenario:
                print(f"🔄 Adapting {characters[0]['name']}'s greeting to scenario...")
                adapted_greeting = adapt_character_message(characters[0], args.scenario, 1)
                print(f"✅ Adapted greeting generated")

            history = [{'name': characters[0]['name'], 'content': adapted_greeting}]
            print(f"[TURN 1] {characters[0]['name']}:")
            print(f"{adapted_greeting}\n")
            start_turn = 2  # Start from turn 2 since turn 1 is already loaded

        # Initialize scenario progressor
        scenario_progressor = None
        if args.scenario:
            scenario_progressor = ScenarioProgressor(args.scenario)
            print(f"🚀 Scenario progression initialized: {scenario_progressor.get_current_stage_description()}")

        current_char_index = 1 if len(history) == 0 else len(history) % len(characters)
        other_char_index = 0 if len(history) == 0 else (len(history) - 1) % len(characters)

        # Main conversation loop
        for turn in range(start_turn, args.max_turns + 1):
            print(f"\n{'='*80}")
            print(f"[TURN {turn}] {characters[current_char_index]['name']}")
            print('='*80)
            
            try:
                # Get current and other characters
                current_char = characters[current_char_index]
                other_char = characters[other_char_index]

                # Determine current scenario context (default to args.scenario or enhanced if progression exists)
                current_scenario = args.scenario
                if scenario_progressor:
                    # Always use the latest scenario context from progressor if available
                    current_scenario = scenario_progressor.get_scenario_context_for_stage()

                # Check for scenario progression
                scenario_progression_message = ""
                if scenario_progressor:
                    scenario_progression_message = check_scenario_progression(scenario_progressor, history, turn)
                    if scenario_progression_message:
                        print(f"🚀 SCENARIO PROGRESSION: {scenario_progression_message}")
                        # Add scenario progression to history as a narrative element
                        history.append({'name': 'Narrator', 'content': scenario_progression_message})
                        # Update the scenario context for this turn and for future turns
                        current_scenario = scenario_progressor.get_scenario_context_for_stage()
                        args.scenario = current_scenario  # Update the main scenario for future use
                        print(f"📊 Updated scenario context: {scenario_progressor.get_current_stage_description()}")

                # Extract lorebook entries based on scenario keywords
                lorebook_entries = []
                if current_scenario:  # Use current_scenario which may be updated by progression
                    from character_loader import extract_lorebook_entries
                    # Search for keywords in scenario that match character lorebooks
                    scenario_keywords = current_scenario.lower().split()
                    for char in characters:
                        entries = extract_lorebook_entries(char['raw_data'], history, max_entries=2)
                        # Filter entries by scenario relevance
                        relevant_entries = []
                        for entry in entries:
                            if any(keyword in entry.lower() for keyword in scenario_keywords if len(keyword) > 3):
                                relevant_entries.append(entry)
                        lorebook_entries.extend(relevant_entries[:1])  # Add at most 1 relevant entry per character

                resp = generate_response_adaptive(
                    current_char, other_char, history, turn,
                    enable_environmental=not args.no_environmental,
                    similarity_threshold=args.similarity,
                    verbose=args.verbose,
                    scenario_context=current_scenario,
                    lorebook_entries=lorebook_entries if lorebook_entries else None
                )
                
                # Validate response
                if not isinstance(resp, str):
                    resp = str(resp)
                if len(resp.strip()) < 10:
                    from response_generator import generate_emergency_response
                    resp = generate_emergency_response(current_char, other_char, history, {}, turn)
                
                history.append({'name': current_char['name'], 'content': resp})
                print(resp)
                
                # Switch characters (cycle through all characters)
                other_char_index = current_char_index
                current_char_index = (current_char_index + 1) % len(characters)
                
                # Delay before next turn
                if turn < args.max_turns:
                    print(f"\n⏳ Waiting {args.delay} seconds...")
                    time.sleep(args.delay)
                    
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user. Saving...")
                break
                
            except Exception as e:
                print(f"\n❌ Error on turn {turn}: {e}")
                import traceback
                traceback.print_exc()
                
                # Generate fallback response
                from response_generator import generate_emergency_response
                fallback_resp = generate_emergency_response(current_char, other_char, history, {}, turn)
                history.append({'name': current_char['name'], 'content': fallback_resp})
                print(fallback_resp)
                
                # Switch characters (cycle through all characters)
                other_char_index = current_char_index
                current_char_index = (current_char_index + 1) % len(characters)
                continue
        
        # Save and analyze
        save_conversation(history, output_file)
        print_final_analysis(history, args.similarity)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
