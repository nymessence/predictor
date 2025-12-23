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
from chess_game_module import ChessGame
from tic_tac_toe_game import TicTacToeGame
from rock_paper_scissors_game import RockPaperScissorsGame
from hangman_game import HangmanGame
from twenty_one_game import TwentyOneGame
from number_guessing_game import NumberGuessingGame
from word_association_game import WordAssociationGame
from connect_four_game import ConnectFourGame
from uno_game import UnoGame


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

    # Game mode arguments - mutually exclusive
    game_group = parser.add_mutually_exclusive_group()
    game_group.add_argument('--chess', action='store_true',
                        help='Enable chess game mode where characters play chess and discuss moves')
    game_group.add_argument('--tic-tac-toe', action='store_true',
                        help='Enable tic-tac-toe game mode where characters play tic-tac-toe')
    game_group.add_argument('--rock-paper-scissors', action='store_true',
                        help='Enable rock-paper-scissors game mode where characters play rock-paper-scissors')
    game_group.add_argument('--hangman', action='store_true',
                        help='Enable hangman game mode where characters guess letters')
    game_group.add_argument('--twenty-one', action='store_true',
                        help='Enable twenty-one (simplified blackjack) game mode where characters try to reach 21')
    game_group.add_argument('--number-guessing', action='store_true',
                        help='Enable number guessing game mode where characters guess a secret number with high/low feedback')
    game_group.add_argument('--word-association', action='store_true',
                        help='Enable word association game mode where characters take turns saying related words')
    game_group.add_argument('--connect-four', action='store_true',
                        help='Enable connect-four game mode where characters try to connect four discs in a row')
    game_group.add_argument('--uno', action='store_true',
                        help='Enable uno card game mode where characters match colors or values')
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
    """Setup output filename preserving directory path"""
    if output_arg:
        output_file = output_arg
    else:
        # Create a filename with all character names
        clean_names = [re.sub(r'[<>:"\\|?* ]', '_', name) for name in character_names]  # Don't replace / in names
        output_file = f"{'_&_'.join(clean_names)}_conversation.json"

    # Split path and filename to handle them separately
    dir_path = os.path.dirname(output_file)
    file_name = os.path.basename(output_file)

    # Sanitize only the filename, not the directory path
    sanitized_filename = re.sub(r'[<>:"\\|?* ]', '_', file_name)

    # Reconstruct path preserving directory separators
    if dir_path:
        output_file = os.path.join(dir_path, sanitized_filename)
    else:
        output_file = sanitized_filename

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


def save_conversation_periodic(history: list, output_file: str, turn: int):
    """Save conversation periodically during the game"""
    try:
        # Create autosave filename
        dir_path = os.path.dirname(output_file)
        file_name = os.path.basename(output_file)
        name_part, ext = os.path.splitext(file_name)
        autosave_file = os.path.join(dir_path, f"{name_part}_autosave_{turn}{ext}")

        with open(autosave_file, 'w', encoding='utf-8') as f:
            json.dump([{
                'turn': i+1,
                'name': h['name'],
                'content': h['content']
            } for i, h in enumerate(history) if isinstance(h, dict) and h.get('name') and h.get('content')],
            f, indent=4, ensure_ascii=False)
        # Only print for debugging - don't clutter output
        # print(f"🔄 Autosaved to {autosave_file}")
    except Exception as e:
        print(f"⚠️  Auto-save failed: {e}")


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

        # Define helper functions for parsing game responses with JSON format
        def parse_game_json_response(response_text, character_name):
            """Parse the JSON response from the AI containing dialogue, move/action, and game state."""
            import json
            import re

            # First, try to find JSON within the response using regex
            # Look for JSON between curly braces
            json_pattern = r'\{[^{}]*\}'  # Simple non-nested JSON objects
            nested_json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'  # Allow one level of nesting
            complex_json_pattern = r'\{[\s\S]*?\}'  # Greedy capture for complex JSON

            # Try different patterns to find JSON in the response
            patterns = [complex_json_pattern, nested_json_pattern, json_pattern]
            for pattern in patterns:
                matches = re.findall(pattern, response_text, re.DOTALL)
                if matches:
                    for json_str in matches:
                        try:
                            json_clean = json_str.strip()
                            parsed = json.loads(json_clean)

                            # Extract common fields used across different games
                            dialogue = parsed.get('dialogue', parsed.get('strategy_thoughts', parsed.get('thoughts', '')))
                            move = parsed.get('move', parsed.get('action', parsed.get('letter', parsed.get('choice', parsed.get('selection', '')))))
                            board_state = parsed.get('board_state', parsed.get('state', parsed.get('game_state', parsed.get('result', ''))))

                            # Return successfully if we have at least one valid piece of data
                            if dialogue or move:
                                return dialogue.strip(), move.strip(), board_state.strip()
                        except (json.JSONDecodeError, KeyError):
                            continue  # Try the next match

            # If no JSON could be parsed, try to extract move-like patterns from plain text
            # Look for common chess patterns: e4, Nf3, exd5, O-O, etc.
            chess_pattern = r'\b([a-h][1-8]|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]|[KQ]?O-O(?:-O)?[+#]?)\b'
            chess_matches = re.findall(chess_pattern, response_text, re.IGNORECASE)

            if chess_matches and len(chess_matches) > 0:
                # Extract move and use the rest as dialogue
                move_found = chess_matches[0]

                # Remove the move from the text to use as dialogue
                dialogue_text = re.sub(chess_pattern, '', response_text, 1, re.IGNORECASE).strip()

                # Remove extra whitespace and punctuation
                dialogue_text = re.sub(r'\s+', ' ', dialogue_text)

                return dialogue_text, move_found, ""

            # Look for common patterns in other games
            # Tic-tac-toe: [0,2], (1,1), etc.
            ttt_pattern = r'[\[\(]\s*(\d+)\s*[,;\s]\s*(\d+)\s*[\]\)]'
            ttt_matches = re.findall(ttt_pattern, response_text)

            if ttt_matches and len(ttt_matches) > 0:
                row, col = ttt_matches[0]
                move_found = f"[{row}, {col}]"

                # Remove the move from the text to use as dialogue
                dialogue_text = re.sub(ttt_pattern, '', response_text, 1).strip()
                dialogue_text = re.sub(r'\s+', ' ', dialogue_text)

                return dialogue_text, move_found, ""

            # Look for rock-paper-scissors choices
            rps_pattern = r'\b(rock|paper|scissors|r|p|s)\b'
            rps_matches = re.findall(rps_pattern, response_text, re.IGNORECASE)

            if rps_matches and len(rps_matches) > 0:
                choice_found = rps_matches[0].lower()
                if choice_found in ['r']:
                    choice_found = 'rock'
                elif choice_found in ['p']:
                    choice_found = 'paper'
                elif choice_found in ['s']:
                    choice_found = 'scissors'

                # Remove the choice from the text to use as dialogue
                dialogue_text = re.sub(rps_pattern, '', response_text, 1, re.IGNORECASE).strip()
                dialogue_text = re.sub(r'\s+', ' ', dialogue_text)

                return dialogue_text, choice_found, ""

            # Look for hangman letter guesses (single letters)
            hangman_pattern = r'\b([a-zA-Z])\b'
            letter_matches = re.findall(hangman_pattern, response_text)

            # Filter to common letters mentioned in context of guessing
            words_with_letters = re.findall(r'guess[esd]?\s+(?:that\s+)?(?:it\'s\s+)?(?:the\s+)?(?:letter\s+)?["\']?([a-zA-Z])["\']?', response_text, re.IGNORECASE)
            if words_with_letters:
                letter_found = words_with_letters[0].lower()

                # Use the sentence containing the letter as dialogue
                for match in words_with_letters:
                    match_idx = response_text.find(match)
                    if match_idx != -1:
                        # Extract a portion around the match
                        start = max(0, match_idx - 30)
                        end = min(len(response_text), match_idx + 30)
                        return response_text[start:end].strip(), letter_found, ""

            # If no specific patterns found directly, try more comprehensive natural language parsing
            # Look for patterns where the AI mentions a move in natural language like:
            # "I will move my pawn to e4" or "I'll advance my pawn to e4" or "My move is e4"

            # Enhanced natural language parsing for chess moves
            # Handle more complex phrases like "move my pawn from e2 to e4"
            natural_chess_pattern = r'(?:move|advance|play|go|move\s+to|advance\s+to)\s+(?:my\s+)?(?:pawn|piece|knight|bishop|rook|queen|king)?\s*(?:from\s+([a-h][1-8])\s+)?to\s+([a-h][1-8])'
            natural_matches = re.findall(natural_chess_pattern, response_text, re.IGNORECASE)

            if natural_matches:
                for from_sq, to_sq in natural_matches:
                    if from_sq and to_sq:  # Both source and destination found
                        # Construct a proper move notation from source to destination
                        full_move = f"{from_sq}{to_sq}"

                        # Extract the sentence containing the move
                        sentences = re.split(r'[.!?]+', response_text)
                        for sentence in sentences:
                            if from_sq in sentence and to_sq in sentence:
                                # Return the sentence as dialogue and the constructed move
                                return sentence.strip(), f"{from_sq}-{to_sq}", ""

            # Alternative pattern for "from e2 to e4" format
            alt_pattern = r'from\s+([a-h][1-8])\s+to\s+([a-h][1-8])'
            alt_matches = re.findall(alt_pattern, response_text, re.IGNORECASE)

            if alt_matches:
                for from_sq, to_sq in alt_matches:
                    # Create a sentence pattern to find the sentence containing this move
                    sentence_pattern = r'[^.!?]*\bfrom\s+' + re.escape(from_sq) + r'\s+to\s+' + re.escape(to_sq) + r'\b[^.!?]*[.!?]'
                    sentence_matches = re.findall(sentence_pattern, response_text, re.IGNORECASE)
                    if sentence_matches:
                        move_str = f"{from_sq}{to_sq}"  # Chess notation e2e4
                        sentence = sentence_matches[0].strip()
                        return sentence, move_str, ""

                    # If we couldn't find a specific sentence, construct move anyway
                    move_str = f"{from_sq}{to_sq}"
                    return response_text, move_str, ""

            # Also check for moves mentioned without "to" pattern: "e4", "Nf3", etc.
            # But be very specific to avoid false positives with non-chess terms
            # Only consider valid chess notation (coordinates and piece moves)
            chess_pattern = r'\b([a-h][1-8]|[KQRBN][a-h][1-8]|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[KQRBN]?[a-h]?[1-8]?[a-h][1-8][+#]?|O-O(?:-O)?)\b'
            potential_moves = re.findall(chess_pattern, response_text, re.IGNORECASE)

            # Filter to ensure these look like actual chess moves (not just any coordinate in text)
            # Look for context terms that suggest this is a chess-related move
            chess_context_terms = ['chess', 'move', 'play', 'go', 'advance', 'position', 'strategy', 'game', 'board', 'castle', 'capture']
            has_chess_context = any(term.lower() in response_text.lower() for term in chess_context_terms)

            # Only return as a move if it's a valid chess notation AND appears in a chess context
            for potential_move in potential_moves:
                # Additional validation to make sure it's not a false positive
                # The move should be near chess-related context
                if has_chess_context:
                    # Clean the dialogue by removing the move and surrounding context
                    dialogue_text = re.sub(r'\b' + re.escape(potential_move) + r'\b', '', response_text).strip()
                    # Clean up extra punctuation and spacing
                    dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                    return dialogue_text, potential_move, ""

                # Special handling for clear chess move patterns that appear even without explicit context
                elif re.match(r'^[a-h][1-8]$', potential_move) or re.match(r'^[KQRBN][a-h][1-8]$', potential_move):
                    # For these very clear patterns, allow them if they appear in reasonable chess-related text
                    dialogue_text = re.sub(r'\b' + re.escape(potential_move) + r'\b', '', response_text).strip()
                    dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                    return dialogue_text, potential_move, ""

            # If still no move found, return the entire text as dialogue with no move
            return response_text, "", ""

        # Game-specific JSON parsing functions to avoid cross-game pattern contamination
        def parse_chess_json_response(response_text, character_name):
            """Parse JSON response specifically for chess, extracting dialogue, move, and board state.
            STRICT MODE: Only accepts properly formatted JSON with required fields."""
            import json
            import re

            # First and ONLY try: Look for properly formatted JSON with BOTH required fields
            # Find JSON that has both dialogue and move fields as required
            json_pattern = r'(\{[^{}]*("dialogue"|\'dialogue\')[^{}]*:[^{}]*["\'][^{}]*["\'][^{}]*,?[^{}]*("move"|\'move\')[^{}]*:[^{}]*["\'][^{}]*["\'][^{}]*\})'
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            # Process potential JSON matches
            for match_tuple in matches:
                # Extract the actual JSON string (first element of the tuple)
                json_str = match_tuple[0].strip()

                try:
                    # Try to parse the JSON string
                    parsed = json.loads(json_str)

                    # Extract required fields
                    dialogue = parsed.get('dialogue', '')
                    move = parsed.get('move', '').strip()
                    board_state = parsed.get('board_state', '') or parsed.get('game_state', '')

                    # Validate that both required fields are present and non-empty
                    # This is critical: both dialogue AND move must have meaningful content
                    if dialogue.strip() and move.strip():
                        return dialogue.strip(), move.strip(), board_state.strip()

                except json.JSONDecodeError:
                    # If JSON parsing fails, continue to try next match
                    continue

            # CRITICAL: If no valid JSON found with both required fields, return no move
                    if re.match(r'^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|[a-h][1-8]|O-O(?:-O)?$', potential_move, re.IGNORECASE):
                        # Remove just this specific move reference from the dialogue
                        dialogue_text = re.sub(r'\b' + re.escape(potential_move) + r'\b', '', response_text).strip()
                        dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                        return dialogue_text, potential_move, ""

            # Look for "from X to Y" patterns which are explicit move intentions
            from_to_pattern = r'(?:move\s+from|from|I\s+move\s+from)\s+([a-h][1-8])\s+(?:to|on|toward|I\s+move\s+to)\s+([a-h][1-8])'
            from_to_matches = re.findall(from_to_pattern, response_text, re.IGNORECASE)
            if from_to_matches:
                # The AI explicitly stated a from-to move, which is the most intentional
                for from_sq, to_sq in from_to_matches:
                    # Format as combined coordinate for chess move processing
                    move_value = f"{from_sq}{to_sq}"
                    # Create a targeted dialogue removal to avoid removing other coordinates
                    dialogue_text = re.sub(r'(?:move\s+from|from|I\s+move\s+from)\s+' + re.escape(from_sq) + r'\s+(?:to|on|toward|I\s+move\s+to)\s+' + re.escape(to_sq), '', response_text, flags=re.IGNORECASE).strip()
                    dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                    return dialogue_text, move_value, ""

            # Look for move patterns in contexts that indicate the AI is responding with their move NOW
            current_move_patterns = [
                r'(?:In\s+response|In\s+reply|My\s+response|My\s+next\s+move|My\s+turn|It\'?s\s+my\s+turn|On\s+my\s+turn|My\s+move|This\s+turn|This\s+round|This\s+game)\s+(?:is|will\s+be|to|should\s+be)\s+([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|[a-h][1-8]|O-O(?:-O)?)',  # "In response I will Nf3", "My move is e4", etc.
                r'(?:For\s+this\s+turn|For\s+this\s+move|For\s+now|Currently|Currently,\s+I|Right\s+now|Right\s+now,\s+I|At\s+this\s+moment)\s+([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|[a-h][1-8]|O-O(?:-O)?)',  # "For this turn, I'll e4", etc.
            ]

            # Check current move patterns
            for pattern in current_move_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                for potential_move in matches:
                    # Validate that it looks like chess notation
                    if re.match(r'^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|[a-h][1-8]|O-O(?:-O)?$', potential_move, re.IGNORECASE):
                        # Remove just this specific move reference from the dialogue
                        dialogue_text = re.sub(r'\b' + re.escape(potential_move) + r'\b', '', response_text).strip()
                        dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                        return dialogue_text, potential_move, ""

            # Finally, if no explicit intent found, look for unambiguous chess notation in clear game contexts
            # Avoid extracting chess terms from discussion contexts like "circling around" or "without saying"
            chess_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|[KQRBN]?[a-h][1-8]|[a-h][1-8]|O-O(?:-O)?)\b'
            potential_moves = re.findall(chess_pattern, response_text, re.IGNORECASE)

            # Filter potential moves to avoid discussion contexts
            # Avoid phrases like "circling around", "without saying", "important without", etc. that have no game context
            filtered_moves = []
            for potential_move in potential_moves:
                # Skip if the move appears in a discussion context that doesn't indicate intention
                discussion_contexts = [
                    rf'(?:circling\s+around\s+something|without\s+saying|without\s+saying\s+it|without|without\s+saying\s+it\s+directly|something\s+important\s+without|important\s+without|saying\s+it\s+directly|something\s+important|important|directly|directly\.\s*$)',
                    rf'(?:feel\s+like|like\s+we\'?re|like\s+)',
                    rf'(?:This\s+is\s+strange\.[^.]*){re.escape(potential_move)}',  # "This is strange. [move]" - usually not an intended move
                    rf'{re.escape(potential_move)}[^.!?]*?(?:important|strange|without|circle|around)'  # Move followed by discussion words
                ]

                is_in_discussion_context = any(
                    re.search(context_pattern, response_text, re.IGNORECASE)
                    for context_pattern in discussion_contexts
                )

                # Only include if NOT in a discussion/non-intent context
                if not is_in_discussion_context:
                    filtered_moves.append(potential_move)

            # Look for strong chess game context terms that indicate this is a move-making response
            strong_chess_terms = [
                'making a move', 'choosing', 'my move', 'I move', 'I play', 'I choose',
                'for this turn', 'this move', 'my turn to move', 'response move',
                'will move to', 'to move to', 'going to play', 'I play', 'I select',
                'my response is', 'in this position', 'on the board',
                'I will play', 'I intend to move', 'my intended move', 'on my turn'
            ]
            has_strong_chess_context = any(term.lower() in response_text.lower() for term in strong_chess_terms)

            # Only return as a move if it's in a clear game-making context and not in discussion context
            for potential_move in reversed(filtered_moves):  # Try from the end (most recent mentions first)
                if has_strong_chess_context:
                    # Remove just this specific move reference from the dialogue
                    dialogue_text = re.sub(r'\b' + re.escape(potential_move) + r'\b', '', response_text).strip()
                    dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                    return dialogue_text, potential_move, ""

            # Special handling for "from X to Y" chess notation patterns
            from_to_pattern = r'from\s+([a-h][1-8])\s+to\s+([a-h][1-8])'
            from_to_matches = re.findall(from_to_pattern, response_text, re.IGNORECASE)

            for from_sq, to_sq in from_to_matches:
                # Extract the sentence containing the move
                sentence_matches = re.findall(rf'[^.!?]*\bfrom\s+{re.escape(from_sq)}\s+to\s+{re.escape(to_sq)}\b[^.!?]*[.!?]', response_text, re.IGNORECASE)
                if sentence_matches:
                    move_val = f"{from_sq}{to_sq}"  # Convert to coordinate format like "e2e4"
                    dialogue_text = sentence_matches[0].strip()
                    return dialogue_text, move_val, ""

            # Look for moves in formats like "15. Qd3" or "Move 15: e4" which indicate current turn intent
            numbered_move_pattern = r'(?:Move|move|#)\s*(?:[0-9]+\s*[\.:]?\s*)?([KQRBN]?[a-h][1-8]|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|O-O(?:-O)?)'
            numbered_matches = re.findall(numbered_move_pattern, response_text, re.IGNORECASE)
            if numbered_matches:
                for potential_move in numbered_matches:
                    if re.match(r'^[KQRBN]?[a-h][1-8]|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|O-O(?:-O)?$', potential_move, re.IGNORECASE):
                        # Remove just this specific numbered move reference from the dialogue
                        numbered_pattern_to_remove = r'(?:Move|move|#)\s*[0-9]*\s*[\.:]?\s*' + re.escape(potential_move)
                        dialogue_text = re.sub(numbered_pattern_to_remove, '', response_text, re.IGNORECASE).strip()
                        dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                        return dialogue_text, potential_move, ""

            # Look for bold/italic formatting like **e4** or *Nf3* which often indicate moves
            bold_move_pattern = r'\*{1,2}([KQRBN]?[a-h][1-8]|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|O-O(?:-O)?)\*{1,2}'
            bold_matches = re.findall(bold_move_pattern, response_text, re.IGNORECASE)
            if bold_matches:
                for potential_move in bold_matches:
                    if re.match(r'^[KQRBN]?[a-h][1-8]|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|O-O(?:-O)?$', potential_move, re.IGNORECASE):
                        # Remove just this specific bold move reference from the dialogue
                        bold_pattern_to_remove = r'\*{1,2}' + re.escape(potential_move) + r'\*{1,2}'
                        dialogue_text = re.sub(bold_pattern_to_remove, '', response_text, re.IGNORECASE).strip()
                        dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                        return dialogue_text, potential_move, ""

            # If no chess-specific move found, return as dialogue only
            return response_text, "", ""

        def parse_move_notation(move_notation, chess_game, current_player_color):
            """Parse algebraic move notation and return the from/to positions."""
            import re

            if not move_notation:
                return False, None, None

            move_notation = move_notation.strip()

            # If move notation is empty, return failure
            if not move_notation:
                return False, None, None

            # Handle castling notation first
            if move_notation.lower() in ['o-o', '0-0']:  # Kingside castling
                # For white: king from (7,4) to (7,6), rook from (7,7) to (7,5)
                # For black: king from (0,4) to (0,6), rook from (0,7) to (0,5)
                row = 7 if current_player_color == 'white' else 0
                from_pos = (row, 4)  # King start position for castling
                to_pos = (row, 6)   # King end position for kingside castling

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            elif move_notation.lower() in ['o-o-o', '0-0-0']:  # Queenside castling
                # For white: king from (7,4) to (7,2), rook from (7,0) to (7,3)
                # For black: king from (0,4) to (0,2), rook from (0,0) to (0,3)
                row = 7 if current_player_color == 'white' else 0
                from_pos = (row, 4)  # King start position for castling
                to_pos = (row, 2)   # King end position for queenside castling

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            # Handle coordinate format "e2e4" (source to destination)
            coord_format_pattern = r'^([a-h][1-8])([a-h][1-8])$'
            coord_match = re.match(coord_format_pattern, move_notation.lower())
            if coord_match:
                from_sq = coord_match.group(1)
                to_sq = coord_match.group(2)

                from_col = ord(from_sq[0]) - ord('a')
                from_row = 8 - int(from_sq[1])
                to_col = ord(to_sq[0]) - ord('a')
                to_row = 8 - int(to_sq[1])

                from_pos = (from_row, from_col)
                to_pos = (to_row, to_col)

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            # Handle standard chess notation: e4, Nf3, exd5, Bxf7+, etc.
            # Check if the move looks like proper chess notation
            chess_pattern = r'^([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][KQRBN]?[+#]?)$'
            if re.match(chess_pattern, move_notation.lower()):
                # Standard algebraic notation processing
                # Extract destination square (last 2 characters that look like a square)
                dest_match = re.search(r'([a-h][1-8])$', move_notation.lower())
                if dest_match:
                    dest_sq = dest_match.group(1)
                    dest_col = ord(dest_sq[0]) - ord('a')
                    dest_row = 8 - int(dest_sq[1])
                    dest_pos = (dest_row, dest_col)

                    # If the notation is just a destination square (like 'e4'), try to find the piece that can move there
                    if len(move_notation.strip()) == 2:  # Simple notation like 'e4'
                        # Find all pieces of current player's color that can move to this destination
                        for row_idx in range(8):
                            for col_idx in range(8):
                                piece = chess_game.get_piece_at(row_idx, col_idx)
                                if piece and chess_game.is_own_piece(piece, current_player_color):
                                    # Check if this piece can move to the destination
                                    temp_moves = chess_game._get_piece_valid_moves(row_idx, col_idx)
                                    if dest_pos in temp_moves:
                                        from_pos = (row_idx, col_idx)
                                        if chess_game.is_move_legal(from_pos, dest_pos, current_player_color):
                                            return True, from_pos, dest_pos
                        return False, None, None  # No valid piece can make this move

                    # For piece moves like "Nf3", "Bc4+" etc., extract piece and destination
                    piece_dest_match = re.search(r'^([KQRBN])([a-h][1-8])$', move_notation.lower().replace('+', '').replace('#', ''))
                    if piece_dest_match:
                        piece_letter = piece_dest_match.group(1).upper()
                        to_sq = piece_dest_match.group(2)

                        # Find the piece on the board that matches the piece type and can move to the destination
                        for row_idx in range(8):
                            for col_idx in range(8):
                                piece = chess_game.get_piece_at(row_idx, col_idx)
                                if piece and chess_game.is_own_piece(piece, current_player_color):
                                    # Check if this piece type matches and can move to the destination
                                    piece_type = piece.upper()[0]  # Get first character: R, N, B, Q, K
                                    if piece_type == piece_letter:
                                        # Check if this piece can move to the destination
                                        temp_moves = chess_game._get_piece_valid_moves(row_idx, col_idx)
                                        if dest_pos in temp_moves:
                                            from_pos = (row_idx, col_idx)
                                            if chess_game.is_move_legal(from_pos, dest_pos, current_player_color):
                                                return True, from_pos, dest_pos
                        return False, None, None  # No valid piece of type can make this move

                    # For more complex notation involving capture or source disambiguation
                    complex_match = re.search(r'^([KQRBN])([a-h])?([1-8])?x?([a-h][1-8])[+#]?$', move_notation.replace('+', '').replace('#', ''))
                    if complex_match:
                        piece_letter = complex_match.group(1).upper()
                        source_file = complex_match.group(2)  # Optional source file
                        source_rank = complex_match.group(3)  # Optional source rank
                        to_sq = complex_match.group(4)  # Destination

                        to_col = ord(to_sq[0]) - ord('a')
                        to_row = 8 - int(to_sq[1])
                        to_pos = (to_row, to_col)

                        # Find the specific piece that matches and can make this move
                        valid_pieces = []
                        for row_idx in range(8):
                            for col_idx in range(8):
                                piece = chess_game.get_piece_at(row_idx, col_idx)
                                if piece and chess_game.is_own_piece(piece, current_player_color):
                                    piece_type = piece.upper()[0]
                                    if piece_type == piece_letter:
                                        # Check if this piece can move to the destination
                                        temp_moves = chess_game._get_piece_valid_moves(row_idx, col_idx)
                                        if to_pos in temp_moves:
                                            # Check if it matches source disambiguation if present
                                            file_ok = not source_file or chr(ord('a') + col_idx) == source_file
                                            rank_ok = not source_rank or str(8 - row_idx) == source_rank
                                            if file_ok and rank_ok:
                                                valid_pieces.append((row_idx, col_idx))

                        # Check for context clues indicating this is a reference to a previous move vs. a new move
                        # If the AI is asking opponent to counter their move, that's referencing, not making a new move
                        reference_indicators = [
                            r'(?:how\s+will\s+you\s+|you\s+will\s+|how\s+can\s+you\s+|will\s+you\s+|you\s+should\s+|you\s+need\s+to\s+)counter',
                            r'(?:how\s+will\s+you\s+|you\s+will\s+|how\s+can\s+you\s+|will\s+you\s+|you\s+should\s+|you\s+need\s+to\s+)respond\s+to',
                            r'(?:against\s+my\s+|in\s+response\s+to\s+my\s+|to\s+counter\s+my\s+|my\s+move\s+was|my\s+last\s+move)',
                            r'(?:my\s+move\s+is\s+|\s+\d+\.\s+[KQRBN]?[a-h]?[1-8]?[a-h][1-8]\s+)',  # Captures "my move is [notation]" or "[number]. [notation]"
                            r'(?:in\s+the\s+position\s+after\s+|following\s+|after\s+my\s+|my\s+previous)',
                            r'(?:as\s+I\s+played|as\s+I\s+moved|like\s+I\s+played|like\s+I\s+did|like\s+my\s+move)',
                        ]

                        is_reference = any(
                            re.search(pattern, response_text, re.IGNORECASE)
                            for pattern in reference_indicators
                        )

                        # Only process as new move if it's NOT a reference to historical moves
                        if not is_reference:
                            # If we found exactly one valid piece, make the move
                            if len(valid_pieces) == 1:
                                from_pos = valid_pieces[0]
                                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                                    return True, from_pos, to_pos
                            # If multiple pieces could make the move without disambiguation, return failure
                            elif len(valid_pieces) > 1:
                                # Try to resolve with additional disambiguation if available
                                for from_pos in valid_pieces:
                                    if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                                        return True, from_pos, to_pos
                                return False, None, None
                        else:
                            # This looks like a reference to a previous move, not a new move
                            return False, None, None  # Don't process historical references as new moves

                    # For capture notation like "exd5", "Nxg7", etc.
                    capture_match = re.search(r'^([KQRBN]?)([a-h])?([1-8])?x([a-h][1-8])[+#]?$', move_notation.replace('+', '').replace('#', ''))
                    if capture_match:
                        piece_letter = capture_match.group(1).upper()
                        source_file = capture_match.group(2)  # Optional source file
                        source_rank = capture_match.group(3)  # Optional source rank
                        to_sq = capture_match.group(4)  # Destination

                        to_col = ord(to_sq[0]) - ord('a')
                        to_row = 8 - int(to_sq[1])
                        to_pos = (to_row, to_col)

                        # Find the piece that matches and can capture at destination
                        valid_pieces = []
                        for row_idx in range(8):
                            for col_idx in range(8):
                                piece = chess_game.get_piece_at(row_idx, col_idx)
                                if piece and chess_game.is_own_piece(piece, current_player_color):
                                    # If piece type specified, check it matches
                                    piece_type_ok = True
                                    if piece_letter:
                                        piece_type = piece.upper()[0]
                                        piece_type_ok = (piece_type == piece_letter)

                                    if piece_type_ok:
                                        # Check if this piece can move to capture at destination
                                        temp_moves = chess_game._get_piece_valid_moves(row_idx, col_idx)
                                        if to_pos in temp_moves:
                                            # Check if it matches source disambiguation if present
                                            file_ok = not source_file or chr(ord('a') + col_idx) == source_file
                                            rank_ok = not source_rank or str(8 - row_idx) == source_rank
                                            if file_ok and rank_ok:
                                                valid_pieces.append((row_idx, col_idx))

                        # If we found exactly one valid piece, make the move
                        if len(valid_pieces) == 1:
                            from_pos = valid_pieces[0]
                            if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                                return True, from_pos, to_pos
                        # If multiple pieces could make the move, return failure
                        elif len(valid_pieces) > 1:
                            return False, None, None

            # If it's not standard algebraic notation, check if it's in the form "e2 to e4" or "e2-e4" or "e2,e4"
            coord_to_coord_pattern = r'([a-h][1-8])\s*(?:to|[-,])\s*([a-h][1-8])'
            match = re.search(coord_to_coord_pattern, move_notation, re.IGNORECASE)
            if match:
                from_sq = match.group(1).lower()
                to_sq = match.group(2).lower()

                # Convert to coordinates
                from_col = ord(from_sq[0]) - ord('a')
                from_row = 8 - int(from_sq[1])
                to_col = ord(to_sq[0]) - ord('a')
                to_row = 8 - int(to_sq[1])

                from_pos = (from_row, from_col)
                to_pos = (to_row, to_col)

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            # If we couldn't parse it, return failure
            return False, None, None

        def parse_ttt_json_response(response_text, character_name):
            """Parse JSON response specifically for tic-tac-toe, extracting dialogue, move, and board state."""
            import json
            import re

            # First and ONLY try: Look for properly formatted JSON with BOTH required fields
            json_pattern = r'(\{[^{}]*("dialogue"|\'dialogue\')[^{}]*:[^{}]*["\'][^{}]*["\'][^{}]*,?[^{}]*("move"|\'move\')[^{}]*:[^{}]*["\'][^{}]*["\'][^{}]*\})'
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            if matches:
                for match_tuple in matches:
                    try:
                        json_clean = match_tuple[0].strip()
                        parsed = json.loads(json_clean)

                        dialogue = parsed.get('dialogue', '')
                        move = parsed.get('move', '').strip()
                        board_state = parsed.get('board_state', '') or parsed.get('game_state', '')

                        # Validate that BOTH required fields are present and non-empty
                        if dialogue.strip() and move.strip():
                            return dialogue, move, board_state
                    except json.JSONDecodeError:
                        continue  # Try the next match

            # CRITICAL: If no valid JSON found with both required fields, return no move
            # This forces the game to recognize that the AI didn't follow the required format
            # and will trigger the safeguards to advance to the next player
            return response_text, "", ""

        def parse_rps_json_response(response_text, character_name):
            """Parse JSON response specifically for rock-paper-scissors, extracting dialogue, choice, and reasoning."""
            import json
            import re

            # First try to find JSON within the response
            json_pattern = r'\{[^{}]*\}'  # Non-greedy match for JSON objects
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            if matches:
                for json_str in matches:
                    try:
                        json_clean = json_str.strip()
                        parsed = json.loads(json_clean)

                        dialogue = parsed.get('dialogue', '')
                        move = parsed.get('move', '').strip()  # Expected to be rock, paper, or scissors
                        reasoning = parsed.get('reasoning', '')

                        if dialogue or move:
                            return dialogue, move, reasoning
                    except json.JSONDecodeError:
                        continue  # Try the next match

            # If no JSON found, try to extract rock-paper-scissors specific choices
            # Look for keywords: rock, paper, scissors, or their abbreviations
            rps_pattern = r'\b(rock|paper|scissors|r|p|s)\b'
            rps_matches = re.findall(rps_pattern, response_text, re.IGNORECASE)

            if rps_matches:
                for choice in rps_matches:
                    # Normalize the choice
                    normalized_choice = choice.lower()
                    if normalized_choice in ['r', 'rock']:
                        final_choice = 'rock'
                    elif normalized_choice in ['p', 'paper']:
                        final_choice = 'paper'
                    elif normalized_choice in ['s', 'scissors']:
                        final_choice = 'scissors'
                    else:
                        continue  # Skip unrecognized matches

                    # Extract the sentence containing the choice
                    sentences = re.split(r'[.!?]+', response_text)
                    for sentence in sentences:
                        if choice.lower() in sentence.lower():
                            # Remove the choice word from dialogue
                            dialogue_text = re.sub(r'\b' + re.escape(choice) + r'\b', '', response_text, re.IGNORECASE).strip()
                            dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                            return dialogue_text, final_choice, ""

            # If no rps-specific choice found, return as dialogue only
            return response_text, "", ""

        def parse_hangman_json_response(response_text, character_name):
            """Parse JSON response specifically for hangman, extracting dialogue, guessed letter, and reasoning."""
            import json
            import re

            # First try to find JSON within the response
            json_pattern = r'\{[^{}]*\}'  # Non-greedy match for JSON objects
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            if matches:
                for json_str in matches:
                    try:
                        json_clean = json_str.strip()
                        parsed = json.loads(json_clean)

                        dialogue = parsed.get('dialogue', '')
                        letter = parsed.get('letter', '').strip()
                        reasoning = parsed.get('reasoning', '')

                        if dialogue or letter:
                            return dialogue, letter, reasoning
                    except json.JSONDecodeError:
                        continue  # Try the next match

            # CRITICAL: If no valid JSON found with both required fields, return no letter
            # This forces the game to recognize that the AI didn't follow the required format
            # and will trigger the safeguards to advance to the next player
            return response_text, "", ""

        def parse_twentyone_json_response(response_text, character_name):
            """Parse JSON response specifically for twenty-one, extracting dialogue, action, and reasoning."""
            import json
            import re

            # First try to find JSON within the response
            json_pattern = r'\{[^{}]*\}'  # Non-greedy match for JSON objects
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            if matches:
                for json_str in matches:
                    try:
                        json_clean = json_str.strip()
                        parsed = json.loads(json_clean)

                        dialogue = parsed.get('dialogue', '')
                        action = parsed.get('action', '').strip()  # Expected to be 'hit' or 'stand'
                        reasoning = parsed.get('reasoning', '')

                        if dialogue or action:
                            return dialogue, action, reasoning
                    except json.JSONDecodeError:
                        continue  # Try the next match

            # CRITICAL: If no valid JSON found with both required fields, return no action
            # This forces the game to recognize that the AI didn't follow the required format
            # and will trigger the safeguards to advance to the next player
            return response_text, "", ""

        # Specific function to parse chess move notation
        def parse_number_guessing_json_response(response_text, character_name):
            """Parse the JSON response specifically for number guessing, extracting dialogue, number guess, and strategy."""
            import json
            import re

            # First try to find JSON within the response
            json_pattern = r'\{[^{}]*\}'  # Non-greedy match for JSON objects
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            if matches:
                for json_str in matches:
                    try:
                        json_clean = json_str.strip()
                        parsed = json.loads(json_clean)

                        dialogue = parsed.get('dialogue', '')
                        number = parsed.get('number', '').strip()  # Expected to be a number
                        strategy = parsed.get('strategy', '') or parsed.get('reasoning', '')

                        # Validate that BOTH required fields are present and non-empty
                        if dialogue.strip() and number.strip():
                            return dialogue, number, strategy
                    except json.JSONDecodeError:
                        continue  # Try the next match

            # CRITICAL: If no valid JSON found with both required fields, return no number
            # This forces the game to recognize that the AI didn't follow the required format
            # and will trigger the safeguards to advance to the next player
            return response_text, "", ""

        def parse_word_association_json_response(response_text, character_name):
            """Parse JSON response specifically for word association, extracting dialogue, word, and connection."""
            import json
            import re

            # First try to find JSON within the response
            json_pattern = r'\{[^{}]*\}'  # Non-greedy match for JSON objects
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            if matches:
                for json_str in matches:
                    try:
                        json_clean = json_str.strip()
                        parsed = json.loads(json_clean)

                        dialogue = parsed.get('dialogue', '')
                        word = parsed.get('word', '').strip()
                        connection = parsed.get('connection', '') or parsed.get('reasoning', '')

                        if dialogue or word:
                            return dialogue, word, connection
                    except json.JSONDecodeError:
                        continue  # Try the next match

            # CRITICAL: If no valid JSON found with both required fields, return no word
            # This forces the game to recognize that the AI didn't follow the required format
            # and will trigger the safeguards to advance to the next player
            return response_text, "", ""

        # Specific function to parse chess move notation
        def parse_move_notation(move_notation, chess_game, current_player_color):
            """Parse algebraic move notation and return the from/to positions."""
            import re

            if not move_notation:
                return False, None, None

            move_notation = move_notation.strip()

            # If move notation is empty, return failure
            if not move_notation:
                return False, None, None

            # Handle castling notation first
            if move_notation.lower() in ['o-o', '0-0']:  # Kingside castling
                # For white: king from (7,4) to (7,6), rook from (7,7) to (7,5)
                # For black: king from (0,4) to (0,6), rook from (0,7) to (0,5)
                row = 7 if current_player_color == 'white' else 0
                from_pos = (row, 4)  # King start position for castling
                to_pos = (row, 6)   # King end position for kingside castling

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            elif move_notation.lower() in ['o-o-o', '0-0-0']:  # Queenside castling
                # For white: king from (7,4) to (7,2), rook from (7,0) to (7,3)
                # For black: king from (0,4) to (0,2), rook from (0,0) to (0,3)
                row = 7 if current_player_color == 'white' else 0
                from_pos = (row, 4)  # King start position for castling
                to_pos = (row, 2)   # King end position for queenside castling

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            # Handle coordinate format "e2e4" (source to destination)
            coord_format_pattern = r'^([a-h][1-8])([a-h][1-8])$'
            coord_match = re.match(coord_format_pattern, move_notation.lower())
            if coord_match:
                from_sq = coord_match.group(1)
                to_sq = coord_match.group(2)

                from_col = ord(from_sq[0]) - ord('a')
                from_row = 8 - int(from_sq[1])
                to_col = ord(to_sq[0]) - ord('a')
                to_row = 8 - int(to_sq[1])

                from_pos = (from_row, from_col)
                to_pos = (to_row, to_col)

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            # Handle standard chess notation: e4, Nf3, exd5, Bxf7+, etc.
            # Check if the move looks like proper chess notation
            chess_pattern = r'^([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][KQRBN]?[+#]?)$'
            if re.match(chess_pattern, move_notation.lower()):
                # Standard algebraic notation processing
                # Extract destination square (last 2 characters that look like a square)
                dest_match = re.search(r'([a-h][1-8])$', move_notation.lower())
                if dest_match:
                    dest_sq = dest_match.group(1)
                    dest_col = ord(dest_sq[0]) - ord('a')
                    dest_row = 8 - int(dest_sq[1])
                    dest_pos = (dest_row, dest_col)

                    # If the notation is just a destination square (like 'e4'), try to find the piece that can move there
                    if len(move_notation.strip()) == 2:  # Simple notation like 'e4'
                        # Find all pieces of current player's color that can move to this destination
                        for row_idx in range(8):
                            for col_idx in range(8):
                                piece = chess_game.get_piece_at(row_idx, col_idx)
                                if piece and chess_game.is_own_piece(piece, current_player_color):
                                    # Check if this piece can move to the destination
                                    temp_moves = chess_game._get_piece_valid_moves(row_idx, col_idx)
                                    if dest_pos in temp_moves:
                                        from_pos = (row_idx, col_idx)
                                        if chess_game.is_move_legal(from_pos, dest_pos, current_player_color):
                                            return True, from_pos, dest_pos
                        return False, None, None  # No valid piece can make this move

                    # For more complex notation, handle different cases
                    # First, check if this is a piece move like "Nf3", "Bc4", "Qh5", etc.
                    piece_move_pattern = r'^([KQRBN])([a-h][1-8])$'
                    piece_move_match = re.match(piece_move_pattern, move_notation.upper().replace('+', '').replace('#', '').replace('x', ''))
                    if piece_move_match:
                        piece_letter = piece_move_match.group(1)  # K, Q, R, B, or N
                        # dest_pos already found from earlier matching
                        # Now find which piece of the specified type can move to dest_pos

                        for row_idx in range(8):
                            for col_idx in range(8):
                                piece = chess_game.get_piece_at(row_idx, col_idx)
                                if piece and chess_game.is_own_piece(piece, current_player_color):
                                    # Get piece type (first character, uppercase)
                                    actual_piece_type = piece.upper()[0]
                                    # For pawns, the notation is just "e4" not "Pe4", so pawn moves are handled separately
                                    if actual_piece_type == piece_letter:
                                        # Check if this piece can move to the destination
                                        possible_moves = chess_game._get_piece_valid_moves(row_idx, col_idx)
                                        if dest_pos in possible_moves:
                                            from_pos = (row_idx, col_idx)
                                            if chess_game.is_move_legal(from_pos, dest_pos, current_player_color):
                                                return True, from_pos, dest_pos
                        return False, None, None  # No valid piece of this type can make the move

                    # Then, handle capture notation like "Nxf3", "Bxe5", etc.
                    piece_capture_pattern = r'^([KQRBN])x?([a-h][1-8])$'
                    piece_capture_match = re.match(piece_capture_pattern, move_notation.upper().replace('+', '').replace('#', ''))
                    if piece_capture_match:
                        piece_letter = piece_capture_match.group(1)  # K, Q, R, B, or N
                        # dest_pos already found from earlier matching

                        for row_idx in range(8):
                            for col_idx in range(8):
                                piece = chess_game.get_piece_at(row_idx, col_idx)
                                if piece and chess_game.is_own_piece(piece, current_player_color):
                                    actual_piece_type = piece.upper()[0]
                                    if actual_piece_type == piece_letter:
                                        # Check if this piece can move to the destination (capturing)
                                        possible_moves = chess_game._get_piece_valid_moves(row_idx, col_idx)
                                        if dest_pos in possible_moves:
                                            from_pos = (row_idx, col_idx)
                                            if chess_game.is_move_legal(from_pos, dest_pos, current_player_color):
                                                return True, from_pos, dest_pos
                        return False, None, None  # No valid piece of this type can make the capture

                    # Then handle disambiguation moves like "Nbf3", "Raxd1", "N1f3", etc.
                    disambiguation_pattern = r'^([KQRBN])([a-h]|[1-8])([a-h][1-8])$'
                    disambiguation_match = re.match(disambiguation_pattern, move_notation.upper().replace('+', '').replace('#', '').replace('x', ''))
                    if disambiguation_match:
                        piece_letter = disambiguation_match.group(1)  # K, Q, R, B, or N
                        disambiguator = disambiguation_match.group(2)  # File (a-h) or rank (1-8)
                        target_sq = disambiguation_match.group(3)      # Target square

                        if len(disambiguator) == 1 and len(target_sq) == 2:  # Like Nf3 where f is the disambiguator
                            target_col = ord(target_sq[0]) - ord('a')
                            target_row = 8 - int(target_sq[1])
                            target_pos = (target_row, target_col)

                            for row_idx in range(8):
                                for col_idx in range(8):
                                    piece = chess_game.get_piece_at(row_idx, col_idx)
                                    if piece and chess_game.is_own_piece(piece, current_player_color):
                                        actual_piece_type = piece.upper()[0]
                                        if actual_piece_type == piece_letter:
                                            # Check if piece file or rank matches the disambiguator
                                            piece_file = chr(ord('a') + col_idx)
                                            piece_rank = str(8 - row_idx)

                                            # Match if either file or rank matches disambiguator
                                            file_matches = disambiguator == piece_file
                                            rank_matches = disambiguator == piece_rank

                                            if (file_matches or rank_matches):
                                                # Check if this piece can move to target
                                                possible_moves = chess_game._get_piece_valid_moves(row_idx, col_idx)
                                                if target_pos in possible_moves:
                                                    from_pos = (row_idx, col_idx)
                                                    if chess_game.is_move_legal(from_pos, target_pos, current_player_color):
                                                        return True, from_pos, target_pos
                            return False, None, None  # No qualifying piece found

                    # For other complex notations (like "e2e4", "d7d8Q", etc.)
                    else:
                        # Look for source square in the notation (e.g., "e2e4", "d7d8Q", etc.)
                        source_match = re.search(r'([a-h][1-8]).*?([a-h][1-8])', move_notation.lower())
                        if source_match:
                            from_sq = source_match.group(1)
                            to_sq = source_match.group(2)  # Should match dest_sq

                            from_col = ord(from_sq[0]) - ord('a')
                            from_row = 8 - int(from_sq[1])
                            from_pos = (from_row, from_col)

                            if chess_game.is_move_legal(from_pos, dest_pos, current_player_color):
                                return True, from_pos, dest_pos

            # If it's not standard algebraic notation, check if it's in the form "e2 to e4" or "e2-e4" or "e2,e4"
            coord_to_coord_pattern = r'([a-h][1-8])\s*(?:to|[-,])\s*([a-h][1-8])'
            match = re.search(coord_to_coord_pattern, move_notation, re.IGNORECASE)
            if match:
                from_sq = match.group(1).lower()
                to_sq = match.group(2).lower()

                # Convert to coordinates
                from_col = ord(from_sq[0]) - ord('a')
                from_row = 8 - int(from_sq[1])
                to_col = ord(to_sq[0]) - ord('a')
                to_row = 8 - int(to_sq[1])

                from_pos = (from_row, from_col)
                to_pos = (to_row, to_col)

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            # If we couldn't parse it, return failure
            return False, None, None

        # Check for game mode specific constraints
        if args.chess and args.max_turns != MAX_TURNS:
            print(f"⚠️  WARNING: --max-turns parameter is not recommended in chess mode as games continue until completion")
            print(f"   Chess games will continue until a winner is determined, regardless of turn count.")
        elif (args.tic_tac_toe or args.rock_paper_scissors or args.hangman or args.twenty_one or args.number_guessing or args.word_association or args.connect_four or args.uno) and args.max_turns != MAX_TURNS:
            print(f"⚠️  WARNING: --max-turns parameter is not recommended in game modes as games continue until completion")
            print(f"   Games will continue until a winner is determined, regardless of turn count.")

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

        # Update API client module with runtime configuration
        from api_client import update_config
        update_config(args.api_endpoint, api_key, args.model)
        
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

        # Define token/history management functions
        def estimate_token_count(text: str) -> int:
            """Estimate token count by counting words/symbols. Approximation for token limit purposes."""
            # Simple approximation: count words (typically 1.3-1.5 tokens per word for English)
            import re
            words = re.findall(r'\b\w+\b', text)
            # Using a conservative estimate of ~1.3 tokens per word to approximate GPT tokenization
            return int(len(words) * 1.3)

        def limit_history_window(full_history: list, max_tokens: int = 1000) -> list:
            """
            Limit the history to a sliding window that fits within token limits.
            Keeps the most recent entries while ensuring token limit is respected.
            """
            if not full_history:
                return full_history

            # Estimate total tokens in history
            history_text = " ".join([f"{entry['name']}: {entry['content']}" for entry in full_history])
            total_estimated_tokens = estimate_token_count(history_text)

            if total_estimated_tokens <= max_tokens:
                return full_history  # No need to limit if within token budget

            # Start with most recent entries and work backwards to fit within token limit
            limited_history = []
            current_tokens = 0

            # Check if first entry is Narrator context and preserve it if possible
            initial_narrator = None
            entries_to_process = full_history
            if full_history and len(full_history) > 0 and full_history[0]['name'] == 'Narrator':
                initial_context = full_history[0]
                initial_tokens = estimate_token_count(f"{initial_context['name']}: {initial_context['content']}")
                if initial_tokens < max_tokens * 0.30:  # Use up to 30% of tokens for initial context
                    initial_narrator = initial_context
                    current_tokens = initial_tokens
                    # Process remaining entries excluding the first narrator entry
                    entries_to_process = full_history[1:]
                else:
                    # Don't preserve narrator if it takes too much space
                    entries_to_process = full_history[:]
            else:
                entries_to_process = full_history[:]

            # Add entries from the end (most recent) backwards until we reach token limit
            # We'll reverse them later to maintain chronological order
            temp_entries = []
            for i in range(len(entries_to_process) - 1, -1, -1):
                entry = entries_to_process[i]
                entry_text = f"{entry['name']}: {entry['content']}"
                entry_tokens = estimate_token_count(entry_text)

                if current_tokens + entry_tokens <= max_tokens:
                    temp_entries.append(entry)  # Add to temp list, we'll reverse later
                    current_tokens += entry_tokens
                else:
                    break  # Reached token limit

            # Reverse the temporary list to restore chronological order
            temp_entries.reverse()

            # If we preserved the narrator, add it to the front
            if initial_narrator:
                return [initial_narrator] + temp_entries
            else:
                return temp_entries

        # Determine game mode
        if args.chess:
            print("♔♖♗♕♔♗♖♘ Playing Chess Mode Activated ♘♖♗♔♕♗♖♔")
            print("=" * 80)

            # In chess mode, we only support 2 characters and ignore max-turns parameter
            if len(characters) != 2:
                print("❌ Chess mode requires exactly 2 characters")
                sys.exit(1)

            # Assign players to chess colors based on order (first character is white)
            white_player = characters[0]
            black_player = characters[1]
            white_player['chess_color'] = 'white'
            black_player['chess_color'] = 'black'

            # Initialize chess game
            chess_game = ChessGame()
            print("Initial Chess Board:")
            print(chess_game.print_board())

            # Initialize conversation history with chess context
            chess_history = []

            # Add opening chess context to history
            opening_context = f"Starting a game of chess. {white_player['name']} is playing as white, and {black_player['name']} is playing as black."
            chess_history.append({'name': 'Narrator', 'content': opening_context})
            print(f"[TURN 1] Narrator: {opening_context}")

            # Chess game loop
            game_count = 1
            turn = 1  # Main turn counter

            while True:
                print(f"\n{'='*80}")
                print(f"CHESS GAME #{game_count}")
                print('='*80)

                # Reset chess game for each game (in case of draws)
                chess_game = ChessGame()

                # Add game start context to history
                game_start_context = f"Starting chess game #{game_count}. {white_player['name']} (white) vs {black_player['name']} (black)."
                chess_history.append({'name': 'Narrator', 'content': game_start_context})
                print(f"[TURN {turn}] Narrator: {game_start_context}")
                turn += 1

                # Chess game turn counter
                chess_turn = 1

                # Start the chess game loop
                while not chess_game.game_over:
                    print(f"\n{'='*60}")
                    print(f"[CHESS TURN {chess_turn}] Board Position:")
                    print(chess_game.print_board())
                    print(f"Move History: {' '.join(chess_game.move_history)}")
                    print(f"Current Player: {chess_game.current_player}")
                    print('='*60)

                    # Determine who is making the move
                    if chess_game.current_player == 'white':
                        current_char = white_player
                        other_char = black_player
                    else:
                        current_char = black_player
                        other_char = white_player

                    print(f"\n[TURN {turn}] {current_char['name']} (playing as {chess_game.current_player})")

                    try:
                        # Create chess context for the turn, requesting explicit JSON output
                        chess_context = f"""
CRITICAL CHESS GAME INSTRUCTIONS - YOU MUST FOLLOW THE EXACT JSON FORMAT BELOW OR THE GAME WILL STALL:

CURRENT BOARD POSITION:
{chess_game.print_board()}

GAME STATE:
- Move History: {' '.join(chess_game.move_history)}
- Current Player: {chess_game.current_player}
- Your Color: {current_char['chess_color']}

YOU MUST RESPOND IN EXACTLY THIS JSON FORMAT (NO EXCEPTIONS):
{{
  "dialogue": "Your dialogue and thought process about the position",
  "move": "YOUR CHESS MOVE IN ALGEBRAIC NOTATION ONLY (e.g., 'e4', 'Nf3', 'O-O', 'exd5')",
  "board_state": "Board visualization after your move (optional)"
}}

ABSOLUTELY CRITICAL REQUIREMENTS:
1. PROVIDE EXACTLY THE THREE FIELDS: "dialogue", "move", "board_state"
2. THE "move" FIELD MUST CONTAIN EXCLUSIVELY VALID ALGEBRAIC NOTATION
3. DO NOT WRAP MOVES IN ASTERISKS, BOLD TEXT, OR NARRATIVE FORM
4. DO NOT SAY "My move is e4" OR "**e4**" OR "I will play e4"
5. PUT ONLY THE MOVE NOTATION IN THE "move" FIELD: 'e4', 'Nf3', 'O-O', etc.
6. YOUR MOVE MUST BE LEGAL IN THE CURRENT BOARD POSITION

EXAMPLE OF CORRECT FORMAT:
{{
  "dialogue": "I'm planning to develop my knight and control the center",
  "move": "Nf3",
  "board_state": "[board visualization]"
}}

EXAMPLES OF INCORRECT FORMATS THAT WILL BE REJECTED:
- Just saying "e4" in narrative text
- Writing "**e4**" or "***e4***"
- Saying "My move is e4" without proper JSON
- Providing only dialogue without move field
- Any format that does not match exact JSON specification above

VIOLATION OF JSON FORMAT RULES WILL RESULT IN:
- Turn immediately passed to opponent
- No credit given for narrative text provided
- Your turn will be skipped until proper JSON format is used
- Up to 3 format violations allowed before automatic forfeit

Think carefully and respond in the EXACT JSON format specified above.
                        """.strip()

                        # Extract lorebook entries based on game context keywords
                        lorebook_entries = []
                        game_scenario = "Playing a game of chess"
                        from character_loader import extract_lorebook_entries
                        # Search for keywords in game scenario that match character lorebooks
                        scenario_keywords = game_scenario.lower().split()
                        for char in characters:
                            entries = extract_lorebook_entries(char['raw_data'], chess_history, max_entries=2)
                            # Filter entries by scenario relevance
                            relevant_entries = []
                            for entry in entries:
                                if any(keyword in entry.lower() for keyword in scenario_keywords if len(keyword) > 3):
                                    relevant_entries.append(entry)
                            lorebook_entries.extend(relevant_entries[:1])  # Add at most 1 relevant entry per character

                        # Limit history to avoid token overflow while keeping recent context
                        limited_chess_history = limit_history_window(chess_history)

                        # Generate response with chess context
                        resp = generate_response_adaptive(
                            current_char, other_char, limited_chess_history, turn,
                            enable_environmental=not args.no_environmental,
                            similarity_threshold=args.similarity,
                            verbose=args.verbose,
                            scenario_context="Playing a game of chess",
                            lorebook_entries=lorebook_entries if lorebook_entries else None
                        )

                        # Validate response
                        if not isinstance(resp, str):
                            resp = str(resp)

                        chess_history.append({'name': current_char['name'], 'content': resp})
                        print(resp)

                        # Parse the JSON response
                        dialogue, move_notation, board_state = parse_chess_json_response(resp, current_char['name'])

                        if dialogue:
                            print(f"💬 Dialogue: {dialogue}")

                        # Track consecutive failed moves to prevent infinite loops
                        if not hasattr(chess_game, 'consecutive_failed_moves'):
                            chess_game.consecutive_failed_moves = {'white': 0, 'black': 0}
                        if not hasattr(chess_game, 'last_failed_move'):
                            chess_game.last_failed_move = {'white': '', 'black': ''}

                        # Track if a real move was successfully processed in this turn
                        move_was_processed = False

                        # Attempt to make the chess move if notation is provided
                        if move_notation:
                            # Try to parse the move notation
                            success, from_pos, to_pos = parse_move_notation(move_notation, chess_game, current_char['chess_color'])

                            if success and from_pos and to_pos:
                                move_success = chess_game.make_move(from_pos, to_pos)
                                if move_success:
                                    print(f"✅ Move successfully made: {chess_game.move_history[-1] if chess_game.move_history else move_notation}")
                                    turn += 1  # Move was successful, increment turn
                                    chess_turn += 1  # Also increment chess turn
                                    move_was_processed = True
                                    # Reset consecutive failed moves for this player
                                    chess_game.consecutive_failed_moves[current_char['chess_color']] = 0
                                    chess_game.last_failed_move[current_char['chess_color']] = ''
                                else:
                                    print(f"❌ Move failed - illegal move attempted: {move_notation}")
                                    # Add feedback to history
                                    feedback = f"Your move '{move_notation}' was invalid or illegal. Please try again with a valid chess move from the current position. REMEMBER: You must respond in proper JSON format with both 'dialogue' and 'move' fields."
                                    chess_history.append({'name': 'Referee', 'content': feedback})
                                    # Increment consecutive failed moves counter
                                    chess_game.consecutive_failed_moves[current_char['chess_color']] += 1
                                    chess_game.last_failed_move[current_char['chess_color']] = move_notation
                                    # If too many consecutive failures, force move to other player to prevent infinite loops
                                    if chess_game.consecutive_failed_moves[current_char['chess_color']] >= 2:
                                        print(f"⚠️  {current_char['name']} has failed to make a valid move {chess_game.consecutive_failed_moves[current_char['chess_color']]} times. Moving to next player to prevent infinite loop.")
                                        # ALSO switch the current player in the chess game object to ensure the game state progresses correctly
                                        chess_game.current_player = 'black' if chess_game.current_player == 'white' else 'white'
                                        turn += 1  # Force increment to prevent infinite loops
                                        move_was_processed = True
                                        chess_game.consecutive_failed_moves[current_char['chess_color']] = 0
                                        chess_game.last_failed_move[current_char['chess_color']] = ''
                            else:
                                print(f"❌ Could not parse move notation: {move_notation}")
                                # Add feedback to history
                                feedback = f"I couldn't parse the move '{move_notation}'. Please provide a valid chess move in algebraic notation. REMEMBER: You must respond in proper JSON format with both 'dialogue' and 'move' fields."
                                chess_history.append({'name': 'Referee', 'content': feedback})
                                # Increment consecutive failed moves counter
                                chess_game.consecutive_failed_moves[current_char['chess_color']] += 1
                                chess_game.last_failed_move[current_char['chess_color']] = move_notation
                                # If too many consecutive failures or the same move is repeated, force move to other player
                                if (chess_game.consecutive_failed_moves[current_char['chess_color']] >= 2 or
                                    move_notation == chess_game.last_failed_move[current_char['chess_color']]):
                                    print(f"⚠️  {current_char['name']} has failed to make a valid move {chess_game.consecutive_failed_moves[current_char['chess_color']]} times or repeated the same invalid move. Moving to next player to prevent infinite loop.")
                                    # ALSO switch the current player in the chess game object to ensure the game state progresses correctly
                                    chess_game.current_player = 'black' if chess_game.current_player == 'white' else 'white'
                                    turn += 1  # Force increment to prevent infinite loops
                                    move_was_processed = True
                                    chess_game.consecutive_failed_moves[current_char['chess_color']] = 0
                                    chess_game.last_failed_move[current_char['chess_color']] = ''
                        else:
                            print(f"⚠️  No move provided in required JSON format. Current player: {chess_game.current_player}")
                            # Add feedback to history
                            feedback = f"You MUST provide a valid chess move in the proper JSON format: {{\"dialogue\": \"your thoughts\", \"move\": \"e4\", \"board_state\": \"optional\"}}. Your response must include both 'dialogue' and 'move' fields in JSON format."
                            chess_history.append({'name': 'Referee', 'content': feedback})
                            # Increment consecutive failed moves counter
                            chess_game.consecutive_failed_moves[current_char['chess_color']] += 1
                            chess_game.last_failed_move[current_char['chess_color']] = ''
                            # If too many consecutive failures, force move to other player
                            if chess_game.consecutive_failed_moves[current_char['chess_color']] >= 2:
                                print(f"⚠️  {current_char['name']} has failed to make a valid move {chess_game.consecutive_failed_moves[current_char['chess_color']]} times. Moving to next player to prevent infinite loop.")
                                # ALSO switch the current player in the chess game object to ensure the game state progresses correctly
                                chess_game.current_player = 'black' if chess_game.current_player == 'white' else 'white'
                                turn += 1  # Force increment to prevent infinite loops
                                move_was_processed = True
                                chess_game.consecutive_failed_moves[current_char['chess_color']] = 0
                                chess_game.last_failed_move[current_char['chess_color']] = ''

                        # CRITICAL SAFEGUARD: If no move was processed, we MUST force turn advancement anyway
                        # This prevents the game from getting stuck due to parsing issues or other problems
                        if not move_was_processed and turn >= 2:  # Require at least 1 successful move before forcing
                            print(f"⚠️  CRITICAL: No move was processed. Forcing turn advancement to prevent infinite loop.")
                            # Switch the current player in the chess game for the next iteration
                            chess_game.current_player = 'black' if chess_game.current_player == 'white' else 'white'
                            turn += 1  # Force increment to break potential infinite loop
                            move_was_processed = True

                        # Periodic save every 10 turns
                        if turn % 10 == 0:
                            save_conversation_periodic(chess_history, output_file, turn)

                        # Delay before next turn
                        if not chess_game.game_over and turn < args.max_turns:  # We should still respect the max turns as a safety
                            print(f"\n⏳ Waiting {args.delay} seconds...")
                            time.sleep(args.delay)
                        elif chess_game.game_over:
                            break

                    except KeyboardInterrupt:
                        print("\n⚠️  Interrupted by user. Saving...")
                        break

                    except Exception as e:
                        print(f"\n❌ Error on turn {turn}: {e}")
                        import traceback
                        traceback.print_exc()

                        # Generate fallback response
                        from response_generator import generate_emergency_response
                        fallback_resp = generate_emergency_response(current_char, other_char, chess_history, {}, turn)
                        chess_history.append({'name': current_char['name'], 'content': fallback_resp})
                        print(fallback_resp)

                        # Periodic save every 10 turns even in exception cases
                        if turn % 10 == 0:
                            save_conversation_periodic(chess_history, output_file, turn)

                        # DON'T increment turn - same player gets another chance due to error
                        continue  # Continue to next iteration

                # Handle game end
                if chess_game.winner == 'draw':
                    game_end_context = f"Game #{game_count} ended in a draw. Starting a new game..."
                    chess_history.append({'name': 'Narrator', 'content': game_end_context})
                    print(f"🏆 Game #{game_count} ended in a draw. Starting a new game...")
                    game_count += 1
                    continue  # Continue to next game
                else:
                    winner_name = white_player['name'] if chess_game.winner == 'white' else black_player['name']
                    game_end_context = f"Game #{game_count} ended. {winner_name} wins!"
                    chess_history.append({'name': 'Narrator', 'content': game_end_context})
                    print(f"🏆 Game #{game_count} ended. {winner_name} wins!")
                    print(f"Final board position:\n{chess_game.print_board()}")
                    break  # End the overall game loop

            # After chess mode, update the history variable to save properly
            history = chess_history

        elif args.tic_tac_toe:
            print("❌⭕➕ Playing Tic-Tac-Toe Mode Activated ➕❌⭕")
            print("=" * 80)

            # In tic-tac-toe mode, we only support 2 characters
            if len(characters) != 2:
                print("❌ Tic-tac-toe mode requires exactly 2 characters")
                sys.exit(1)

            # Assign players to X and O
            player_x = characters[0]
            player_o = characters[1]
            player_x['ttt_symbol'] = 'X'
            player_o['ttt_symbol'] = 'O'

            # Initialize tic-tac-toe game
            ttt_game = TicTacToeGame()
            print("Initial Tic-Tac-Toe Board:")
            print(ttt_game.print_board())

            # Initialize conversation history with game context
            ttt_history = []

            # Add opening context to history
            opening_context = f"Starting a game of tic-tac-toe. {player_x['name']} is playing as X, and {player_o['name']} is playing as O."
            ttt_history.append({'name': 'Narrator', 'content': opening_context})
            print(f"[TURN 1] Narrator: {opening_context}")

            # Tic-tac-toe game loop
            game_count = 1
            turn = 1  # Main turn counter

            while True:
                print(f"\n{'='*80}")
                print(f"TIC-TAC-TOE GAME #{game_count}")
                print('='*80)

                # Reset game for each game (in case of draws)
                ttt_game = TicTacToeGame()

                # Add game start context to history
                game_start_context = f"Starting tic-tac-toe game #{game_count}. {player_x['name']} (X) vs {player_o['name']} (O)."
                ttt_history.append({'name': 'Narrator', 'content': game_start_context})
                print(f"[TURN {turn}] Narrator: {game_start_context}")
                turn += 1

                # Start the tic-tac-toe game loop
                ttt_turn = 1
                while not ttt_game.game_over:
                    print(f"\n{'='*60}")
                    print(f"[TIC-TAC-TOE TURN {ttt_turn}] Board Position:")
                    print(ttt_game.print_board())
                    print(f"Current Player: {ttt_game.current_player}")
                    print('='*60)

                    # Determine who is making the move
                    if ttt_game.current_player == 'X':
                        current_char = player_x
                        other_char = player_o
                    else:
                        current_char = player_o
                        other_char = player_x

                    print(f"\n[TURN {turn}] {current_char['name']} (playing as {ttt_game.current_player})")

                    try:
                        # Create game context for the turn, requesting explicit JSON output
                        ttt_context = f"""
CRITICAL TIC-TAC-TOE GAME INSTRUCTIONS - YOU MUST FOLLOW THE EXACT JSON FORMAT BELOW OR THE GAME WILL STALL:

CURRENT BOARD POSITION:
{ttt_game.print_board()}

GAME STATE:
- Current Player: {ttt_game.current_player}
- Your Symbol: {current_char['ttt_symbol']}

YOU MUST RESPOND IN EXACTLY THIS JSON FORMAT (NO EXCEPTIONS):
{{
  "dialogue": "Your dialogue and thought process about the game position",
  "move": "THE MOVE IN [ROW, COL] FORMAT ONLY (e.g., [0, 2] for top-right, [1, 1] for center)",
  "board_state": "Board visualization after your move (optional)"
}}

ABSOLUTELY CRITICAL REQUIREMENTS:
1. PROVIDE EXACTLY THE THREE FIELDS: "dialogue", "move", "board_state"
2. THE "move" FIELD MUST BE EXCLUSIVELY IN [ROW, COL] FORMAT WITH NUMBERS 0-2
3. DO NOT WRAP COORDINATES IN ASTERISKS, BOLD TEXT, OR NARRATIVE FORM
4. DO NOT SAY "I move to [0,2]" OR "[0,2]" (without brackets) OR "**[0,2]**"
5. PUT ONLY THE COORDINATE IN BRACKETS IN THE "move" FIELD: '[0, 2]', '[1, 1]', '[2, 0]', etc.
6. YOUR MOVE MUST BE ON AN UNOCCUPIED POSITION OF THE BOARD

EXAMPLE OF CORRECT FORMAT:
{{
  "dialogue": "I'm planning to take the center position to control the game",
  "move": "[1, 1]",
  "board_state": "[board visualization]"
}}

EXAMPLES OF INCORRECT FORMATS THAT WILL BE REJECTED:
- Just saying "[1, 1]" in narrative text
- Writing "**[1,1]**" or "[[1,1]]"
- Saying "I move to [1,1]" without proper JSON
- Providing only dialogue without move field
- Using non-numeric coordinates like "[a,b]" or "[one, one]"
- Any format that does not match exact JSON specification above

VIOLATION OF JSON FORMAT RULES WILL RESULT IN:
- Turn immediately passed to opponent
- No credit given for narrative text provided
- Your turn will be skipped until proper JSON format is used
- Up to 3 format violations allowed before automatic forfeit

Think carefully and respond in the EXACT JSON format specified above.
                        """.strip()

                        # Extract lorebook entries based on game context keywords
                        lorebook_entries = []
                        game_scenario = "Playing a game of tic-tac-toe"
                        from character_loader import extract_lorebook_entries
                        # Search for keywords in game scenario that match character lorebooks
                        scenario_keywords = game_scenario.lower().split()
                        for char in characters:
                            entries = extract_lorebook_entries(char['raw_data'], ttt_history, max_entries=2)
                            # Filter entries by scenario relevance
                            relevant_entries = []
                            for entry in entries:
                                if any(keyword in entry.lower() for keyword in scenario_keywords if len(keyword) > 3):
                                    relevant_entries.append(entry)
                            lorebook_entries.extend(relevant_entries[:1])  # Add at most 1 relevant entry per character

                        # Limit history to avoid token overflow while keeping recent context
                        limited_ttt_history = limit_history_window(ttt_history)

                        # Generate response with game context
                        resp = generate_response_adaptive(
                            current_char, other_char, limited_ttt_history, turn,
                            enable_environmental=not args.no_environmental,
                            similarity_threshold=args.similarity,
                            verbose=args.verbose,
                            scenario_context="Playing a game of tic-tac-toe",
                            lorebook_entries=lorebook_entries if lorebook_entries else None
                        )

                        # Validate response
                        if not isinstance(resp, str):
                            resp = str(resp)

                        ttt_history.append({'name': current_char['name'], 'content': resp})
                        print(resp)

                        # Parse the JSON response
                        dialogue, move_notation, board_state = parse_ttt_json_response(resp, current_char['name'])

                        if dialogue:
                            print(f"💬 Dialogue: {dialogue}")

                        # Track consecutive failed moves to prevent infinite loops
                        if not hasattr(ttt_game, 'consecutive_failed_moves'):
                            ttt_game.consecutive_failed_moves = {'X': 0, 'O': 0}
                        if not hasattr(ttt_game, 'last_failed_move'):
                            ttt_game.last_failed_move = {'X': '', 'O': ''}

                        # Determine current player symbol for tracking
                        current_symbol = current_char.get('ttt_symbol', 'X' if current_char == player_x else 'O')

                        # Attempt to make the move if notation is provided
                        if move_notation:
                            # Parse the move notation (should be in format like [0, 2], "0,2", "0 2", etc.)
                            import re
                            # Try to extract row, col from the move_notation
                            row, col = None, None

                            # Look for [row, col] format
                            match = re.search(r'\[([0-2])\s*,\s*([0-2])\]', move_notation)
                            if not match:
                                # Look for (row, col) format
                                match = re.search(r'\(([0-2])\s*,\s*([0-2])\)', move_notation)
                            if not match:
                                # Look for "row, col" or "row col" format with numbers 0-2
                                nums = re.findall(r'[0-2]', move_notation)
                                if len(nums) >= 2:
                                    row, col = int(nums[0]), int(nums[1])

                            if match:
                                row, col = int(match.group(1)), int(match.group(2))

                            if row is not None and col is not None:
                                success = ttt_game.make_move(row, col)
                                if success:
                                    print(f"✅ Move successfully made: {row}, {col}")
                                    turn += 1  # Move was successful, increment turn
                                    ttt_turn += 1  # Also increment tic-tac-toe turn
                                    # Reset consecutive failed moves for this player
                                    ttt_game.consecutive_failed_moves[current_symbol] = 0
                                    ttt_game.last_failed_move[current_symbol] = ''
                                else:
                                    print(f"❌ Move failed - invalid move attempted: {row}, {col} (position may be occupied or out of bounds)")
                                    # Add feedback to history
                                    feedback = f"Your move at position ({row}, {col}) was invalid or occupied. Please try again with a valid empty position."
                                    ttt_history.append({'name': 'Referee', 'content': feedback})
                                    # Increment consecutive failed moves counter
                                    ttt_game.consecutive_failed_moves[current_symbol] += 1
                                    ttt_game.last_failed_move[current_symbol] = f"{row},{col}"
                                    # If too many consecutive failures, force move to other player to prevent infinite loops
                                    if ttt_game.consecutive_failed_moves[current_symbol] >= 3:
                                        print(f"⚠️  {current_char['name']} has failed to make a valid move 3 times. Moving to next player to prevent infinite loop.")
                                        turn += 1  # Force increment to prevent infinite loops
                                        ttt_game.consecutive_failed_moves[current_symbol] = 0
                                        ttt_game.last_failed_move[current_symbol] = ''
                                    # Otherwise, same player gets another chance (don't increment turn)
                            else:
                                print(f"❌ Could not parse move notation: {move_notation}")
                                # Add feedback to history
                                feedback = f"I couldn't parse the move '{move_notation}'. Please provide a valid move in format [row, col] where row and col are 0-2."
                                ttt_history.append({'name': 'Referee', 'content': feedback})
                                # Increment consecutive failed moves counter
                                ttt_game.consecutive_failed_moves[current_symbol] += 1
                                ttt_game.last_failed_move[current_symbol] = move_notation
                                # If too many consecutive failures or the same move is repeated, force move to other player
                                if ttt_game.consecutive_failed_moves[current_symbol] >= 3 or move_notation == ttt_game.last_failed_move[current_symbol]:
                                    print(f"⚠️  {current_char['name']} has failed to make a valid move 3 times or repeated the same invalid move. Moving to next player to prevent infinite loop.")
                                    turn += 1  # Force increment to prevent infinite loops
                                    ttt_game.consecutive_failed_moves[current_symbol] = 0
                                    ttt_game.last_failed_move[current_symbol] = ''
                                # Otherwise, same player gets another chance (don't increment turn)
                        else:
                            print(f"⚠️  No move provided. Current player: {ttt_game.current_player}")
                            # Add feedback to history
                            feedback = f"Please provide a valid move in your response."
                            ttt_history.append({'name': 'Referee', 'content': feedback})
                            # Increment consecutive failed moves counter
                            ttt_game.consecutive_failed_moves[current_symbol] = 0  # Reset on no move provided since they're being prompted to make one
                            ttt_game.last_failed_move[current_symbol] = ''
                            # For no move cases, always increment turn to prevent infinite loops (this forces them to try to make a move)
                            turn += 1

                        # Periodic save every 10 turns
                        if turn % 10 == 0:
                            save_conversation_periodic(ttt_history, output_file, turn)

                        # Delay before next turn
                        if not ttt_game.game_over and turn < args.max_turns:
                            print(f"\n⏳ Waiting {args.delay} seconds...")
                            time.sleep(args.delay)
                        elif ttt_game.game_over:
                            break

                    except KeyboardInterrupt:
                        print("\n⚠️  Interrupted by user. Saving...")
                        break

                    except Exception as e:
                        print(f"\n❌ Error on turn {turn}: {e}")
                        import traceback
                        traceback.print_exc()

                        # Generate fallback response
                        from response_generator import generate_emergency_response
                        fallback_resp = generate_emergency_response(current_char, other_char, ttt_history, {}, turn)
                        ttt_history.append({'name': current_char['name'], 'content': fallback_resp})
                        print(fallback_resp)

                        # Periodic save every 10 turns even in exception cases
                        if turn % 10 == 0:
                            save_conversation_periodic(ttt_history, output_file, turn)

                        # DON'T increment turn - same player gets another chance due to error
                        continue  # Continue to next iteration

                # Handle game end
                if ttt_game.winner == 'draw':
                    game_end_context = f"Game #{game_count} ended in a draw. Starting a new game..."
                    ttt_history.append({'name': 'Narrator', 'content': game_end_context})
                    print(f"🏆 Game #{game_count} ended in a draw. Starting a new game...")
                    game_count += 1
                    continue  # Continue to next game
                else:
                    winner_name = player_x['name'] if ttt_game.winner == 'X' else player_o['name']
                    game_end_context = f"Game #{game_count} ended. {winner_name} wins!"
                    ttt_history.append({'name': 'Narrator', 'content': game_end_context})
                    print(f"🏆 Game #{game_count} ended. {winner_name} wins!")
                    print(f"Final board position:\n{ttt_game.print_board()}")
                    break  # End the overall game loop

            # After tic-tac-toe mode, update the history variable to save properly
            history = ttt_history

        elif args.hangman:
            print("🔤🎮 Playing Hangman Mode Activated 🎮🔤")
            print("=" * 80)

            # In hangman mode, we support 2 characters (guesser and host/assistant)
            if len(characters) != 2:
                print("❌ Hangman mode requires exactly 2 characters")
                sys.exit(1)

            # Assign roles: first character guesses, second character can provide hints/comments
            guesser = characters[0]
            supporter = characters[1]

            # Initialize hangman game
            hangman_game = HangmanGame()
            print("New word has been selected for guessing.")
            print(f"Word display: {hangman_game.get_current_display()}")
            print(f"Remaining incorrect guesses: {hangman_game.get_remaining_guesses()}")

            # Initialize conversation history with game context
            hangman_history = []

            # Add opening context to history
            opening_context = f"Starting a game of hangman. {guesser['name']} is guessing letters to find the secret word. {supporter['name']} can provide hints or commentary."
            hangman_history.append({'name': 'Narrator', 'content': opening_context})
            print(f"[TURN 1] Narrator: {opening_context}")

            # Hangman game loop
            turn = 1  # Main turn counter

            while not hangman_game.game_over:
                print(f"\n{'='*60}")
                print(f"[HANGMAN TURN {turn}] Current State:")
                print(f"Word: {hangman_game.get_current_display()}")
                print(f"Guessed letters: {hangman_game.get_guessed_letters()}")
                print(f"Incorrect guesses: {hangman_game.get_remaining_guesses()}")
                print(f"Hangman status:\n{hangman_game.get_hangman_status()}")
                print('='*60)

                # The guesser makes a guess
                current_char = guesser
                other_char = supporter

                print(f"\n[TURN {turn}] {current_char['name']} (making a letter guess)")

                try:
                    # Create game context for the turn, requesting explicit JSON output
                    hangman_context = f"""
CRITICAL HANGMAN GAME INSTRUCTIONS - YOU MUST FOLLOW THE EXACT JSON FORMAT BELOW OR THE GAME WILL STALL:

CURRENT GAME STATE:
- Current word: {hangman_game.get_current_display()}
- Guessed letters: {hangman_game.get_guessed_letters()}
- Remaining incorrect guesses: {hangman_game.get_remaining_guesses()}
- Hangman status:
{hangman_game.get_hangman_status()}

YOU MUST RESPOND IN EXACTLY THIS JSON FORMAT (NO EXCEPTIONS):
{{
  "dialogue": "Your dialogue and thought process about which letter to guess",
  "letter": "YOUR LETTER GUESS (single lowercase letter a-z only)",
  "reasoning": "Why you chose this letter"
}}

ABSOLUTELY CRITICAL REQUIREMENTS:
1. PROVIDE EXACTLY THE THREE FIELDS: "dialogue", "letter", "reasoning"
2. THE "letter" FIELD MUST CONTAIN EXCLUSIVELY A SINGLE LOWERCASE LETTER a-z
3. DO NOT WRAP LETTER IN ASTERISKS, QUOTES, OR NARRATIVE FORM
4. DO NOT SAY "I guess 'e'" OR "'e'" OR "**e**" OR "letter 'e'"
5. PUT ONLY THE RAW LETTER IN THE "letter" FIELD: 'a', 'b', 'c', etc.
6. DO NOT REPEAT LETTERS THAT ARE ALREADY IN {hangman_game.get_guessed_letters()}

EXAMPLE OF CORRECT FORMAT:
{{
  "dialogue": "I think the letter 'e' is likely to be in the word based on frequency analysis",
  "letter": "e",
  "reasoning": "E is the most common letter in English words"
}}

EXAMPLES OF INCORRECT FORMATS THAT WILL BE REJECTED:
- Just saying "e" in narrative text
- Writing "**e**" or "*e*" or "'e'" (without JSON structure)
- Saying "I guess 'e'" without proper JSON structure
- Providing only dialogue without letter field
- Using uppercase letters like "E" instead of lowercase "e"
- Including multiple letters like "ae" or words like "hello"
- Any format that does not match exact JSON specification above

VIOLATION OF JSON FORMAT RULES WILL RESULT IN:
- Turn immediately passed to next player
- No credit given for narrative text provided
- Your turn will be skipped until proper JSON format is used
- Up to 3 format violations allowed before automatic forfeit

Think carefully and respond in the EXACT JSON format specified above.
Make your letter guess now.
                    """.strip()

                    # Extract lorebook entries based on game context keywords
                    lorebook_entries = []
                    game_scenario = "Playing a game of hangman"
                    from character_loader import extract_lorebook_entries
                    # Search for keywords in game scenario that match character lorebooks
                    scenario_keywords = game_scenario.lower().split()
                    for char in characters:
                        entries = extract_lorebook_entries(char['raw_data'], hangman_history, max_entries=2)
                        # Filter entries by scenario relevance
                        relevant_entries = []
                        for entry in entries:
                            if any(keyword in entry.lower() for keyword in scenario_keywords if len(keyword) > 3):
                                relevant_entries.append(entry)
                        lorebook_entries.extend(relevant_entries[:1])  # Add at most 1 relevant entry per character

                    # Limit history to avoid token overflow while keeping recent context
                    limited_hangman_history = limit_history_window(hangman_history)

                    # Generate response with game context
                    resp = generate_response_adaptive(
                        current_char, other_char, limited_hangman_history, turn,
                        enable_environmental=not args.no_environmental,
                        similarity_threshold=args.similarity,
                        verbose=args.verbose,
                        scenario_context="Playing a game of hangman",
                        lorebook_entries=lorebook_entries if lorebook_entries else None
                    )

                    # Validate response
                    if not isinstance(resp, str):
                        resp = str(resp)

                    hangman_history.append({'name': current_char['name'], 'content': resp})
                    print(resp)

                    # Parse the JSON response
                    dialogue, letter_guess, reasoning = parse_hangman_json_response(resp, current_char['name'])

                    if dialogue:
                        print(f"💬 Dialogue: {dialogue}")

                    # Track consecutive failed moves to prevent infinite loops
                    if not hasattr(hangman_game, 'consecutive_failed_guesses'):
                        hangman_game.consecutive_failed_guesses = 0
                    if not hasattr(hangman_game, 'last_failed_guess'):
                        hangman_game.last_failed_guess = ''

                    # Attempt to make the guess if letter is provided
                    if letter_guess:
                        success = hangman_game.guess_letter(letter_guess)
                        if success:
                            print(f"✅ Letter '{letter_guess}' successfully guessed!")
                            print(f"New word display: {hangman_game.get_current_display()}")
                            if letter_guess in hangman_game.word:
                                print(f"Letter '{letter_guess}' was in the word!")
                            else:
                                print(f"Letter '{letter_guess}' was not in the word. Remaining incorrect guesses: {hangman_game.get_remaining_guesses()}")
                            turn += 1  # Guess was successful, increment turn
                            # Reset consecutive failed guesses
                            hangman_game.consecutive_failed_guesses = 0
                            hangman_game.last_failed_guess = ''
                        else:
                            print(f"❌ Guess failed - invalid letter: {letter_guess}")
                            # Add feedback to history
                            feedback = f"Your guess '{letter_guess}' was invalid. Please guess a single letter that hasn't been guessed yet."
                            hangman_history.append({'name': 'Referee', 'content': feedback})
                            # Increment consecutive failed guesses counter
                            hangman_game.consecutive_failed_guesses += 1
                            hangman_game.last_failed_guess = letter_guess
                            # If too many consecutive failures, force move to prevent infinite loops
                            if hangman_game.consecutive_failed_guesses >= 3:
                                print(f"⚠️  {current_char['name']} has failed to make a valid guess 3 times. Moving to next turn to prevent infinite loop.")
                                turn += 1  # Force increment to prevent infinite loops
                                hangman_game.consecutive_failed_guesses = 0
                                hangman_game.last_failed_guess = ''
                            # Otherwise, same player gets another chance (don't increment turn)
                    else:
                        print(f"⚠️  No letter provided for guess.")
                        # Add feedback to history
                        feedback = f"Please provide a valid letter to guess in your response."
                        hangman_history.append({'name': 'Referee', 'content': feedback})
                        # Increment consecutive failed guesses counter
                        hangman_game.consecutive_failed_guesses += 1
                        # If too many consecutive failures, force move to prevent infinite loops
                        if hangman_game.consecutive_failed_guesses >= 3:
                            print(f"⚠️  {current_char['name']} has failed to make a valid guess 3 times. Moving to next turn to prevent infinite loop.")
                            turn += 1  # Force increment to prevent infinite loops
                            hangman_game.consecutive_failed_guesses = 0
                            hangman_game.last_failed_guess = ''
                        # Otherwise, same player gets another chance (don't increment turn)

                    # Periodic save every 10 turns
                    if turn % 10 == 0:
                        save_conversation_periodic(hangman_history, output_file, turn)

                    # Add delay before next turn
                    if not hangman_game.game_over and turn < args.max_turns:
                        print(f"\n⏳ Waiting {args.delay} seconds...")
                        time.sleep(args.delay)
                    elif hangman_game.game_over:
                        break

                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user. Saving...")
                    break

                except Exception as e:
                    print(f"\n❌ Error on turn {turn}: {e}")
                    import traceback
                    traceback.print_exc()

                    # Generate fallback response
                    from response_generator import generate_emergency_response
                    fallback_resp = generate_emergency_response(current_char, other_char, hangman_history, {}, turn)
                    hangman_history.append({'name': current_char['name'], 'content': fallback_resp})
                    print(fallback_resp)

                    # Periodic save every 10 turns even in exception cases
                    if turn % 10 == 0:
                        save_conversation_periodic(hangman_history, output_file, turn)

                    turn += 1  # Increment turn anyway
                    continue  # Continue to next iteration

            # Handle game end
            if hangman_game.winner == 'Player':
                game_end_context = f"Game ended! {guesser['name']} guessed the word '{hangman_game.word}' correctly!"
                hangman_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"🎉 {guesser['name']} wins! The word was '{hangman_game.word}'!")
            else:
                game_end_context = f"Game ended! {guesser['name']} ran out of guesses. The word was '{hangman_game.word}'."
                hangman_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"💀 {guesser['name']} loses! The word was '{hangman_game.word}'.")

            # After hangman mode, update the history variable to save properly
            history = hangman_history

        elif args.twenty_one:
            print("🃏💰 Playing Twenty-One Mode Activated 💰🃏")
            print("=" * 80)

            # In twenty-one mode, we support 2 characters (player and dealer)
            if len(characters) != 2:
                print("❌ Twenty-One mode requires exactly 2 characters")
                sys.exit(1)

            # Assign roles: first character is player, second is dealer
            player_char = characters[0]
            dealer_char = characters[1]

            # Initialize twenty-one game
            twentyone_game = TwentyOneGame()
            print("Initial dealing completed.")
            print(f"Player's hand: {twentyone_game.get_player_hand_str()}")
            print(f"Player's score: {twentyone_game.get_player_score()}")
            print(f"Dealer's visible card: {twentyone_game.get_dealer_hand_str(hide_first=True)}")

            # Initialize conversation history with game context
            twentyone_history = []

            # Add opening context to history
            opening_context = f"Starting a game of twenty-one. {player_char['name']} is the player trying to get close to 21, and {dealer_char['name']} is the dealer."
            twentyone_history.append({'name': 'Narrator', 'content': opening_context})
            print(f"[TURN 1] Narrator: {opening_context}")

            # Twenty-One game loop
            turn = 1  # Main turn counter

            while not twentyone_game.game_over:
                print(f"\n{'='*60}")
                print(f"[TWENTY-ONE TURN {turn}] Current State:")
                print(f"Player's hand: {twentyone_game.get_player_hand_str()}")
                print(f"Player's score: {twentyone_game.get_player_score()}")
                print(f"Dealer's visible card: {twentyone_game.get_dealer_hand_str(hide_first=True)}")
                if twentyone_game.player_stands:
                    print(f"Dealer's full hand: {twentyone_game.get_dealer_hand_str(hide_first=False)}")
                    print(f"Dealer's score: {twentyone_game.get_dealer_score()}")
                print('='*60)

                # The player decides to hit or stand
                current_char = player_char
                other_char = dealer_char

                print(f"\n[TURN {turn}] {current_char['name']} (deciding to hit or stand)")

                try:
                    # Create game context for the turn, requesting explicit JSON output
                    twentyone_context = f"""
CRITICAL TWENTY-ONE GAME INSTRUCTIONS - YOU MUST FOLLOW THE EXACT JSON FORMAT BELOW OR THE GAME WILL STALL:

CURRENT GAME STATE:
- Your hand: {twentyone_game.get_player_hand_str()}
- Your score: {twentyone_game.get_player_score()}
- Dealer's visible card: {twentyone_game.get_dealer_hand_str(hide_first=True)}

YOU MUST RESPOND IN EXACTLY THIS JSON FORMAT (NO EXCEPTIONS):
{{
  "dialogue": "Your dialogue and thought process about your decision",
  "action": "YOUR ACTION ('hit' for another card or 'stand' to hold current hand)",
  "reasoning": "Why you made this decision"
}}

ABSOLUTELY CRITICAL REQUIREMENTS:
1. PROVIDE EXACTLY THE THREE FIELDS: "dialogue", "action", "reasoning"
2. THE "action" FIELD MUST CONTAIN EXCLUSIVELY 'hit' OR 'stand' (lowercase only)
3. DO NOT WRAP ACTION IN ASTERISKS, QUOTES, OR NARRATIVE FORM
4. DO NOT SAY "I decide to hit" OR "I will stand" OR "**hit**" OR "**stand**"
5. PUT ONLY THE RAW ACTION IN THE "action" FIELD: 'hit' or 'stand'
6. MAKE SURE YOUR CHOICE IS STRATEGICALLY SOUND GIVEN YOUR SCORE AND DEALER'S CARD

EXAMPLE OF CORRECT FORMAT:
{{
  "dialogue": "Given my current score of 17 and the dealer's visible card, I will proceed conservatively",
  "action": "stand",
  "reasoning": "Standing is the safest option given I'm close to 21 and the dealer's card suggests a possible strong hand"
}}

EXAMPLES OF INCORRECT FORMATS THAT WILL BE REJECTED:
- Just saying "hit" or "stand" in narrative text
- Writing "**hit**" or "*stand*" or "I'll hit" (without JSON structure)
- Saying "I choose to stand" without proper JSON format
- Providing only dialogue without action field
- Using uppercase like "HIT" or "STAND" instead of lowercase "hit"/"stand"
- Using variants like "hit me" or "stay" instead of exactly "hit"/"stand"
- Any format that does not match exact JSON specification above

VIOLATION OF JSON FORMAT RULES WILL RESULT IN:
- Turn immediately passed to next player
- No credit given for narrative text provided
- Your turn will be skipped until proper JSON format is used
- Up to 3 format violations allowed before automatic forfeit

Think carefully and respond in the EXACT JSON format specified above.
Make your strategic decision now.
                    """.strip()

                    # Extract lorebook entries based on game context keywords
                    lorebook_entries = []
                    game_scenario = "Playing a game of twenty-one"
                    from character_loader import extract_lorebook_entries
                    # Search for keywords in game scenario that match character lorebooks
                    scenario_keywords = game_scenario.lower().split()
                    for char in characters:
                        entries = extract_lorebook_entries(char['raw_data'], twentyone_history, max_entries=2)
                        # Filter entries by scenario relevance
                        relevant_entries = []
                        for entry in entries:
                            if any(keyword in entry.lower() for keyword in scenario_keywords if len(keyword) > 3):
                                relevant_entries.append(entry)
                        lorebook_entries.extend(relevant_entries[:1])  # Add at most 1 relevant entry per character

                    # Limit history to avoid token overflow while keeping recent context
                    limited_twentyone_history = limit_history_window(twentyone_history)

                    # Generate response with game context
                    resp = generate_response_adaptive(
                        current_char, other_char, limited_twentyone_history, turn,
                        enable_environmental=not args.no_environmental,
                        similarity_threshold=args.similarity,
                        verbose=args.verbose,
                        scenario_context="Playing a game of twenty-one",
                        lorebook_entries=lorebook_entries if lorebook_entries else None
                    )

                    # Validate response
                    if not isinstance(resp, str):
                        resp = str(resp)

                    twentyone_history.append({'name': current_char['name'], 'content': resp})
                    print(resp)

                    # Parse the JSON response
                    dialogue, action_choice, reasoning = parse_twentyone_json_response(resp, current_char['name'])

                    if dialogue:
                        print(f"💬 Dialogue: {dialogue}")

                    # Attempt to process the action if provided
                    if action_choice:
                        action_choice = action_choice.lower().strip()
                        if action_choice in ['hit', 'h']:
                            success = twentyone_game.player_hit()
                            if success:
                                print(f"✅ Player hits and draws a card!")
                                print(f"New hand: {twentyone_game.get_player_hand_str()}")
                                print(f"New score: {twentyone_game.get_player_score()}")

                                # Check if player busted
                                if twentyone_game.get_player_score() > 21:
                                    print(f"💀 Player busted with score {twentyone_game.get_player_score()}!")

                                turn += 1  # Action was successful, increment turn
                            else:
                                print(f"❌ Hit action failed - game may be over or invalid state.")
                        elif action_choice in ['stand', 's']:
                            twentyone_game.player_stand()
                            print(f"✅ Player stands with score {twentyone_game.get_player_score()}")
                            print(f"Dealer reveals full hand: {twentyone_game.get_dealer_hand_str(hide_first=False)}")
                            print(f"Dealer's score: {twentyone_game.get_dealer_score()}")
                            turn += 1  # Action was successful, increment turn
                        # Track consecutive failed moves to prevent infinite loops
                        if not hasattr(twentyone_game, 'consecutive_failed_actions'):
                            twentyone_game.consecutive_failed_actions = 0
                        if not hasattr(twentyone_game, 'last_failed_action'):
                            twentyone_game.last_failed_action = ''

                        if action_choice:
                            action_choice = action_choice.lower().strip()
                            if action_choice in ['hit', 'h']:
                                success = twentyone_game.player_hit()
                                if success:
                                    print(f"✅ Player hits and draws a card!")
                                    print(f"New hand: {twentyone_game.get_player_hand_str()}")
                                    print(f"New score: {twentyone_game.get_player_score()}")

                                    # Check if player busted
                                    if twentyone_game.get_player_score() > 21:
                                        print(f"💀 Player busted with score {twentyone_game.get_player_score()}!")

                                    turn += 1  # Action was successful, increment turn
                                    # Reset consecutive failed actions
                                    twentyone_game.consecutive_failed_actions = 0
                                    twentyone_game.last_failed_action = ''
                                else:
                                    print(f"❌ Hit action failed - game may be over or invalid state.")
                                    # Increment consecutive failed actions counter
                                    twentyone_game.consecutive_failed_actions += 1
                                    twentyone_game.last_failed_action = action_choice
                                    # If too many consecutive failures, force move to prevent infinite loops
                                    if twentyone_game.consecutive_failed_actions >= 3:
                                        print(f"⚠️  {current_char['name']} has failed to make a valid action 3 times. Moving to next turn to prevent infinite loop.")
                                        turn += 1  # Force increment to prevent infinite loops
                                        twentyone_game.consecutive_failed_actions = 0
                                        twentyone_game.last_failed_action = ''
                            elif action_choice in ['stand', 's']:
                                twentyone_game.player_stand()
                                print(f"✅ Player stands with score {twentyone_game.get_player_score()}")
                                print(f"Dealer reveals full hand: {twentyone_game.get_dealer_hand_str(hide_first=False)}")
                                print(f"Dealer's score: {twentyone_game.get_dealer_score()}")
                                turn += 1  # Action was successful, increment turn
                                # Reset consecutive failed actions
                                twentyone_game.consecutive_failed_actions = 0
                                twentyone_game.last_failed_action = ''
                            else:
                                print(f"❌ Invalid action: {action_choice}. Must be 'hit' or 'stand'.")
                                # Add feedback to history
                                feedback = f"Your action '{action_choice}' was invalid. Please choose 'hit' or 'stand'."
                                twentyone_history.append({'name': 'Referee', 'content': feedback})
                                # Increment consecutive failed actions counter
                                twentyone_game.consecutive_failed_actions += 1
                                twentyone_game.last_failed_action = action_choice
                                # If too many consecutive failures, force move to prevent infinite loops
                                if twentyone_game.consecutive_failed_actions >= 3:
                                    print(f"⚠️  {current_char['name']} has failed to make a valid action 3 times. Moving to next turn to prevent infinite loop.")
                                    turn += 1  # Force increment to prevent infinite loops
                                    twentyone_game.consecutive_failed_actions = 0
                                    twentyone_game.last_failed_action = ''
                                # Otherwise, same player gets another chance (don't increment turn)
                        else:
                            print(f"⚠️  No action provided.")
                            # Add feedback to history
                            feedback = f"Please provide a valid action ('hit' or 'stand') in your response."
                            twentyone_history.append({'name': 'Referee', 'content': feedback})
                            # Increment consecutive failed actions counter
                            twentyone_game.consecutive_failed_actions += 1
                            # If too many consecutive failures, force move to prevent infinite loops
                            if twentyone_game.consecutive_failed_actions >= 3:
                                print(f"⚠️  {current_char['name']} has failed to make a valid action 3 times. Moving to next turn to prevent infinite loop.")
                                turn += 1  # Force increment to prevent infinite loops
                                twentyone_game.consecutive_failed_actions = 0
                                twentyone_game.last_failed_action = ''
                            # Otherwise, same player gets another chance (don't increment turn)

                    # Periodic save every 10 turns
                    if turn % 10 == 0:
                        save_conversation_periodic(twentyone_history, output_file, turn)

                    # Add delay before next turn
                    if not twentyone_game.game_over and turn < args.max_turns:
                        print(f"\n⏳ Waiting {args.delay} seconds...")
                        time.sleep(args.delay)
                    elif twentyone_game.game_over:
                        break

                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user. Saving...")
                    break

                except Exception as e:
                    print(f"\n❌ Error on turn {turn}: {e}")
                    import traceback
                    traceback.print_exc()

                    # Generate fallback response
                    from response_generator import generate_emergency_response
                    fallback_resp = generate_emergency_response(current_char, other_char, twentyone_history, {}, turn)
                    twentyone_history.append({'name': current_char['name'], 'content': fallback_resp})
                    print(fallback_resp)

                    # Periodic save every 10 turns even in exception cases
                    if turn % 10 == 0:
                        save_conversation_periodic(twentyone_history, output_file, turn)

                    # DON'T increment turn - same player gets another chance due to error
                    continue  # Continue to next iteration

            # Handle game end
            if twentyone_game.winner == 'Player':
                game_end_context = f"Game ended! {player_char['name']} wins with a score of {twentyone_game.get_player_score()}!"
                twentyone_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"🎉 {player_char['name']} wins with {twentyone_game.get_player_score()}!")
            elif twentyone_game.winner == 'Dealer':
                dealer_score = twentyone_game.get_dealer_score() if twentyone_game.dealer_stands else "bust"
                game_end_context = f"Game ended! {dealer_char['name']} wins. {player_char['name']} busts or has a lower score."
                twentyone_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"💀 {dealer_char['name']} wins! {player_char['name']} loses with {twentyone_game.get_player_score()}.")
            else:  # Draw
                game_end_context = f"Game ended in a draw! Both players have the same score."
                twentyone_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"⚖️ Game ended in a draw! Both scores: Player {twentyone_game.get_player_score()}, Dealer {twentyone_game.get_dealer_score()}.")

            # After twenty-one mode, update the history variable to save properly
            history = twentyone_history

        elif args.number_guessing:
            print("🧩🔢 Playing Number Guessing Mode Activated 🔢🧩")
            print("=" * 80)

            # In number guessing mode, we support 2 characters
            if len(characters) != 2:
                print("❌ Number guessing mode requires exactly 2 characters")
                sys.exit(1)

            # Assign players
            guesser = characters[0]
            challenger = characters[1]

            # Initialize number guessing game
            number_game = NumberGuessingGame(min_num=1, max_num=100)
            print(f"Secret number has been selected between {number_game.min_num} and {number_game.max_num}.")
            print(f"Maximum allowed attempts: {number_game.max_attempts}")

            # Initialize conversation history with game context
            number_history = []

            # Add opening context to history
            opening_context = f"Starting a number guessing game. {guesser['name']} is guessing the number (1-{number_game.max_num}) while {challenger['name']} provides feedback."
            number_history.append({'name': 'Narrator', 'content': opening_context})
            print(f"[TURN 1] Narrator: {opening_context}")

            # Number guessing game loop
            game_count = 1
            turn = 1  # Main turn counter

            while not number_game.game_over:
                print(f"\n{'='*60}")
                print(f"[NUMBER GUESSING GAME #{game_count}] Current State:")
                print(f"Number range: {number_game.min_num} - {number_game.max_num}")
                print(f"Remaining attempts: {number_game.get_remaining_guesses()}")
                print(f"Previous guesses: {number_game.get_guessed_letters() if hasattr(number_game, 'get_guessed_letters') else 'None yet'}")
                print(f"Game status: {number_game.get_game_status()}")
                print('='*60)

                # The guesser makes a guess
                current_char = guesser
                other_char = challenger

                print(f"\n[TURN {turn}] {current_char['name']} (making a number guess)")

                try:
                    # Create game context for the turn, requesting explicit JSON output
                    number_context = f"""
CRITICAL NUMBER GUESSING GAME INSTRUCTIONS - YOU MUST FOLLOW THE EXACT JSON FORMAT BELOW OR THE GAME WILL STALL:

CURRENT GAME STATE:
- Number to guess is between {number_game.min_num} and {number_game.max_num}
- Previous guesses: {number_game.get_guessed_letters() if hasattr(number_game, 'get_guessed_letters') else 'None yet'}
- Remaining attempts: {number_game.get_remaining_guesses()}

YOU MUST RESPOND IN EXACTLY THIS JSON FORMAT (NO EXCEPTIONS):
{{
  "dialogue": "Your dialogue and thought process about which number to guess",
  "number": "YOUR NUMBER GUESS (integer between {number_game.min_num} and {number_game.max_num} only)",
  "strategy": "Why you chose this number"
}}

ABSOLUTELY CRITICAL REQUIREMENTS:
1. PROVIDE EXACTLY THE THREE FIELDS: "dialogue", "number", "strategy"
2. THE "number" FIELD MUST CONTAIN EXCLUSIVELY A VALID INTEGER IN THE RANGE {number_game.min_num}-{number_game.max_num}
3. DO NOT WRAP NUMBER IN ASTERISKS, QUOTES, OR NARRATIVE FORM
4. DO NOT SAY "I guess 42" OR "42" (without quotes) OR "**42**" OR "number 42"
5. PUT ONLY THE RAW NUMBER IN THE "number" FIELD: '42', '50', '17', etc. (as digits)
6. DO NOT REPEAT ANY NUMBER FROM {number_game.get_guessed_letters() if hasattr(number_game, 'get_guessed_letters') else '[]'}

EXAMPLE OF CORRECT FORMAT:
{{
  "dialogue": "Based on the remaining range and previous guesses, I believe the middle value offers the best information",
  "number": "50",
  "strategy": "Binary search approach to eliminate half the possibilities"
}}

EXAMPLES OF INCORRECT FORMATS THAT WILL BE REJECTED:
- Just saying "50" in narrative text
- Writing "**50**" or "*50*" or "I guess 50" (without JSON structure)
- Saying "My guess is 50" without proper JSON format
- Providing only dialogue without number field
- Using non-numeric guesses like "fifty" instead of "50"
- Including multiple numbers like "40-50" or ranges like "between 40 and 50"
- Any format that does not match exact JSON specification above

VIOLATION OF JSON FORMAT RULES WILL RESULT IN:
- Turn immediately passed to next player
- No credit given for narrative text provided
- Your turn will be skipped until proper JSON format is used
- Up to 3 format violations allowed before automatic forfeit

Think carefully and respond in the EXACT JSON format specified above.
Make your numerical guess now.
                    """.strip()

                    # Limit history to avoid token overflow while keeping recent context
                    limited_number_history = limit_history_window(number_history)

                    # Generate response with game context
                    resp = generate_response_adaptive(
                        current_char, other_char, limited_number_history, turn,
                        enable_environmental=not args.no_environmental,
                        similarity_threshold=args.similarity,
                        verbose=args.verbose,
                        scenario_context="Playing a number guessing game"
                    )

                    # Validate response
                    if not isinstance(resp, str):
                        resp = str(resp)

                    number_history.append({'name': current_char['name'], 'content': resp})
                    print(resp)

                    # Parse the JSON response specifically for number guessing
                    dialogue, number_guess, strategy = parse_number_guessing_json_response(resp, current_char['name'])

                    if dialogue:
                        print(f"💬 Dialogue: {dialogue}")

                    # Attempt to make the guess if number is provided
                    if number_guess:
                        try:
                            number_value = int(number_guess)
                            success = number_game.make_guess(number_value)
                            if success:
                                print(f"✅ Number '{number_value}' successfully guessed!")
                                print(f"Feedback: {number_game.make_guess(number_value)}")
                                if number_game.winner:
                                    print(f"Game over! Winner: {number_game.winner}")
                                    break
                                turn += 1  # Guess was successful, increment turn
                            else:
                                print(f"❌ Invalid guess: {number_guess}")
                                # Add feedback to history
                                feedback = f"Your number '{number_guess}' was invalid. Please guess a number between {number_game.min_num} and {number_game.max_num}."
                                number_history.append({'name': 'Referee', 'content': feedback})
                                turn += 1  # Increment turn anyway
                        except ValueError:
                            print(f"❌ Could not parse as number: {number_guess}")
                            # Add feedback to history
                            feedback = f"Your choice '{number_guess}' was not a valid number. Please provide a numeric guess."
                            number_history.append({'name': 'Referee', 'content': feedback})
                            turn += 1  # Increment turn anyway
                    else:
                        print(f"⚠️  No number provided for guess.")
                        # Add feedback to history
                        feedback = f"Please provide a valid number to guess in your response."
                        number_history.append({'name': 'Referee', 'content': feedback})
                        turn += 1  # Increment turn anyway

                    # Delay before next turn
                    if not number_game.game_over and turn < args.max_turns:
                        print(f"\n⏳ Waiting {args.delay} seconds...")
                        time.sleep(args.delay)
                    elif number_game.game_over:
                        break

                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user. Saving...")
                    break

                except Exception as e:
                    print(f"\n❌ Error on turn {turn}: {e}")
                    import traceback
                    traceback.print_exc()

                    # Generate fallback response
                    from response_generator import generate_emergency_response
                    fallback_resp = generate_emergency_response(current_char, other_char, number_history, {}, turn)
                    number_history.append({'name': current_char['name'], 'content': fallback_resp})
                    print(fallback_resp)
                    # DON'T increment turn - same player gets another chance due to error
                    continue  # Continue to next iteration

            # Handle game end
            if number_game.winner == 'Player':
                game_end_context = f"Game #{game_count} ended. {guesser['name']} guessed the number '{number_game.secret_number}' correctly!"
                number_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"🎉 {guesser['name']} wins! The number was '{number_game.secret_number}'!")
            else:
                game_end_context = f"Game #{game_count} ended. {guesser['name']} ran out of attempts. The number was '{number_game.secret_number}'."
                number_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"💀 {guesser['name']} loses! The number was '{number_game.secret_number}'.")

            # After number guessing mode, update the history variable to save properly
            history = number_history

        elif args.word_association:
            print("💭🔗 Playing Word Association Mode Activated 🔗💭")
            print("=" * 80)

            # In word association mode, we support 2 characters
            if len(characters) != 2:
                print("❌ Word association mode requires exactly 2 characters")
                sys.exit(1)

            # Assign players
            player1 = characters[0]
            player2 = characters[1]

            # Initialize word association game
            word_game = WordAssociationGame()
            print(f"Initial word chain: {word_game.get_word_chain()}")
            print(f"Current player: {word_game.get_current_player()}")

            # Initialize conversation history with game context
            word_history = []

            # Add opening context to history
            opening_context = f"Starting a word association game. {player1['name']} and {player2['name']} will take turns saying words related to the previous word."
            word_history.append({'name': 'Narrator', 'content': opening_context})
            print(f"[TURN 1] Narrator: {opening_context}")

            # Word association game loop
            game_count = 1
            turn = 1  # Main turn counter

            while not word_game.game_over:
                print(f"\n{'='*60}")
                print(f"[WORD ASSOCIATION GAME #{game_count}] Current State:")
                print(f"Word chain: {word_game.get_word_chain()}")
                print(f"Last word: {word_game.get_last_word()}")
                print(f"Current player: {word_game.get_current_player()}")
                print(f"Game status:\n{word_game.get_game_status()}")
                print('='*60)

                # Determine who is making the word association
                current_char = player1 if word_game.current_player == 'Player1' else player2
                other_char = player2 if word_game.current_player == 'Player1' else player1

                print(f"\n[TURN {turn}] {current_char['name']} (saying a related word)")

                try:
                    # Create game context for the turn, requesting explicit JSON output
                    word_context = f"""
CRITICAL WORD ASSOCIATION GAME INSTRUCTIONS - YOU MUST FOLLOW THE EXACT JSON FORMAT BELOW OR THE GAME WILL STALL:

CURRENT GAME STATE:
- Current word chain: {word_game.get_word_chain()}
- Last word: {word_game.get_last_word()}
- Current player: {word_game.get_current_player()}

YOU MUST RESPOND IN EXACTLY THIS JSON FORMAT (NO EXCEPTIONS):
{{
  "dialogue": "Your dialogue and thought process about your word choice",
  "word": "YOUR SINGULAR RELATED WORD (single lowercase word only)",
  "connection": "How your word connects to the previous word"
}}

ABSOLUTELY CRITICAL REQUIREMENTS:
1. PROVIDE EXACTLY THE THREE FIELDS: "dialogue", "word", "connection"
2. THE "word" FIELD MUST CONTAIN EXCLUSIVELY ONE SINGLE WORD (not a phrase or sentence)
3. DO NOT WRAP WORD IN ASTERISKS, QUOTES, OR NARRATIVE FORM
4. DO NOT SAY "I choose the word 'apple'" OR "**apple**" OR "'apple'" OR "the word 'apple'"
5. PUT ONLY THE RAW LOWERCASE WORD IN THE "word" FIELD: 'apple', 'happy', 'running', etc.
6. ENSURE YOUR WORD HAS A CLEAR SEMANTIC CONNECTION TO "{word_game.get_last_word() or 'the concept'}"

EXAMPLE OF CORRECT FORMAT:
{{
  "dialogue": "Considering the previous word's meaning and related concepts, I select this word",
  "word": "ocean",
  "connection": "It relates to water, waves, and sea which connects to the previous concept"
}}

EXAMPLES OF INCORRECT FORMATS THAT WILL BE REJECTED:
- Just saying "ocean" in narrative text without JSON structure
- Writing "**ocean**" or "*ocean*" or "the word 'ocean'" (without proper JSON)
- Saying "I choose the word ocean" without proper JSON structure
- Providing only dialogue without word field
- Using capitalized words like "Ocean" instead of lowercase "ocean"
- Including phrases or multiple words like "blue ocean" or "ocean water"
- Using numbers or special characters in the word field
- Any format that does not match exact JSON specification above

VIOLATION OF JSON FORMAT RULES WILL RESULT IN:
- Turn immediately passed to next player
- No credit given for narrative text provided
- Your turn will be skipped until proper JSON format is used
- Up to 3 format violations allowed before automatic forfeit

Think carefully and respond in the EXACT JSON format specified above.
Make your word association now.
                    """.strip()

                    # Limit history to avoid token overflow while keeping recent context
                    limited_word_history = limit_history_window(word_history)

                    # Generate response with game context
                    resp = generate_response_adaptive(
                        current_char, other_char, limited_word_history, turn,
                        enable_environmental=not args.no_environmental,
                        similarity_threshold=args.similarity,
                        verbose=args.verbose,
                        scenario_context="Playing a word association game"
                    )

                    # Validate response
                    if not isinstance(resp, str):
                        resp = str(resp)

                    word_history.append({'name': current_char['name'], 'content': resp})
                    print(resp)

                    # Parse the JSON response specifically for word association
                    dialogue, word_choice, connection = parse_game_json_response(resp, current_char['name'])

                    if dialogue:
                        print(f"💬 Dialogue: {dialogue}")

                    # Submit the word if provided
                    if word_choice:
                        success = word_game.submit_word(word_choice, word_game.get_current_player())
                        if success:
                            print(f"✅ Word '{word_choice}' successfully submitted!")
                            print(f"New word chain: {word_game.get_word_chain()}")
                            if word_game.winner:
                                print(f"Game over! Winner: {word_game.winner}")
                                break
                            turn += 1  # Word was successful, increment turn
                        else:
                            print(f"❌ Invalid word submission: {word_choice}")
                            # Add feedback to history
                            feedback = f"Your word '{word_choice}' was invalid. Please choose a valid word related to the previous word and avoid repeating words."
                            word_history.append({'name': 'Referee', 'content': feedback})
                            turn += 1  # Increment turn anyway
                    else:
                        print(f"⚠️  No word provided.")
                        # Add feedback to history
                        feedback = f"Please provide a valid word related to the previous word in your response."
                        word_history.append({'name': 'Referee', 'content': feedback})
                        turn += 1  # Increment turn anyway

                    # Delay before next turn
                    if not word_game.game_over and turn < args.max_turns:
                        print(f"\n⏳ Waiting {args.delay} seconds...")
                        time.sleep(args.delay)
                    elif word_game.game_over:
                        break

                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user. Saving...")
                    break

                except Exception as e:
                    print(f"\n❌ Error on turn {turn}: {e}")
                    import traceback
                    traceback.print_exc()

                    # Generate fallback response
                    from response_generator import generate_emergency_response
                    fallback_resp = generate_emergency_response(current_char, other_char, word_history, {}, turn)
                    word_history.append({'name': current_char['name'], 'content': fallback_resp})
                    print(fallback_resp)
                    # DON'T increment turn - same player gets another chance due to error
                    continue  # Continue to next iteration

            # Handle game end
            if word_game.winner and word_game.winner != 'draw':
                winner_name = player1['name'] if word_game.winner == 'Player1' else player2['name']
                game_end_context = f"Game #{game_count} ended. {winner_name} wins after {len(word_game.word_chain)} words!"
                word_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"🎉 {winner_name} wins after {len(word_game.word_chain)} words in the chain!")
            elif word_game.winner == 'draw':
                game_end_context = f"Game #{game_count} ended in a draw after reaching maximum chain length."
                word_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"⚖️ Game #{game_count} ended in a draw after reaching the maximum chain length.")
            else:
                game_end_context = f"Game #{game_count} ended."
                word_history.append({'name': 'Narrator', 'content': game_end_context})
                print(f"🏁 Game #{game_count} ended.")

            # After word association mode, update the history variable to save properly
            history = word_history

        elif args.connect_four:
            print("🔴🟡🔵 Playing Connect-Four Mode Activated 🔵🟡🔴")
            print("=" * 80)

            # In connect-four mode, we only support 2 characters
            if len(characters) != 2:
                print("❌ Connect-four mode requires exactly 2 characters")
                sys.exit(1)

            # Assign players to red and yellow
            player1 = characters[0]
            player2 = characters[1]
            player1['cf_symbol'] = 'R'  # Red disc
            player2['cf_symbol'] = 'Y'  # Yellow disc

            # Initialize connect-four game
            connect_four_game = ConnectFourGame()
            print("Initial Connect-Four Board:")
            print(connect_four_game.print_board())

            # Initialize conversation history with game context
            connect_four_history = []

            # Add opening context to history
            opening_context = f"Starting a game of connect-four. {player_red['name']} is playing as red (R), and {player_yellow['name']} is playing as yellow (Y)."
            connect_four_history.append({'name': 'Narrator', 'content': opening_context})
            print(f"[TURN 1] Narrator: {opening_context}")

            # Connect-four game loop
            game_count = 1
            turn = 1  # Main turn counter

            while True:
                print(f"\n{'='*80}")
                print(f"CONNECT-FOUR GAME #{game_count}")
                print('='*80)

                # Reset game for each game (in case of draws)
                connect_four_game = ConnectFourGame()

                # Add game start context to history
                game_start_context = f"Starting connect-four game #{game_count}. {player_red['name']} (red) vs {player_yellow['name']} (yellow)."
                connect_four_history.append({'name': 'Narrator', 'content': game_start_context})
                print(f"[TURN {turn}] Narrator: {game_start_context}")
                turn += 1

                # Start the connect-four game loop
                cf_turn = 1
                while not connect_four_game.game_over:
                    print(f"\n{'='*60}")
                    print(f"[CONNECT-FOUR TURN {cf_turn}] Board Position:")
                    print(connect_four_game.print_board())
                    print(f"Current Player: {connect_four_game.current_player}")
                    print('='*60)

                    # Determine who is making the move
                    if connect_four_game.current_player == 'red':
                        current_char = player1
                        other_char = player2
                    else:
                        current_char = player2
                        other_char = player1

                    print(f"\n[TURN {turn}] {current_char['name']} (playing as {connect_four_game.current_player})")

                    # Track consecutive failed moves to prevent infinite loops
                    if not hasattr(connect_four_game, 'consecutive_failed_moves'):
                        connect_four_game.consecutive_failed_moves = {'R': 0, 'Y': 0}
                    if not hasattr(connect_four_game, 'last_failed_move'):
                        connect_four_game.last_failed_move = {'R': '', 'Y': ''}

                    try:
                        # Create game context for the turn, requesting explicit JSON output
                        cf_context = f"""
CRITICAL CONNECT-FOUR GAME INSTRUCTIONS - YOU MUST FOLLOW THE EXACT JSON FORMAT BELOW OR THE GAME WILL STALL:

CURRENT BOARD POSITION:
{connect_four_game.print_board()}

GAME STATE:
- Current Player: {connect_four_game.current_player}
- Your Symbol: {current_char['cf_symbol']}

YOU MUST RESPOND IN EXACTLY THIS JSON FORMAT (NO EXCEPTIONS):
{{
  "dialogue": "Your dialogue and thought process about which column to choose",
  "column": "YOUR COLUMN CHOICE (single digit 0-6 only)",
  "strategy": "Why you chose this column"
}}

ABSOLUTELY CRITICAL REQUIREMENTS:
1. PROVIDE EXACTLY THE THREE FIELDS: "dialogue", "column", "strategy"
2. THE "column" FIELD MUST CONTAIN EXCLUSIVELY A SINGLE DIGIT: 0, 1, 2, 3, 4, 5, OR 6
3. DO NOT WRAP COLUMN IN ASTERISKS, QUOTES, OR NARRATIVE FORM
4. DO NOT SAY "I choose column 3" OR "column 3" OR "**3**" OR "number 3"
5. PUT ONLY THE RAW DIGIT IN THE "column" FIELD: '0', '1', '2', '3', '4', '5', OR '6'
6. VERIFY THE COLUMN HAS AVAILABLE SPACE (IS NOT FULL)

EXAMPLE OF CORRECT FORMAT:
{{
  "dialogue": "I am analyzing the board state to determine optimal positioning",
  "column": "3",
  "strategy": "This column provides the best opportunity for forming a four-in-a-row"
}}

EXAMPLES OF INCORRECT FORMATS THAT WILL BE REJECTED:
- Just saying "3" in narrative text without JSON structure
- Writing "**3**" or "*3*" or "column 3" (without proper JSON format)
- Saying "I choose column 3" without proper JSON structure
- Providing only dialogue without column field
- Using non-numeric columns like "three" instead of "3"
- Using values outside the range 0-6 like "7", "8", etc.
- Including multiple values or complex expressions like "[3]" or "column 3 please"
- Any format that does not match exact JSON specification above

VIOLATION OF JSON FORMAT RULES WILL RESULT IN:
- Turn immediately passed to opponent
- No credit given for narrative text provided
- Your turn will be skipped until proper JSON format is used
- Up to 3 format violations allowed before automatic forfeit

Think carefully and respond in the EXACT JSON format specified above.
Think through your strategy before responding.
                        """.strip()

                        # Extract lorebook entries based on game context keywords
                        lorebook_entries = []
                        game_scenario = "Playing a game of connect-four"
                        from character_loader import extract_lorebook_entries
                        # Search for keywords in game scenario that match character lorebooks
                        scenario_keywords = game_scenario.lower().split()
                        for char in [current_char]:
                            entries = extract_lorebook_entries(char['raw_data'], connect_four_history, max_entries=2)
                            # Filter entries by scenario relevance
                            relevant_entries = []
                            for entry in entries:
                                if any(keyword in entry.lower() for keyword in scenario_keywords if len(keyword) > 3):
                                    relevant_entries.append(entry)
                            lorebook_entries.extend(relevant_entries[:1])  # Add at most 1 relevant entry

                        # Generate response with game context
                        resp = generate_response_adaptive(
                            current_char, other_char, connect_four_history, turn,
                            enable_environmental=not args.no_environmental,
                            similarity_threshold=args.similarity,
                            verbose=args.verbose,
                            scenario_context="Playing a game of connect-four",
                            lorebook_entries=lorebook_entries if lorebook_entries else None
                        )

                        # Validate response
                        if not isinstance(resp, str):
                            resp = str(resp)

                        connect_four_history.append({'name': current_char['name'], 'content': resp})
                        print(resp)

                        # Parse the JSON response specifically for connect-four
                        import json
                        import re

                        # First and ONLY try: Look for properly formatted JSON with required fields
                        json_pattern = r'(\{[^{}]*("dialogue"|\'dialogue\')[^{}]*:[^{}]*["\'][^{}]*["\'][^{}]*,?[^{}]*("column"|\'column\')[^{}]*:[^{}]*["\'][^{}]*["\'][^{}]*\})'
                        matches = re.findall(json_pattern, resp, re.DOTALL)

                        dialogue = ""
                        column_choice = None
                        strategy = ""

                        for match_tuple in matches:
                            try:
                                json_clean = match_tuple[0].strip()
                                parsed = json.loads(json_clean)

                                dialogue = parsed.get('dialogue', '')
                                column_choice = parsed.get('column', '')
                                strategy = parsed.get('strategy', '')

                                # Validate that BOTH required fields are present and non-empty
                                if dialogue.strip() and column_choice is not None:
                                    break
                            except json.JSONDecodeError:
                                continue  # Try the next match

                        if dialogue:
                            print(f"💬 Dialogue: {dialogue}")

                        # Track if a real move was successfully processed in this turn
                        move_was_processed = False

                        # Determine the current player's symbol for tracking
                        current_symbol = current_char.get('cf_symbol', 'R' if connect_four_game.current_player == 'red' else 'Y')

                        # Attempt to make the move if column is provided in proper JSON format
                        if column_choice:
                            try:
                                # Try to extract the column number from the choice
                                col_num = None
                                if isinstance(column_choice, str):
                                    # Handle string representations with strict validation
                                    col_choice_clean = column_choice.strip()
                                    # Only accept pure numeric column values or properly quoted numbers
                                    import re
                                    single_digit_match = re.match(r'^["\']?([0-6])["\']?$', col_choice_clean)
                                    if single_digit_match:
                                        col_num = int(single_digit_match.group(1))
                                    else:
                                        # If it's not a single digit format, it's invalid
                                        print(f"❌ Column choice format invalid: {column_choice}")
                                        # Add feedback to history
                                        feedback = f"Your column choice '{column_choice}' was invalid format. Please provide EXACTLY a single digit (0-6) in the 'column' field of your JSON."
                                        connect_four_history.append({'name': 'Referee', 'content': feedback})
                                        # Increment consecutive failed moves counter
                                        connect_four_game.consecutive_failed_moves[current_symbol] += 1
                                        connect_four_game.last_failed_move[current_symbol] = column_choice
                                        # If too many consecutive failures, force move to other player to prevent infinite loops
                                        if connect_four_game.consecutive_failed_moves[current_symbol] >= 2:
                                            print(f"⚠️  {current_char['name']} has failed to make a valid move {connect_four_game.consecutive_failed_moves[current_symbol]} times. Moving to next player to prevent infinite loop.")
                                            # ALSO switch the current player in the connect-four game object to ensure the game state progresses correctly
                                            connect_four_game.current_player = 'yellow' if connect_four_game.current_player == 'red' else 'red'
                                            turn += 1  # Force increment to prevent infinite loops
                                            connect_four_game.consecutive_failed_moves[current_symbol] = 0
                                            connect_four_game.last_failed_move[current_symbol] = ''
                                            move_was_processed = True
                                        # Otherwise, same player gets another chance (don't increment turn)
                                elif isinstance(column_choice, int) and 0 <= column_choice <= 6:
                                    col_num = column_choice
                                else:
                                    print(f"❌ Invalid column choice: {column_choice}")
                                    # Add feedback to history
                                    feedback = f"Your column choice '{column_choice}' was invalid. Please select a valid column number (0-6)."
                                    connect_four_history.append({'name': 'Referee', 'content': feedback})
                                    # Increment consecutive failed moves counter
                                    connect_four_game.consecutive_failed_moves[current_symbol] += 1
                                    connect_four_game.last_failed_move[current_symbol] = str(column_choice)  # Convert to string for tracking
                                    # If too many consecutive failures, force move to other player to prevent infinite loops
                                    if connect_four_game.consecutive_failed_moves[current_symbol] >= 2:
                                        print(f"⚠️  {current_char['name']} has failed to make a valid move {connect_four_game.consecutive_failed_moves[current_symbol]} times. Moving to next player to prevent infinite loop.")
                                        # ALSO switch the current player in the connect-four game object to ensure the game state progresses correctly
                                        connect_four_game.current_player = 'yellow' if connect_four_game.current_player == 'red' else 'red'
                                        turn += 1  # Force increment to prevent infinite loops
                                        connect_four_game.consecutive_failed_moves[current_symbol] = 0
                                        connect_four_game.last_failed_move[current_symbol] = ''
                                        move_was_processed = True
                                    # Otherwise, same player gets another chance (don't increment turn)

                                if col_num is not None and 0 <= col_num <= 6:
                                    success = connect_four_game.make_move(col_num)
                                    if success:
                                        print(f"✅ Disc dropped successfully in column {col_num}!")
                                        print(f"New board:\n{connect_four_game.print_board()}")
                                        if connect_four_game.winner:
                                            print(f"Game over! Winner: {connect_four_game.winner}")
                                        turn += 1  # Move was successful, increment turn
                                        cf_turn += 1  # Also increment cf turn
                                        # Reset consecutive failed moves for this player
                                        connect_four_game.consecutive_failed_moves[current_symbol] = 0
                                        connect_four_game.last_failed_move[current_symbol] = ''
                                        move_was_processed = True
                                    else:
                                        print(f"❌ Move failed - invalid column: {col_num} (may be full or out of bounds)")
                                        # Add feedback to history
                                        feedback = f"Your choice of column {col_num} was invalid. Please select a valid column (0-6) that has space."
                                        connect_four_history.append({'name': 'Referee', 'content': feedback})
                                        # Increment consecutive failed moves counter
                                        connect_four_game.consecutive_failed_moves[current_symbol] += 1
                                        connect_four_game.last_failed_move[current_symbol] = str(col_num)
                                        # If too many consecutive failures, force move to other player to prevent infinite loops
                                        if connect_four_game.consecutive_failed_moves[current_symbol] >= 2:
                                            print(f"⚠️  {current_char['name']} has failed to make a valid move {connect_four_game.consecutive_failed_moves[current_symbol]} times. Moving to next player to prevent infinite loop.")
                                            # ALSO switch the current player in the connect-four game object to ensure the game state progresses correctly
                                            connect_four_game.current_player = 'yellow' if connect_four_game.current_player == 'red' else 'red'
                                            turn += 1  # Force increment to prevent infinite loops
                                            connect_four_game.consecutive_failed_moves[current_symbol] = 0
                                            connect_four_game.last_failed_move[current_symbol] = ''
                                        # Otherwise, same player gets another chance (don't increment turn)
                                else:
                                    print(f"❌ Invalid column choice: {column_choice}")
                                    # Add feedback to history
                                    feedback = f"Your column choice '{column_choice}' was invalid. Please select a valid column number (0-6)."
                                    connect_four_history.append({'name': 'Referee', 'content': feedback})
                                    # Increment consecutive failed moves counter
                                    connect_four_game.consecutive_failed_moves[current_symbol] += 1
                                    connect_four_game.last_failed_move[current_symbol] = str(column_choice)
                                    # If too many consecutive failures, force move to other player to prevent infinite loops
                                    if connect_four_game.consecutive_failed_moves[current_symbol] >= 2:
                                        print(f"⚠️  {current_char['name']} has failed to make a valid move {connect_four_game.consecutive_failed_moves[current_symbol]} times. Moving to next player to prevent infinite loop.")
                                        # ALSO switch the current player in the connect-four game object to ensure the game state progresses correctly
                                        connect_four_game.current_player = 'yellow' if connect_four_game.current_player == 'red' else 'red'
                                        turn += 1  # Force increment to prevent infinite loops
                                        connect_four_game.consecutive_failed_moves[current_symbol] = 0
                                        connect_four_game.last_failed_move[current_symbol] = ''
                                    # Otherwise, same player gets another chance (don't increment turn)
                            except ValueError as e:
                                print(f"❌ Could not parse column choice: {column_choice}, error: {e}")
                                # Add feedback to history
                                feedback = f"I couldn't parse the column '{column_choice}'. Please provide a valid column number (0-6). REMEMBER: You must respond in proper JSON format with both 'dialogue' and 'column' fields."
                                connect_four_history.append({'name': 'Referee', 'content': feedback})
                                # Increment consecutive failed moves counter
                                connect_four_game.consecutive_failed_moves[current_symbol] += 1
                                connect_four_game.last_failed_move[current_symbol] = str(column_choice)
                                # If too many consecutive failures, force move to other player to prevent infinite loops
                                if connect_four_game.consecutive_failed_moves[current_symbol] >= 2:
                                    print(f"⚠️  {current_char['name']} has failed to make a valid move {connect_four_game.consecutive_failed_moves[current_symbol]} times. Moving to next player to prevent infinite loop.")
                                    # ALSO switch the current player in the connect-four game object to ensure the game state progresses correctly
                                    connect_four_game.current_player = 'yellow' if connect_four_game.current_player == 'red' else 'red'
                                    turn += 1  # Force increment to prevent infinite loops
                                    connect_four_game.consecutive_failed_moves[current_symbol] = 0
                                    connect_four_game.last_failed_move[current_symbol] = ''
                                # Otherwise, same player gets another chance (don't increment turn)
                        else:
                            print(f"⚠️  No column provided in required JSON format. Current player: {connect_four_game.current_player}")
                            # Add feedback to history
                            feedback = f"You MUST provide a valid column number in the proper JSON format: {{\"dialogue\": \"your thoughts\", \"column\": \"3\", \"strategy\": \"your reasoning\"}}. Your response must include both 'dialogue' and 'column' fields in JSON format."
                            connect_four_history.append({'name': 'Referee', 'content': feedback})
                            # Increment consecutive failed moves counter
                            connect_four_game.consecutive_failed_moves[current_symbol] += 1
                            connect_four_game.last_failed_move[current_symbol] = ''
                            # If too many consecutive failures, force move to other player to prevent infinite loops
                            if connect_four_game.consecutive_failed_moves[current_symbol] >= 2:
                                print(f"⚠️  {current_char['name']} has failed to make a valid move {connect_four_game.consecutive_failed_moves[current_symbol]} times. Moving to next player to prevent infinite loop.")
                                # ALSO switch the current player in the connect-four game object to ensure the game state progresses correctly
                                connect_four_game.current_player = 'yellow' if connect_four_game.current_player == 'red' else 'red'
                                turn += 1  # Force increment to prevent infinite loops
                                connect_four_game.consecutive_failed_moves[current_symbol] = 0
                                connect_four_game.last_failed_move[current_symbol] = ''
                            # Otherwise, same player gets another chance (don't increment turn)

                        # CRITICAL SAFEGUARD: If no real move was processed, we MUST force turn advancement anyway
                        # This prevents the game from getting stuck due to parsing issues or other problems
                        if not move_was_processed:
                            print(f"⚠️  CRITICAL: No move was processed. Forcing turn advancement to prevent infinite loop.")
                            # Switch the current player in the connect-four game for the next iteration
                            connect_four_game.current_player = 'yellow' if connect_four_game.current_player == 'red' else 'red'
                            turn += 1  # Force increment to break potential infinite loop
                            move_was_processed = True

                        # Delay before next turn
                        if not connect_four_game.game_over and turn < args.max_turns:
                            print(f"\n⏳ Waiting {args.delay} seconds...")
                            time.sleep(args.delay)
                        elif connect_four_game.game_over:
                            break

                    except KeyboardInterrupt:
                        print("\n⚠️  Interrupted by user. Saving...")
                        break

                    except Exception as e:
                        print(f"\n❌ Error on turn {turn}: {e}")
                        import traceback
                        traceback.print_exc()

                        # Generate fallback response
                        from response_generator import generate_emergency_response
                        fallback_resp = generate_emergency_response(current_char, other_char, connect_four_history, {}, turn)
                        connect_four_history.append({'name': current_char['name'], 'content': fallback_resp})
                        print(fallback_resp)
                        turn += 1  # Increment turn anyway
                        continue  # Continue to next iteration

                # Handle game end
                if connect_four_game.winner == 'draw':
                    game_end_context = f"Game #{game_count} ended in a draw. Starting a new game..."
                    connect_four_history.append({'name': 'Narrator', 'content': game_end_context})
                    print(f"🏆 Game #{game_count} ended in a draw. Starting a new game...")
                    game_count += 1
                    continue  # Continue to next game
                else:
                    winner_name = player1['name'] if connect_four_game.winner == 'red' else player2['name']
                    game_end_context = f"Game #{game_count} ended. {winner_name} wins!"
                    connect_four_history.append({'name': 'Narrator', 'content': game_end_context})
                    print(f"🏆 Game #{game_count} ended. {winner_name} wins!")
                    print(f"Final board position:\n{connect_four_game.print_board()}")
                    break  # End the overall game loop

            # After connect-four mode, update the history variable to save properly
            history = connect_four_history

        elif args.uno:
            print("🃏🌈 Playing Uno Mode Activated 🌈🃏")
            print("=" * 80)

            # In uno mode, we support 2 characters
            if len(characters) != 2:
                print("❌ Uno mode requires exactly 2 characters")
                sys.exit(1)

            # Assign players
            player1 = characters[0]
            player2 = characters[1]

            # Initialize uno game
            uno_game = UnoGame()

            # Initialize conversation history with game context
            uno_history = []

            # Add opening context to history
            opening_context = f"Starting a game of Uno. {player1['name']} vs {player2['name']}."
            uno_history.append({'name': 'Narrator', 'content': opening_context})
            print(f"[TURN 1] Narrator: {opening_context}")
            print(f"Initial top card: {uno_game.print_top_card()}")

            # Uno game - multiple rounds until there's a decisive winner
            game_count = 1
            turn = 1  # Main turn counter

            while True:
                print(f"\n{'='*80}")
                print(f"UNO GAME #{game_count}")
                print('='*80)

                # Reset for the game
                uno_game.reset_game()

                # Add game start context to history
                game_start_context = f"Starting Uno game #{game_count}. {player1['name']} vs {player2['name']}."
                uno_history.append({'name': 'Narrator', 'content': game_start_context})
                print(f"[TURN {turn}] Narrator: {game_start_context}")
                print(f"Starting top card: {uno_game.print_top_card()}")
                print(f"Player hands sizes: {player1['name']} has {len(uno_game.get_current_player_hand())} cards, {player2['name']} has {len(uno_game.get_other_player_hand_size()) if uno_game.current_player == 'Player1' else len(uno_game.get_current_player_hand())} cards")
                turn += 1

                # Start the uno game loop
                uno_round = 1
                while not uno_game.game_over:
                    print(f"\n{'='*60}")
                    print(f"[UNO ROUND {uno_round}] Current State:")
                    print(f"Top card: {uno_game.print_top_card()}")
                    print(f"Current Player: {uno_game.current_player}")
                    print(f"Other Player hand size: {uno_game.get_other_player_hand_size()}")
                    print(f"Deck size: {len(uno_game.deck.cards)}")
                    print('='*60)

                    # Determine who is making the move
                    current_char = player1 if uno_game.current_player == 'Player1' else player2
                    other_char = player2 if uno_game.current_player == 'Player1' else player1

                    print(f"\n[TURN {turn}] {current_char['name']} (playing as {uno_game.current_player})")

                    try:
                        # Create game context for the turn, requesting explicit JSON output
                        uno_context = f"""
CRITICAL UNO GAME INSTRUCTIONS - YOU MUST FOLLOW THE EXACT JSON FORMAT BELOW OR THE GAME WILL STALL:

CURRENT GAME STATE:
- Top card: {uno_game.print_top_card()}
- Current Player: {uno_game.current_player}
- Your hand: {uno_game.get_hand_for_player(uno_game.current_player)}
- Opponent's hand size: {uno_game.get_other_player_hand_size()}

YOU MUST RESPOND IN EXACTLY THIS JSON FORMAT (NO EXCEPTIONS):
{{
  "dialogue": "Your dialogue and thought process about your choice",
  "action": "YOUR ACTION (either '[index]' for card index or 'draw')",
  "reasoning": "Why you made this choice"
}}

ABSOLUTELY CRITICAL REQUIREMENTS:
1. PROVIDE EXACTLY THE THREE FIELDS: "dialogue", "action", "reasoning"
2. THE "action" FIELD MUST BE EXCLUSIVELY EITHER A NUMBER INDEX (like '0', '1', '2') OR THE WORD 'draw'
3. DO NOT WRAP ACTION IN QUOTES, ASTERISKS, OR NARRATIVE FORM
4. DO NOT SAY "I play card at index 2" OR "I draw a card" OR "card 2" OR "**2**"
5. FOR PLAYING: USE ONLY THE INDEX NUMBER AS STRING: '0', '1', '2', etc.
6. FOR DRAWING: USE ONLY THE WORD: 'draw' (lowercase)
7. VERIFY YOUR PLAYED CARD MATCHES TOP CARD'S COLOR OR VALUE

EXAMPLE CORRECT RESPONSE:
{{
  "dialogue": "Looking at my hand, I see several options but this card best continues my strategy",
  "action": "2",
  "reasoning": "Card at index 2 matches the top card's color and advances my strategy"
}}

FAILURE TO USE EXACT JSON FORMAT WITH PROPER INDEX OR 'draw' WILL STALL THE GAME.
Think through your strategy before responding.
                        """.strip()

                        # Extract lorebook entries based on game context keywords
                        lorebook_entries = []
                        game_scenario = "Playing a game of uno"
                        from character_loader import extract_lorebook_entries
                        # Search for keywords in game scenario that match character lorebooks
                        scenario_keywords = game_scenario.lower().split()
                        for char in [current_char]:
                            entries = extract_lorebook_entries(char['raw_data'], uno_history, max_entries=2)
                            # Filter entries by scenario relevance
                            relevant_entries = []
                            for entry in entries:
                                if any(keyword in entry.lower() for keyword in scenario_keywords if len(keyword) > 3):
                                    relevant_entries.append(entry)
                            lorebook_entries.extend(relevant_entries[:1])  # Add at most 1 relevant entry

                        # Generate response with game context
                        resp = generate_response_adaptive(
                            current_char, other_char, uno_history, turn,
                            enable_environmental=not args.no_environmental,
                            similarity_threshold=args.similarity,
                            verbose=args.verbose,
                            scenario_context="Playing a game of uno",
                            lorebook_entries=lorebook_entries if lorebook_entries else None
                        )

                        # Validate response
                        if not isinstance(resp, str):
                            resp = str(resp)

                        uno_history.append({'name': current_char['name'], 'content': resp})
                        print(resp)

                        # Parse the JSON response for Uno
                        import json
                        import re

                        # First try to find JSON within the response
                        json_pattern = r'\{[^{}]*\}'  # Non-greedy match for JSON objects
                        matches = re.findall(json_pattern, resp, re.DOTALL)

                        dialogue = ""
                        action_choice = ""
                        reasoning = ""

                        for json_str in matches:
                            try:
                                json_clean = json_str.strip()
                                parsed = json.loads(json_clean)

                                dialogue = parsed.get('dialogue', '')
                                action_choice = parsed.get('action', '')
                                reasoning = parsed.get('reasoning', '')

                                if dialogue or action_choice:
                                    break
                            except json.JSONDecodeError:
                                continue  # Try the next match

                        if dialogue:
                            print(f"💬 Dialogue: {dialogue}")

                        # Process the action if provided
                        if action_choice:
                            action_lower = action_choice.lower().strip()

                            if 'draw' in action_lower:
                                # Draw a card
                                draw_success = uno_game.draw_card()
                                if draw_success:
                                    print(f"✅ {current_char['name']} drew a card")
                                    turn += 1  # Action was successful, increment turn
                                    uno_round += 1  # Also increment round
                                else:
                                    print(f"❌ Draw failed - deck may be empty")
                                    # Add feedback to history
                                    feedback = f"Drawing a card failed. Please try to make a valid move if possible."
                                    uno_history.append({'name': 'Referee', 'content': feedback})
                                    turn += 1  # Increment turn anyway
                            elif 'play' in action_lower:
                                # Extract card index from action like "play: [1]"
                                import re
                                index_match = re.search(r'play.*?([0-9]+)', action_lower)
                                if index_match:
                                    try:
                                        card_idx = int(index_match.group(1))
                                        current_hand_size = len(uno_game.get_current_player_hand())
                                        if 0 <= card_idx < current_hand_size:
                                            # Check if this card can be played
                                            hand = uno_game.get_current_player_hand()
                                            card_to_play = hand[card_idx]

                                            if uno_game.can_play_card(card_to_play):
                                                play_success = uno_game.play_card(card_idx)
                                                if play_success:
                                                    print(f"✅ {current_char['name']} played card: {card_to_play} from index {card_idx}")
                                                    print(f"New top card: {uno_game.print_top_card()}")
                                                    if uno_game.winner:
                                                        print(f"Game over! Winner: {uno_game.winner}")
                                                    turn += 1  # Move was successful, increment turn
                                                    uno_round += 1  # Also increment round
                                                else:
                                                    print(f"❌ Play failed - invalid card at index {card_idx}")
                                                    # Add feedback to history
                                                    feedback = f"The card at index {card_idx} could not be played. Please try a different card or draw instead."
                                                    uno_history.append({'name': 'Referee', 'content': feedback})
                                                    turn += 1  # Increment turn anyway
                                            else:
                                                print(f"❌ Card {card_idx} cannot be played on current top card")
                                                # Add feedback to history
                                                feedback = f"The card at index {card_idx} does not match the top card. Please select a compatible card or draw a new card."
                                                uno_history.append({'name': 'Referee', 'content': feedback})
                                                turn += 1  # Increment turn anyway
                                    except ValueError:
                                        print(f"❌ Could not parse card index from action: {action_choice}")
                                        # Add feedback to history
                                        feedback = f"I couldn't parse the card index from '{action_choice}'. Please provide a valid index of a card to play or 'draw'."
                                        uno_history.append({'name': 'Referee', 'content': feedback})
                                        turn += 1  # Increment turn anyway
                                else:
                                    print(f"❌ Could not extract card index from action: {action_choice}")
                                    # Add feedback to history
                                    feedback = f"I couldn't determine which card to play from '{action_choice}'. Please specify an index to play or 'draw'."
                                    uno_history.append({'name': 'Referee', 'content': feedback})
                                    turn += 1  # Increment turn anyway
                            else:
                                print(f"❌ Invalid action: {action_choice}. Must be 'play: [index]' or 'draw'.")
                                # Add feedback to history
                                feedback = f"Your action '{action_choice}' was invalid. Please 'play: [index]' to play a card or 'draw' to draw a card."
                                uno_history.append({'name': 'Referee', 'content': feedback})
                                turn += 1  # Increment turn anyway
                        else:
                            print(f"⚠️  No action provided.")
                            # Add feedback to history
                            feedback = f"Please provide a valid action ('play: [index]' to play a card or 'draw' to draw a card) in your response."
                            uno_history.append({'name': 'Referee', 'content': feedback})
                            turn += 1  # Increment turn anyway

                        # Delay before next turn
                        if not uno_game.game_over and turn < args.max_turns:
                            print(f"\n⏳ Waiting {args.delay} seconds...")
                            time.sleep(args.delay)
                        elif uno_game.game_over:
                            break

                    except KeyboardInterrupt:
                        print("\n⚠️  Interrupted by user. Saving...")
                        break

                    except Exception as e:
                        print(f"\n❌ Error on turn {turn}: {e}")
                        import traceback
                        traceback.print_exc()

                        # Generate fallback response
                        from response_generator import generate_emergency_response
                        fallback_resp = generate_emergency_response(current_char, other_char, uno_history, {}, turn)
                        uno_history.append({'name': current_char['name'], 'content': fallback_resp})
                        print(fallback_resp)
                        turn += 1  # Increment turn anyway
                        continue  # Continue to next iteration

                # Handle game end
                if uno_game.winner == 'draw':
                    game_end_context = f"Game #{game_count} ended in a draw. Starting a new game..."
                    uno_history.append({'name': 'Narrator', 'content': game_end_context})
                    print(f"🏆 Game #{game_count} ended in a draw. Starting a new game...")
                    game_count += 1
                    continue  # Continue to next game
                else:
                    winner_name = player1['name'] if uno_game.winner == 'Player1' else player2['name']
                    game_end_context = f"Game #{game_count} ended. {winner_name} wins!"
                    uno_history.append({'name': 'Narrator', 'content': game_end_context})
                    print(f"🏆 Game #{game_count} ended. {winner_name} wins!")
                    break  # End the overall game loop

            # After Uno mode, update the history variable to save properly
            history = uno_history

        elif args.rock_paper_scissors:
            print("🪨📄✂️ Playing Rock-Paper-Scissors Mode Activated ✂️📄🪨")
            print("=" * 80)

            # In rock-paper-scissors mode, we only support 2 characters
            if len(characters) != 2:
                print("❌ Rock-paper-scissors mode requires exactly 2 characters")
                sys.exit(1)

            # Assign players
            player1 = characters[0]
            player2 = characters[1]

            # Initialize rock-paper-scissors game
            rps_game = RockPaperScissorsGame()

            # Initialize conversation history with game context
            rps_history = []

            # Add opening context to history
            opening_context = f"Starting a game of rock-paper-scissors. {player1['name']} vs {player2['name']}."
            rps_history.append({'name': 'Narrator', 'content': opening_context})
            print(f"[TURN 1] Narrator: {opening_context}")

            # Rock-paper-scissors game - multiple rounds until there's a decisive winner
            game_count = 1
            turn = 1  # Main turn counter

            while True:
                print(f"\n{'='*80}")
                print(f"ROCK-PAPER-SCISSORS GAME #{game_count}")
                print('='*80)

                # Reset for the round
                rps_game.reset_round()

                # Add game start context to history
                game_start_context = f"Starting rock-paper-scissors round #{game_count}. {player1['name']} vs {player2['name']}."
                rps_history.append({'name': 'Narrator', 'content': game_start_context})
                print(f"[TURN {turn}] Narrator: {game_start_context}")
                turn += 1

                # Process moves for both players for this round
                move_success_1, turn = process_rps_move(rps_game, player1, player2, rps_history, turn, args, 'Player1')
                move_success_2, turn = process_rps_move(rps_game, player2, player1, rps_history, turn, args, 'Player2')

                # Process the round
                if move_success_1 and move_success_2:
                    print(f"✅ Both players made their moves")

                    # Determine winner after both moves are set
                    rps_game._determine_round_winner()

                    print(f"Result: {rps_game.get_result_message()}")
                    print(f"Stats: {rps_game.get_game_stats()}")

                    # Handle round end
                    if rps_game.winner == 'draw':
                        game_end_context = f"Round #{game_count} ended in a draw."
                        rps_history.append({'name': 'Narrator', 'content': game_end_context})
                        print(f"🏆 Round #{game_count} ended in a draw. Continuing to next round...")
                        game_count += 1
                        continue  # Continue to next round if it's a draw
                    else:
                        winner_name = player1['name'] if rps_game.winner == 'Player1' else player2['name']
                        game_end_context = f"Round #{game_count} ended. {winner_name} wins the round!"
                        rps_history.append({'name': 'Narrator', 'content': game_end_context})
                        print(f"🏆 Round #{game_count} ended. {winner_name} wins the round!")
                        break  # End the game after a win
                else:
                    print("❌ One or both players failed to make a valid move")
                    break

            # After rock-paper-scissors mode, update the history variable to save properly
            history = rps_history

        else:  # Normal conversation mode
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

                    # Periodic save every 10 turns
                    if turn % 10 == 0:
                        save_conversation_periodic(history, output_file, turn)

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

                    # Periodic save every 10 turns even in exception cases
                    if turn % 10 == 0:
                        save_conversation_periodic(history, output_file, turn)

                    continue

        # Save conversation and analyze
        save_conversation(history, output_file)
        print_final_analysis(history, args.similarity)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
