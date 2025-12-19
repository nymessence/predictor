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
                if has_chess_context:
                    # Clean the dialogue by removing the move and surrounding context
                    dialogue_text = re.sub(r'\b' + re.escape(potential_move) + r'\b', '', response_text).strip()
                    # Clean up extra punctuation and spacing
                    dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                    return dialogue_text, potential_move, ""

                # Special handling for move patterns that might appear without chess context
                # If it's a clear chess move pattern like "e4", "Nf3", etc., we'll be more lenient
                elif re.match(r'^[a-h][1-8]$', potential_move) or re.match(r'^[KQRBN][a-h]?[1-8]?[a-h][1-8]$', potential_move):
                    dialogue_text = re.sub(r'\b' + re.escape(potential_move) + r'\b', '', response_text).strip()
                    dialogue_text = re.sub(r'\s+', ' ', dialogue_text).strip()
                    return dialogue_text, potential_move, ""

            # If still no move found, return the entire text as dialogue with no move
            return response_text, "", ""

        # Specific function to parse chess move notation
        def parse_move_notation(move_notation, chess_game, current_player_color):
            """Parse algebraic move notation and return the from/to positions."""
            import re

            if not move_notation:
                return False, None, None

            move_notation = move_notation.strip()

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

            # Handle standard chess notation: e4, Nf3, exd5, Bxf7+, etc.
            # Check if the move looks like proper chess notation before processing
            chess_pattern = r'^([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][KQRBN]?[+#]?)$'
            if not re.match(chess_pattern, move_notation.lower()):
                # If it's not standard algebraic notation, check if it's in the form "e2 to e4" or "e2-e4"
                coord_to_coord_pattern = r'([a-h][1-8])\s*(?:to|-)\s*([a-h][1-8])'
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

                # For more complex notation, try to extract source info
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

            # If we couldn't parse it, return failure
            return False, None, None

        # Check for game mode specific constraints
        if args.chess and args.max_turns != MAX_TURNS:
            print(f"⚠️  WARNING: --max-turns parameter is not recommended in chess mode as games continue until completion")
            print(f"   Chess games will continue until a winner is determined, regardless of turn count.")
        elif (args.tic_tac_toe or args.rock_paper_scissors or args.hangman or args.twenty_one) and args.max_turns != MAX_TURNS:
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
                        # Create chess context for the turn, requesting JSON output
                        chess_context = f"""
Chess Game Context:
- Current Board Position:
{chess_game.print_board()}

- Move History: {' '.join(chess_game.move_history)}
- Current Player: {chess_game.current_player}
- Your color: {current_char['chess_color']}

You are playing chess. It's your turn to make a move. Please respond in the following JSON format:
{{
  "dialogue": "Your dialogue and thought process about the chess position",
  "move": "The chess move in algebraic notation (e.g., 'e4', 'Nf3', 'O-O', 'exd5', etc.)",
  "board_state": "A visual representation of the board after your move (only include if this is your move)"
}}

Make sure your move is legal in the current position. Think through your strategy before responding.
                        """.strip()

                        # Generate response with chess context
                        resp = generate_response_adaptive(
                            current_char, other_char, chess_history, turn,
                            enable_environmental=not args.no_environmental,
                            similarity_threshold=args.similarity,
                            verbose=args.verbose,
                            scenario_context="Playing a game of chess"
                        )

                        # Validate response
                        if not isinstance(resp, str):
                            resp = str(resp)

                        chess_history.append({'name': current_char['name'], 'content': resp})
                        print(resp)

                        # Parse the JSON response
                        dialogue, move_notation, board_state = parse_game_json_response(resp, current_char['name'])

                        if dialogue:
                            print(f"💬 Dialogue: {dialogue}")

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
                                else:
                                    print(f"❌ Move failed - illegal move attempted: {move_notation}")
                                    # Add feedback to history
                                    feedback = f"Your move '{move_notation}' was invalid or illegal. Please try again with a valid chess move from the current position."
                                    chess_history.append({'name': 'Referee', 'content': feedback})
                                    turn += 1  # Increment turn anyway
                            else:
                                print(f"❌ Could not parse move notation: {move_notation}")
                                # Add feedback to history
                                feedback = f"I couldn't parse the move '{move_notation}'. Please provide a valid chess move in algebraic notation."
                                chess_history.append({'name': 'Referee', 'content': feedback})
                                turn += 1  # No valid move, increment turn
                        else:
                            print(f"⚠️  No move provided. Current player: {chess_game.current_player}")
                            # Add feedback to history
                            feedback = f"Please provide a valid chess move in your response."
                            chess_history.append({'name': 'Referee', 'content': feedback})
                            turn += 1  # No move provided, increment turn

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
                        turn += 1  # Increment turn anyway
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
                        # Create game context for the turn, requesting JSON output
                        ttt_context = f"""
Tic-Tac-Toe Game Context:
- Current Board Position:
{ttt_game.print_board()}

- Current Player: {ttt_game.current_player}
- Your symbol: {current_char['ttt_symbol']}

You are playing tic-tac-toe. It's your turn to make a move. Please respond in the following JSON format:
{{
  "dialogue": "Your dialogue and thought process about the game position",
  "move": "The move in format (row, col) where row and col are 0-2 (e.g., [0, 2] for top right)",
  "board_state": "A visual representation of the board after your move (only include if this is your move)"
}}

Think through your strategy before responding.
                        """.strip()

                        # Generate response with game context
                        resp = generate_response_adaptive(
                            current_char, other_char, ttt_history, turn,
                            enable_environmental=not args.no_environmental,
                            similarity_threshold=args.similarity,
                            verbose=args.verbose,
                            scenario_context="Playing a game of tic-tac-toe"
                        )

                        # Validate response
                        if not isinstance(resp, str):
                            resp = str(resp)

                        ttt_history.append({'name': current_char['name'], 'content': resp})
                        print(resp)

                        # Parse the JSON response
                        dialogue, move_notation, board_state = parse_game_json_response(resp, current_char['name'])

                        if dialogue:
                            print(f"💬 Dialogue: {dialogue}")

                        # Attempt to make the move if notation is provided
                        if move_notation:
                            # Parse the move notation (should be in format like [0, 2], "0,2", "0 2", etc.)
                            row, col = parse_ttt_move(move_notation)

                            if row is not None and col is not None:
                                success = ttt_game.make_move(row, col)
                                if success:
                                    print(f"✅ Move successfully made: {row}, {col}")
                                    turn += 1  # Move was successful, increment turn
                                    ttt_turn += 1  # Also increment tic-tac-toe turn
                                else:
                                    print(f"❌ Move failed - invalid move attempted: {row}, {col} (position may be occupied or out of bounds)")
                                    # Add feedback to history
                                    feedback = f"Your move at position ({row}, {col}) was invalid or occupied. Please try again with a valid empty position."
                                    ttt_history.append({'name': 'Referee', 'content': feedback})
                                    turn += 1  # Increment turn anyway
                            else:
                                print(f"❌ Could not parse move notation: {move_notation}")
                                # Add feedback to history
                                feedback = f"I couldn't parse the move '{move_notation}'. Please provide a valid move in format [row, col] where row and col are 0-2."
                                ttt_history.append({'name': 'Referee', 'content': feedback})
                                turn += 1  # No valid move, increment turn
                        else:
                            print(f"⚠️  No move provided. Current player: {ttt_game.current_player}")
                            # Add feedback to history
                            feedback = f"Please provide a valid move in your response."
                            ttt_history.append({'name': 'Referee', 'content': feedback})
                            turn += 1  # No move provided, increment turn

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
                        turn += 1  # Increment turn anyway
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
                    # Create game context for the turn, requesting JSON output
                    hangman_context = f"""
Hangman Game Context:
- Current word: {hangman_game.get_current_display()}
- Guessed letters: {hangman_game.get_guessed_letters()}
- Remaining incorrect guesses: {hangman_game.get_remaining_guesses()}
- Hangman status:
{hangman_game.get_hangman_status()}

You are playing hangman. Guess a single letter to uncover the secret word. Respond in the following JSON format:
{{
  "dialogue": "Your dialogue and thought process about which letter to guess",
  "letter": "The single letter you want to guess (e.g., 'e')",
  "reasoning": "Why you chose this letter"
}}

Make your guess now.
                    """.strip()

                    # Generate response with game context
                    resp = generate_response_adaptive(
                        current_char, other_char, hangman_history, turn,
                        enable_environmental=not args.no_environmental,
                        similarity_threshold=args.similarity,
                        verbose=args.verbose,
                        scenario_context="Playing a game of hangman"
                    )

                    # Validate response
                    if not isinstance(resp, str):
                        resp = str(resp)

                    hangman_history.append({'name': current_char['name'], 'content': resp})
                    print(resp)

                    # Parse the JSON response
                    dialogue, letter_guess, reasoning = parse_game_json_response(resp, current_char['name'])

                    if dialogue:
                        print(f"💬 Dialogue: {dialogue}")

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
                        else:
                            print(f"❌ Guess failed - invalid letter: {letter_guess}")
                            # Add feedback to history
                            feedback = f"Your guess '{letter_guess}' was invalid. Please guess a single letter that hasn't been guessed yet."
                            hangman_history.append({'name': 'Referee', 'content': feedback})
                            turn += 1  # Increment turn anyway
                    else:
                        print(f"⚠️  No letter provided for guess.")
                        # Add feedback to history
                        feedback = f"Please provide a valid letter to guess in your response."
                        hangman_history.append({'name': 'Referee', 'content': feedback})
                        turn += 1  # No letter provided, increment turn

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
                    # Create game context for the turn, requesting JSON output
                    twentyone_context = f"""
Twenty-One Game Context:
- Your hand: {twentyone_game.get_player_hand_str()}
- Your score: {twentyone_game.get_player_score()}
- Dealer's visible card: {twentyone_game.get_dealer_hand_str(hide_first=True)}

You are playing twenty-one (simplified blackjack). Decide to hit (take another card) or stand (keep current hand). Respond in the following JSON format:
{{
  "dialogue": "Your dialogue and thought process about your decision",
  "action": "Your decision: 'hit' to take another card or 'stand' to keep current hand",
  "reasoning": "Why you made this decision"
}}

Make your decision now.
                    """.strip()

                    # Generate response with game context
                    resp = generate_response_adaptive(
                        current_char, other_char, twentyone_history, turn,
                        enable_environmental=not args.no_environmental,
                        similarity_threshold=args.similarity,
                        verbose=args.verbose,
                        scenario_context="Playing a game of twenty-one"
                    )

                    # Validate response
                    if not isinstance(resp, str):
                        resp = str(resp)

                    twentyone_history.append({'name': current_char['name'], 'content': resp})
                    print(resp)

                    # Parse the JSON response
                    dialogue, action_choice, reasoning = parse_game_json_response(resp, current_char['name'])

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
                        else:
                            print(f"❌ Invalid action: {action_choice}. Must be 'hit' or 'stand'.")
                            # Add feedback to history
                            feedback = f"Your action '{action_choice}' was invalid. Please choose 'hit' or 'stand'."
                            twentyone_history.append({'name': 'Referee', 'content': feedback})
                            turn += 1  # Invalid action, return incremented turn
                    else:
                        print(f"⚠️  No action provided.")
                        # Add feedback to history
                        feedback = f"Please provide a valid action ('hit' or 'stand') in your response."
                        twentyone_history.append({'name': 'Referee', 'content': feedback})
                        turn += 1  # No action provided, increment turn

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
                    turn += 1  # Increment turn anyway
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
