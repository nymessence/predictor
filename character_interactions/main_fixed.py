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
from chess_game import ChessGame


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
    # Chess game argument
    parser.add_argument('--chess', action='store_true',
                        help='Enable chess game mode where characters play chess and discuss moves')
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

        # Check for chess mode specific constraints
        if args.chess and args.max_turns != MAX_TURNS:
            print(f"⚠️  WARNING: --max-turns parameter is not recommended in chess mode as games continue until completion")
            print(f"   Chess games will continue until a winner is determined, regardless of turn count.")

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

        # Check if we're in chess mode
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

                    # Process one move for the current player
                    move_success, turn = process_chess_move(chess_game, current_char, other_char, chess_history, turn, args, chess_game.current_player)

                    if move_success:
                        # If move was successful, increment chess turn
                        chess_turn += 1
                    # If move failed, we don't increment chess turn, so the same player gets another chance

                    # Delay before next turn
                    if not chess_game.game_over and turn < args.max_turns:  # We should still respect the max turns as a safety
                        print(f"\n⏳ Waiting {args.delay} seconds...")
                        time.sleep(args.delay)
                    elif chess_game.game_over:
                        break

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

        # Add a helper function to extract chess moves from the AI response
        def extract_chess_move(response_text, chess_game, current_player_color):
            """
            Extract a chess move from the AI's response text.
            This function looks for standard chess notation and converts it to board coordinates.
            """
            import re

            # Look for common chess move patterns in the response
            # Enhanced pattern matching to extract moves from text
            # Handle various formats like "e4", "Nf3", "exd5", "O-O", "e7e8Q" (pawn promotion), etc.

            # First, try to extract source and destination squares using patterns
            # Look for patterns like "e2 to e4", "move e2 to e4", "e2-e4", etc.
            source_dest_pattern = r'([a-h][1-8])\s*(?:to|-)\s*([a-h][1-8])'
            matches = re.findall(source_dest_pattern, response_text, re.IGNORECASE)

            if matches:
                for from_sq, to_sq in matches:
                    try:
                        # Convert algebraic to board coordinates
                        from_col = ord(from_sq[0].lower()) - ord('a')
                        from_row = 8 - int(from_sq[1])
                        to_col = ord(to_sq[0].lower()) - ord('a')
                        to_row = 8 - int(to_sq[1])

                        from_pos = (from_row, from_col)
                        to_pos = (to_row, to_col)

                        # Check if this is a valid move in the current position
                        if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                            return from_pos, to_pos
                    except (IndexError, ValueError):
                        continue  # Skip invalid squares

            # Next, look for standard algebraic notation (without captures marked explicitly)
            # Pattern: piece letter (optional) + disambiguating file/rank (optional) + destination square
            alg_pattern = r'\b([KQRBN]?[a-h]?[1-8]?[a-h][1-8])\b'
            alg_matches = re.findall(alg_pattern, response_text)

            # Filter for pure square patterns (no piece letters, just destination)
            destination_only = [m for m in alg_matches if len(m) == 2 and m[0] in 'abcdefgh' and m[1] in '12345678']

            # If we find destination squares, try to find matching source pieces
            for dest_sq in destination_only:
                try:
                    dest_col = ord(dest_sq[0].lower()) - ord('a')
                    dest_row = 8 - int(dest_sq[1])

                    # Find all pieces of current player's color that can move to this destination
                    for row in range(8):
                        for col in range(8):
                            piece = chess_game.get_piece_at(row, col)
                            if piece and chess_game.is_own_piece(piece, current_player_color):
                                # Check if this piece can move to the destination
                                temp_moves = chess_game._get_piece_valid_moves(row, col)
                                if (dest_row, dest_col) in temp_moves:
                                    from_pos = (row, col)
                                    to_pos = (dest_row, dest_col)

                                    # Make sure the complete move is legal
                                    if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                                        return from_pos, to_pos
                except (IndexError, ValueError):
                    continue  # Skip invalid squares

            # If we still haven't found a move, try to find any valid move from the text
            # Look for two square references in the text
            squares = re.findall(r'([a-h][1-8])', response_text.lower())
            seen_squares = set()
            unique_squares = []
            for sq in squares:
                if sq not in seen_squares:
                    unique_squares.append(sq)
                    seen_squares.add(sq)

            # Try combinations of squares
            if len(unique_squares) >= 2:
                for i in range(len(unique_squares)):
                    for j in range(len(unique_squares)):
                        if i != j:
                            from_sq = unique_squares[i]
                            to_sq = unique_squares[j]

                            try:
                                # Convert algebraic to board coordinates
                                from_col = ord(from_sq[0]) - ord('a')
                                from_row = 8 - int(from_sq[1])
                                to_col = ord(to_sq[0]) - ord('a')
                                to_row = 8 - int(to_sq[1])

                                from_pos = (from_row, from_col)
                                to_pos = (to_row, to_col)

                                # Check if this is a valid move in the current position
                                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                                    return from_pos, to_pos
                            except (IndexError, ValueError):
                                continue  # Skip invalid squares

            # If no move was found, return None
            return None, None

        def parse_chess_json_response(response_text, character_name):
            """Parse the JSON response from the AI containing dialogue, move, and board state."""
            import json
            import re

            # First try to find JSON within the response
            # Look for JSON between curly braces
            json_pattern = r'\{.*?\}'  # Non-greedy match for JSON objects
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            if not matches:
                # If no JSON found, return with None values
                return response_text, None, None

            # Try each match until one parses correctly
            for json_str in matches:
                try:
                    # Remove any leading/trailing text that might interfere
                    json_clean = json_str.strip()
                    parsed = json.loads(json_clean)

                    dialogue = parsed.get('dialogue', '')
                    move = parsed.get('move', '').strip()
                    board_state = parsed.get('board_state', '')

                    # If we successfully parsed and have at least a move or dialogue
                    if dialogue or move:
                        return dialogue, move, board_state
                except json.JSONDecodeError:
                    continue  # Try the next match

            # If no JSON could be parsed, return the original text as dialogue
            return response_text, None, None

        def parse_move_notation(move_notation, chess_game, current_player_color):
            """Parse algebraic move notation and return the from/to positions."""
            import re

            if not move_notation:
                return False, None, None

            move_notation = move_notation.strip()

            # Handle castling notation
            if move_notation.lower() in ['o-o', '0-0']:  # Kingside castling
                # For white: king from (7,4) to (7,6), rook from (7,7) to (7,5)
                # For black: king from (0,4) to (0,6), rook from (0,7) to (0,5)
                row = 7 if current_player_color == 'white' else 0
                from_pos = (row, 4)  # King start position
                to_pos = (row, 6)   # King end position (kingside castle)

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            elif move_notation.lower() in ['o-o-o', '0-0-0']:  # Queenside castling
                # For white: king from (7,4) to (7,2), rook from (7,0) to (7,3)
                # For black: king from (0,4) to (0,2), rook from (0,0) to (0,3)
                row = 7 if current_player_color == 'white' else 0
                from_pos = (row, 4)  # King start position
                to_pos = (row, 2)   # King end position (queenside castle)

                if chess_game.is_move_legal(from_pos, to_pos, current_player_color):
                    return True, from_pos, to_pos
                return False, None, None

            # Handle standard notation like: e4, Nf3, exd5, Bxf7+, etc.
            # Extract destination square (last 2 characters that look like a square)
            dest_match = re.search(r'([a-h][1-8])$', move_notation.lower())
            if dest_match:
                dest_sq = dest_match.group(1)
                dest_col = ord(dest_sq[0]) - ord('a')
                dest_row = 8 - int(dest_sq[1])
                dest_pos = (dest_row, dest_col)

                # If the notation is just a destination square (like 'e4'), try to find the piece that can move there
                if len(move_notation.strip()) == 2:  # Simple notation like 'e4'
                    # Find all pieces that can move to this destination
                    for row in range(8):
                        for col in range(8):
                            piece = chess_game.get_piece_at(row, col)
                            if piece and chess_game.is_own_piece(piece, current_player_color):
                                # Check if this piece can move to the destination
                                temp_moves = chess_game._get_piece_valid_moves(row, col)
                                if dest_pos in temp_moves:
                                    from_pos = (row, col)
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


        # Chess move processing function
        def process_chess_move(chess_game, current_char, other_char, chess_history, turn, args, current_player_color):
            """Process a single chess move by the current player."""
            print(f"\n[TURN {turn}] {current_char['name']} (playing as {current_player_color})")

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
                dialogue, move_notation, board_state = parse_chess_json_response(resp, current_char['name'])

                if dialogue:
                    print(f"💬 Dialogue: {dialogue}")

                # Attempt to make the chess move if notation is provided
                if move_notation:
                    # Try to parse the move notation
                    success, from_pos, to_pos = parse_move_notation(move_notation, chess_game, current_player_color)

                    if success and from_pos and to_pos:
                        move_success = chess_game.make_move(from_pos, to_pos)
                        if move_success:
                            print(f"✅ Move successfully made: {chess_game.move_history[-1] if chess_game.move_history else move_notation}")
                            return True, turn + 1  # Move was successful, return incremented turn
                        else:
                            print(f"❌ Move failed - illegal move attempted: {move_notation}")
                            # Add feedback to history
                            feedback = f"Your move '{move_notation}' was invalid or illegal. Please try again with a valid chess move from the current position."
                            chess_history.append({'name': 'Referee', 'content': feedback})
                            return False, turn + 1  # Move failed, but return incremented turn
                    else:
                        print(f"❌ Could not parse move notation: {move_notation}")
                        # Add feedback to history
                        feedback = f"I couldn't parse the move '{move_notation}'. Please provide a valid chess move in algebraic notation."
                        chess_history.append({'name': 'Referee', 'content': feedback})
                        return False, turn + 1  # No valid move, increment turn
                else:
                    print(f"⚠️  No move provided. Current player: {current_player_color}")
                    # Add feedback to history
                    feedback = f"Please provide a valid chess move in your response."
                    chess_history.append({'name': 'Referee', 'content': feedback})
                    return False, turn + 1  # No move provided, increment turn

            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user. Saving...")
                return None, turn  # Return None to indicate interruption

            except Exception as e:
                print(f"\n❌ Error on turn {turn}: {e}")
                import traceback
                traceback.print_exc()

                # Generate fallback response
                from response_generator import generate_emergency_response
                fallback_resp = generate_emergency_response(current_char, other_char, chess_history, {}, turn)
                chess_history.append({'name': current_char['name'], 'content': fallback_resp})
                print(fallback_resp)
                return False, turn + 1


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
