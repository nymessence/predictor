# Chess Game Mode Documentation

## Overview

The Chess Game Mode is an extension of the Dynamic AI Character Conversation System that allows two AI characters to play chess while engaging in dialogue. This mode enables characters to discuss strategy, thought processes, and reactions during gameplay while making legal chess moves on a virtual board.

## Features

- **Complete Chess Implementation**: Fully functional chess game with all standard rules
- **Legal Move Validation**: All moves are validated according to standard chess rules
- **Turn-Based Gameplay**: Characters take alternating turns based on their assigned color
- **Board Position Logging**: Complete board state is displayed and logged with each turn
- **Move History**: All moves are recorded in standard algebraic notation
- **Automatic Draw Handling**: Games ending in draws are automatically restarted
- **Winner Declaration**: Clear winner announced at the end of each game
- **AI-Driven Moves**: Characters suggest and make moves through AI responses

## New Command-Line Options

### `--chess` (Flag)
- **Description**: Enables chess game mode
- **Usage**: `--chess` or `--chess=true`
- **Default**: False

### `-t, --max-turns` (With Warning)
- **Description**: In chess mode, this parameter controls the maximum conversation turns as a safety limit, but games continue until completion
- **Note**: When used with `--chess`, a warning is displayed since chess games are meant to continue until a winner is determined

## Usage Examples

### Basic Chess Game
```bash
uv run --active character_interactions/main.py "character_interactions/json/nya_elyria.json" "character_interactions/json/empress_azalea.json" --chess --delay 5 --similarity 0.65 --api-endpoint "https://api.z.ai/api/paas/v4" --model "glm-4.6v-flash" --api-key $Z_AI_API_KEY -o "character_interactions/json/Nya_&_Azalea_Chess.json"
```

### Chess Game with Custom Settings
```bash
uv run --active character_interactions/main.py "character_interactions/json/nya_elyria.json" "character_interactions/json/empress_azalea.json" --chess --delay 3 --similarity 0.7 --api-endpoint "https://api.openai.com/v1" --model "gpt-4" --api-key $OPENAI_API_KEY -o "chess_game_output.json"
```

## How Chess Mode Works

1. **Character Assignment**: The first character specified is assigned as white, the second as black
2. **Board Display**: The initial chess board is displayed at the start of each game
3. **Turn Structure**: Characters take turns based on the current player (white moves first)
4. **Move Description**: Characters describe their thought process and intended move
5. **Move Extraction**: The system parses the character's response to extract the chess move
6. **Move Validation**: The move is validated against the current board position
7. **Move Execution**: Valid moves are applied to the board
8. **Game Continuation**: Play continues until a winning position is reached
9. **Draw Handling**: If a game ends in a draw, a new game begins automatically
10. **Winner Declaration**: When a game ends with a winner, the result is announced

## Move Description and Parsing

Characters describe their chess moves in natural language. The system attempts to extract valid moves from the response using pattern matching:

- **Algebraic Notation**: e4, Nf3, Bb5+, etc.
- **Source-Destination Format**: "move e2 to e4", "e2-e4"
- **Descriptive Moves**: "I will advance my pawn to e4"

The system validates all extracted moves against the current board position and legal move rules before applying them.

## Game Completion and Draw Handling

### Game End Conditions
- **Checkmate**: The player delivering checkmate is declared the winner
- **Resignation**: If a character indicates resignation, the opponent wins
- **Time Forfeit**: Currently not implemented but could be added
- **Insufficient Material**: Draw declared when neither side can win

### Draw Handling
When a game ends in a draw (stalemate, 50-move rule, or insufficient material), the system:
1. Records the draw result in the conversation history
2. Announces the draw to the user
3. Starts a new game with both characters playing again
4. Continues this process until a decisive result is achieved

## Output Format

The output JSON file contains:
- All character dialogues during the game
- Board positions at key moments
- Move history in algebraic notation
- Game results and winner declarations
- Referee feedback for invalid moves

## Constraints and Requirements

1. **Two Characters Required**: Chess mode requires exactly 2 character files
2. **Turn Limit Safety**: While games continue until completion, a turn limit can be set as a safety measure
3. **Legal Moves Only**: All moves must be legal according to standard chess rules
4. **Character Consistency**: Characters maintain their personalities while playing chess

## Implementation Details

The chess implementation includes:
- Complete move validation following FIDE rules
- Check/checkmate detection and validation
- Castling (kingside and queenside) support
- En passant captures
- Pawn promotion (automatically promotes to queen)
- Draw detection (stalemate, 50-move rule, insufficient material)

## Troubleshooting

### Common Issues

- **Characters not making moves**: The AI may need clearer prompts to format moves in standard notation
- **Invalid move detection**: Check that game logic is properly following the board state
- **Parsing failures**: Responses must contain valid square references for move extraction
- **Turn order problems**: Character assignment is based on file order (first=white, second=black)

### Performance Tips

- Use a higher similarity threshold to encourage different moves and avoid repetitive game patterns
- Adjust delay settings based on your API rate limits
- Monitor conversation length as chess games can become lengthy

## Future Enhancements

Potential improvements to the chess mode could include:
- Support for chess clock/timed games
- More sophisticated move parsing with error correction
- Advanced chess analysis and commentary
- Multiple concurrent games in a tournament format
- Enhanced board visualization options