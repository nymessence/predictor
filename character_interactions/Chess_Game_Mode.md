# Game Modes Documentation

## Overview

The Game Modes are extensions of the Dynamic AI Character Conversation System that allow two AI characters to play various games while engaging in dialogue. These modes enable characters to discuss strategy, thought processes, and reactions during gameplay while making valid moves in different game contexts.

## Available Game Modes

### Chess Mode
- **Complete Chess Implementation**: Fully functional chess game with all standard rules
- **Legal Move Validation**: All moves are validated according to standard chess rules
- **Turn-Based Gameplay**: Characters take alternating turns based on their assigned color
- **Board Position Logging**: Complete board state is displayed and logged with each turn
- **Move History**: All moves are recorded in standard algebraic notation
- **Automatic Draw Handling**: Games ending in draws are automatically restarted
- **Winner Declaration**: Clear winner announced at the end of each game
- **AI-Driven Moves**: Characters suggest and make moves through AI responses

### Tic-Tac-Toe Mode
- **Complete Tic-Tac-Toe Implementation**: Standard 3x3 grid gameplay
- **Win Validation**: Checks for three-in-a-row (horizontal, vertical, diagonal)
- **Turn-Based Gameplay**: Characters take alternating turns (X goes first)
- **Board Position Logging**: Complete board state is displayed and logged with each turn
- **Draw Handling**: Detected when board is full with no winner
- **Winner Declaration**: Clear winner announced at the end of each game
- **AI-Driven Moves**: Characters suggest and make moves through AI responses

### Rock-Paper-Scissors Mode
- **Complete Rock-Paper-Scissors Implementation**: Standard game rules with win/loss/draw logic
- **Choice Validation**: Ensures valid choices (rock, paper, scissors)
- **Simultaneous Move Mechanics**: Both players choose simultaneously
- **Statistics Tracking**: Win/loss/draw records maintained
- **Winner Declaration**: Clear winner announced at the end of each round
- **AI-Driven Choices**: Characters discuss and select options through AI responses

### Hangman Mode
- **Complete Hangman Implementation**: Word guessing with progressive reveal
- **Word Selection**: Random selection from predefined word list
- **Letter Guessing**: Players guess letters one at a time
- **Visual Hangman**: ASCII art display of hangman state
- **Limited Mistakes**: Standard 6 incorrect guesses before loss
- **Win/Loss Detection**: Game ends on word completion or too many wrong guesses
- **AI-Guided Gameplay**: Characters strategize letter selection and discuss words
- **Structured JSON Output**: Dialogue, letter guess, and game state

### Twenty-One Mode
- **Complete Twenty-One Implementation**: Simplified blackjack-style gameplay
- **Card Values**: Standard card values (face cards = 10, Ace = 11 or 1)
- **Hit/Stand Mechanics**: Players choose to take cards or stay with current hand
- **Bust Detection**: Game ends if a player exceeds 21 points
- **Dealer AI**: Automatic dealer following standard rules
- **Objective**: Reach 21 points without going over
- **AI-Driven Strategy**: Characters discuss card values and hitting/standing decisions
- **Structured JSON Output**: Dialogue, action (hit/stand), and game state

## New Command-Line Options

### `--chess` (Flag)
- **Description**: Enables chess game mode where characters play chess and discuss moves
- **Usage**: `--chess` or `--chess=true`
- **Default**: False

### `--tic-tac-toe` (Flag)
- **Description**: Enables tic-tac-toe game mode where characters play tic-tac-toe
- **Usage**: `--tic-tac-toe` or `--tic_tac_toe=true`
- **Default**: False

### `--rock-paper-scissors` (Flag)
- **Description**: Enables rock-paper-scissors game mode where characters play rock-paper-scissors
- **Usage**: `--rock-paper-scissors` or `--rock_paper_scissors=true`
- **Default**: False

### `--hangman` (Flag)
- **Description**: Enables hangman game mode where characters guess letters to discover a secret word
- **Usage**: `--hangman` or `--hangman=true`
- **Default**: False

### `--twenty-one` (Flag)
- **Description**: Enables twenty-one (simplified blackjack) game mode where characters try to reach 21 without going over
- **Usage**: `--twenty-one` or `--twenty_one=true`
- **Default**: False

### `-t, --max-turns` (With Warning)
- **Description**: In game modes, this parameter controls the maximum conversation turns as a safety limit, but games continue until completion
- **Note**: When used with any game mode, a warning is displayed since games are meant to continue until a winner is determined

## Usage Examples

### Basic Chess Game
```bash
uv run --active character_interactions/main.py "character_interactions/json/nya_elyria.json" "character_interactions/json/empress_azalea.json" --chess --delay 5 --similarity 0.65 --api-endpoint "https://api.z.ai/api/paas/v4" --model "glm-4.6v-flash" --api-key $Z_AI_API_KEY -o "character_interactions/json/Nya_&_Azalea_Chess.json"
```

### Tic-Tac-Toe Game
```bash
uv run --active character_interactions/main.py "character_interactions/json/nya_elyria.json" "character_interactions/json/empress_azalea.json" --tic-tac-toe --delay 3 --similarity 0.7 --api-endpoint "https://api.openai.com/v1" --model "gpt-4" --api-key $OPENAI_API_KEY -o "ttt_game_output.json"
```

### Rock-Paper-Scissors Game
```bash
uv run --active character_interactions/main.py "character_interactions/json/nya_elyria.json" "character_interactions/json/empress_azalea.json" --rock-paper-scissors --delay 2 --similarity 0.6 --api-endpoint "https://api.openai.com/v1" --model "gpt-4" --api-key $OPENAI_API_KEY -o "rps_game_output.json"
```

### Hangman Game
```bash
uv run --active character_interactions/main.py "character_interactions/json/nya_elyria.json" "character_interactions/json/empress_azalea.json" --hangman --delay 3 --similarity 0.65 --api-endpoint "https://api.openai.com/v1" --model "gpt-4" --api-key $OPENAI_API_KEY -o "hangman_game_output.json"
```

### Twenty-One Game
```bash
uv run --active character_interactions/main.py "character_interactions/json/nya_elyria.json" "character_interactions/json/empress_azalea.json" --twenty-one --delay 4 --similarity 0.7 --api-endpoint "https://api.openai.com/v1" --model "gpt-4" --api-key $OPENAI_API_KEY -o "twentyone_game_output.json"
```

### Chess Game with Custom Settings
```bash
uv run --active character_interactions/main.py "character_interactions/json/nya_elyria.json" "character_interactions/json/empress_azalea.json" --chess --delay 3 --similarity 0.7 --api-endpoint "https://api.openai.com/v1" --model "gpt-4" --api-key $OPENAI_API_KEY -o "chess_game_output.json"
```

## How Game Modes Work

### Chess Mode
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

### Tic-Tac-Toe Mode
1. **Character Assignment**: The first character specified is assigned as X, the second as O
2. **Board Display**: The initial 3x3 board is displayed at the start of each game
3. **Turn Structure**: Characters take alternating turns (X moves first)
4. **Move Description**: Characters describe their thought process and intended position
5. **Position Extraction**: The system parses the character's response to extract the position [row, col]
6. **Position Validation**: The move is validated against the current board position (must be empty)
7. **Move Execution**: Valid moves are applied to the board
8. **Game Continuation**: Play continues until a winning pattern is achieved or board is full
9. **Draw Handling**: If a game ends in a draw, a new game begins automatically
10. **Winner Declaration**: When a game ends with a winner, the result is announced

### Rock-Paper-Scissors Mode
1. **Character Assignment**: Two characters participate as Player1 and Player2
2. **Choice Description**: Characters describe their thought process and intended choice
3. **Choice Extraction**: The system parses the character's response to extract the choice (rock/paper/scissors)
4. **Choice Validation**: The choice is validated as rock, paper, or scissors
5. **Simultaneous Input**: Both players make choices for each round
6. **Outcome Determination**: Results are calculated based on rock-paper-scissors rules
7. **Round Continuation**: Play continues with new rounds until a decisive winner
8. **Draw Handling**: If a round ends in a draw, a new round begins automatically
9. **Winner Declaration**: When a round ends with a decisive winner, the result is announced

### Hangman Mode
1. **Character Assignment**: First character guesses letters, second provides commentary/hints
2. **Word Selection**: Random word is selected from predefined list
3. **Initial Display**: Word is shown as blanks (e.g. "_ _ _ _ _")
4. **Letter Description**: Characters describe their thought process and letter guess
5. **Letter Extraction**: The system parses the character's response to extract the letter
6. **Letter Validation**: The letter is validated as a single alphabetic character not yet guessed
7. **Result Processing**: Letter is checked against the secret word and board updated
8. **Game Continuation**: Play continues until word is guessed or incorrect guesses are exhausted
9. **Draw/Loss Handling**: If player exhausts allowed incorrect guesses, new game begins automatically
10. **Winner Declaration**: When a game ends with the word guessed correctly, the result is announced

### Twenty-One Mode
1. **Character Assignment**: First character is player, second acts as dealer/AI reference
2. **Initial Deal**: Both player and dealer receive 2 cards
3. **Hand Display**: Player's hand and dealer's visible card are displayed
4. **Action Description**: Characters describe their thought process and decision (hit/stand)
5. **Action Extraction**: The system parses the character's response to extract the action
6. **Action Validation**: The action is validated as 'hit' or 'stand'
7. **Action Execution**: Valid actions are processed according to game rules
8. **Game Continuation**: Play continues until player busts, stands, or reaches 21
9. **Draw Handling**: If applicable, new round begins automatically
10. **Winner Declaration**: When a round ends with a winner, the result is announced

## Move/Choice Description and Parsing

Characters describe their game actions in natural language. The system attempts to extract valid moves/choices from the response using pattern matching:

### Chess:
- **Algebraic Notation**: e4, Nf3, Bb5+, etc.
- **Source-Destination Format**: "move e2 to e4", "e2-e4"
- **Descriptive Moves**: "I will advance my pawn to e4"

### Tic-Tac-Toe:
- **Position Format**: [0, 2], [1, 1], (0, 2), "0,2", "0 2", etc.
- **Coordinate Mapping**: Positions from [0,0] to [2,2] representing board positions
- **Descriptive Moves**: "I will place my mark in the center position"

### Rock-Paper-Scissors:
- **Explicit Choice**: "I choose rock", "my pick is paper", "scissors"
- **Direct Statement**: "rock", "paper", "scissors"
- **Strategic Talk**: "I'll counter with paper to beat your rock"

### Hangman:
- **Single Letter**: "I'll guess the letter 'e'", "my choice is 'a'"
- **Letter Format**: Just the letter like "e", "a", "t"
- **Strategic Description**: "I think the word has the letter 's'"

### Twenty-One:
- **Action Words**: "I'll hit", "I'll stand", "hit me", "I stand with this hand"
- **Strategic Description**: "This adds up to 17, I'll take another card" or "I'll stand at 18"
- **Direct Commands**: "hit", "stand"

The system validates all extracted moves/choices against the game rules and current state before applying them.

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

## Game Completion and Draw Handling

### Chess Game End Conditions
- **Checkmate**: The player delivering checkmate is declared the winner
- **Resignation**: If a character indicates resignation, the opponent wins
- **Time Forfeit**: Currently not implemented but could be added
- **Insufficient Material**: Draw declared when neither side can win

### Tic-Tac-Toe Game End Conditions
- **Three-in-a-row**: The player achieving three symbols in a row (horizontal, vertical, or diagonal) wins
- **Board Full**: Draw declared when the 3x3 board has no empty spaces

### Rock-Paper-Scissors Round End Conditions
- **Win Condition**: Rock beats scissors, scissors beat paper, paper beats rock
- **Tie**: Same choices result in a draw, leading to a new round

### Hangman Game End Conditions
- **Word Completed**: Player wins by guessing all letters in the secret word
- **Incorrect Guesses**: Player loses after 6 incorrect letter guesses
- **Complete Reveal**: Game ends when word is fully revealed or hangman is complete

### Twenty-One Game End Conditions
- **Bust**: A player loses by exceeding 21 points
- **Twenty-One**: A player wins by reaching exactly 21 points
- **Comparison**: Game ends when both players stand or both bust, comparing scores
- **Dealer Rules**: Dealer must hit on 16 or below, stand on 17 or above

## Output Format

The output JSON file contains:
- All character dialogues during the game
- Board positions or game states at key moments
- Move/action history in appropriate notation
- Game results and winner declarations
- Referee feedback for invalid moves/choices

## Constraints and Requirements

1. **Two Characters Required**: All game modes require exactly 2 characters
2. **Turn Limit Safety**: While games continue until completion, a turn limit can be set as a safety measure
3. **Valid Actions Only**: All moves/choices must be valid according to the game rules
4. **Character Consistency**: Characters maintain their personalities while playing games

## Implementation Details

The game implementations include:

### Chess:
- Complete move validation following FIDE rules
- Check/checkmate detection and validation
- Castling (kingside and queenside) support
- En passant captures
- Pawn promotion (automatically promotes to queen)
- Draw detection (stalemate, 50-move rule, insufficient material)

### Tic-Tac-Toe:
- Win detection for rows, columns, and diagonals
- Draw detection when board is full
- Valid move checking to ensure empty positions

### Rock-Paper-Scissors:
- Choice validation for rock, paper, and scissors
- Win/loss/tie determination logic
- Statistics tracking for multiple rounds

### Hangman:
- Word selection from a predefined dictionary
- Letter validation and tracking
- Visual hangman display
- Win/loss detection for incorrect guesses

### Twenty-One (Blackjack):
- Card dealing and value calculation
- Hit/stand decision processing
- Dealer AI following standard rules (hit on 16, stand on 17+)
- Bust and win/loss determination

## Troubleshooting

### Common Issues

#### Chess Mode:
- **Characters not making moves**: The AI may need clearer prompts to format moves in standard notation
- **Invalid move detection**: Check that game logic is properly following the board state
- **Parsing failures**: Responses must contain valid square references for move extraction
- **Turn order problems**: Character assignment is based on file order (first=white, second=black)

#### Tic-Tac-Toe Mode:
- **Position format errors**: Ensure positions are provided in [row, col] format
- **Occupied position attempts**: Verify moves target empty board positions
- **Coordinate validation**: Coordinates must be 0-2 for row and column

#### Rock-Paper-Scissors Mode:
- **Invalid choices**: Players must select only rock, paper, or scissors
- **Format issues**: Choice parsing supports various formats but prefers clear statements
- **Simultaneous move coordination**: Both players make moves before resolution

#### Hangman Mode:
- **Letter validation**: Only single letters are accepted
- **Repeat guess prevention**: System prevents guessing already guessed letters
- **Word completion**: Game recognizes when all letters are correctly guessed

#### Twenty-One Mode:
- **Bust detection**: System identifies hands exceeding 21 points
- **Ace value handling**: Aces properly counted as 1 or 11
- **Strategy validation**: Hit/stand decisions must follow game rules

### Performance Tips

- Use a higher similarity threshold to encourage varied strategies and avoid repetitive game patterns
- Adjust delay settings based on your API rate limits
- Monitor conversation length as games with multiple rounds can become lengthy

## Future Enhancements

Potential improvements to the game modes could include:
- Support for timed games (chess clock, hangman timer, etc.)
- More sophisticated move/choice parsing with error correction
- Advanced game analysis and commentary
- Multiple concurrent games in tournament formats
- Enhanced visualization options for all game types
- Additional game modes (Go, Checkers, Poker, etc.)