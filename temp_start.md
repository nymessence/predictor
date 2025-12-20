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
