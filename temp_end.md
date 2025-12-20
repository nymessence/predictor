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

## Token Management System

The system implements a sliding window mechanism to prevent conversation history from growing too large and causing token overflow issues or repetitive loops:

- **Estimation**: Uses approximate token counting to estimate history length
- **Sliding Window**: Keeps most recent interactions while respecting token limits
- **Context Preservation**: Maintains initial game context (Narrator entries) when possible
- **Chronological Order**: Ensures history flows chronologically even when truncated
- **Repetitive Loop Prevention**: Prevents the AI from falling into repetitive patterns due to context exhaustion

### How Token Management Works:
1. **History Estimation**: Estimates token count based on word count approximation
2. **Limit Check**: Determines if history exceeds reasonable token limits (default: 1000 tokens)
3. **Window Application**: Applies sliding window to preserve most recent entries
4. **Order Maintenance**: Maintains chronological order of preserved entries
5. **Context Priority**: Prioritizes initial game context and recent interactions

### Benefits:
- **Prevents Overflow**: Keeps conversation within token limits for API requests
- **Maintains Coherence**: Recent context is preserved for game understanding
- **Reduces Loops**: Eliminates repetitive patterns that occur with oversized history
- **Performance Boost**: Reduces unnecessary context length for faster responses

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

#### Token Management Issues:
- **History truncation**: Conversations may be shortened if they exceed token limits
- **Context preservation**: Important initial context is maintained even when history is truncated
- **Chronological flow**: Order of events is preserved in the shortened history

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