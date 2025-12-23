# Dynamic AI Character Interaction System

Advanced conversation system with anti-repetition mechanisms and game modes

## 📁 File Structure

```
character_interactions/
├── main.py                    # Main execution orchestrator
├── config.py                  # Configuration constants
├── character_loader.py        # Character loading & analysis
├── response_generator.py      # Response quality control with repetition detection
├── repitition_detector.py     # Pattern analysis and blocking
├── contextual_triggers.py     # Dynamic environmental events
├── scenario_adapter.py        # Dynamic scenario adaptation
├── chess_game_module.py       # Complete chess implementation
├── tic_tac_toe_game.py        # Tic-tac-toe game logic
├── rock_paper_scissors_game.py # Rock-paper-scissors game logic
├── hangman_game.py            # Hangman game logic
├── twenty_one_game.py         # 21/Blackjack game logic
├── number_guessing_game.py    # Number guessing game logic
├── word_association_game.py   # Word association game logic
├── connect_four_game.py       # Connect-four game logic
├── uno_game.py                # Uno card game logic
├── utils.py                   # Utility functions
└── Game_Modes_Documentation.md # Comprehensive documentation for game modes
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
# OR with uv: uv sync

# Set API key
export AICHAT_API_KEY='your-api-key-here'
# OR export OPENROUTER_API_KEY='your-openrouter-key' if using OpenRouter
```

### Usage

```bash
# Normal conversation mode
python main.py character1.json character2.json

# Chess game mode
python main.py --chess character1.json character2.json --delay 5 --max-turns 100

# Tic-tac-toe game mode
python main.py --tic-tac-toe character1.json character2.json --delay 3

# Rock-paper-scissors game mode
python main.py --rock-paper-scissors character1.json character2.json

# Hangman game mode
python main.py --hangman character1.json character2.json

# Twenty-one game mode
python main.py --twenty-one character1.json character2.json

# Number guessing game mode
python main.py --number-guessing character1.json character2.json

# Word association game mode
python main.py --word-association character1.json character2.json

# Connect-four game mode
python main.py --connect-four character1.json character2.json

# Uno game mode
python main.py --uno character1.json character2.json

# All modes support additional options:
# --delay N: Wait N seconds between turns (default: 10)
# --max-turns N: Maximum conversation turns (default: 100)
# --similarity N: Repetition threshold (0.0-1.0, default: 0.45)
# -o FILE: Output JSON file (default: auto-generated)
# --scenario "DESCRIPTION": Custom scenario context
```

## ⚡ Key Features

### Anti-Repetition Mechanisms
- **Pattern Detection**: Identifies repetitive phrases, sentence structures, and dialogue patterns
- **Emergency Protocols**: Activates when repetition exceeds threshold levels
- **Lorebook Integration**: Dynamically triggers relevant lorebook entries based on conversation topics
- **Environmental Triggers**: Adds environmental events to break repetitious patterns
- **Token Management**: Sliding window mechanism to maintain context while managing length

### Game Modes
- **Chess**: Complete implementation with all standard rules and move validation
- **Tic-Tac-Toe**: Standard 3x3 gameplay with win/draw detection
- **Rock-Paper-Scissors**: Simultaneous choice system with statistics tracking
- **Hangman**: Word guessing with visual hangman display
- **Twenty-One**: Simplified blackjack with hit/stand mechanics
- **Number Guessing**: Secret number guessing with high/low feedback
- **Word Association**: Sequential word connections based on semantic relationships
- **Connect Four**: Drop-piece gameplay in 7-column, 6-row grid
- **Uno**: Card matching with special card handling

### Advanced Capabilities
- **JSON Format Enforcement**: Strict validation to ensure proper move formats
- **Consecutive Failure Tracking**: Prevents infinite loops after 3 failed attempts
- **Auto-Save Functionality**: Periodic saves every 10 turns to prevent data loss
- **File Path Handling**: Proper preservation of directory structure while sanitizing filenames
- **Resume Capability**: Continue from saved conversation files
- **Scenario Context**: Guided conversations based on custom scenarios

## 🔒 Repetition Prevention & Error Handling

### Technical Safeguards
1. **Consecutive failure tracking**: Counts consecutive invalid moves/actions
2. **Turn advancement on failures**: Automatically moves to next player after 3 failures to prevent infinite loops
3. **Move format validation**: Strict checking of JSON format requirements
4. **Duplicate detection**: Identifies when same moves/words are repeated
5. **Progressive escalation**: Increases penalties for repeated violations

### Implementation Details
- The system tracks up to 3 consecutive failures before advancing turns automatically
- Each game mode implements its own failure tracking mechanisms
- JSON format is enforced with explicit examples and error messaging
- When failures occur, the game automatically progresses to prevent stalling

### Format Requirements
Each game mode requires specific JSON structure:
- Chess: `{"dialogue": "...", "move": "...", "board_state": "..."}`
- Tic-Tac-Toe: `{"dialogue": "...", "move": "[row, col]", "board_state": "..."}`
- Other games follow similar patterns with appropriate field names

## 📚 Documentation

Comprehensive documentation is available in the `Game_Modes_Documentation.md` file, including:
- Detailed instructions for each game mode
- Parameter descriptions and usage examples
- Troubleshooting guides
- Implementation notes

## 🚨 Troubleshooting

### Common Issues
1. **Game Stalling**: Usually caused by AI not following JSON format - the system now automatically advances after 3 failures
2. **File Not Found**: Check character file paths and ensure files exist
3. **API Errors**: Verify your API key is set and account has sufficient credits
4. **Repetition Loops**: The system now includes multiple layers of protection against this

### Performance Tips
- Use higher similarity thresholds (0.6-0.8) for more consistency
- Use lower similarity thresholds (0.3-0.5) to encourage varied responses
- Adjust delay settings based on your API rate limits
- Monitor conversation length as games with multiple rounds can become lengthy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with different character combinations and game modes
5. Submit a pull request with clear descriptions

## 📄 License

MIT License - See LICENSE file for details.
