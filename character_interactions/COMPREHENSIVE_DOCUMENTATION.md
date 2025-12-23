# Comprehensive Documentation for Dynamic AI Character Interaction System

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Game Modes](#game-modes)
5. [Features](#features)
6. [Architecture](#architecture)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)
9. [Changelog](#changelog)

## Overview

The Dynamic AI Character Interaction System is an advanced platform for facilitating conversations and games between AI characters with enhanced anti-repetition mechanisms. It enables realistic interactions between AI characters while maintaining consistent story continuity and preventing infinite loops.

## Installation

### Prerequisites
- Python 3.11+
- OpenAI-compatible API endpoint access (OpenRouter, Anthropic, Gemini, etc.)

### Dependencies
```bash
pip install -r requirements.txt
# Or
uv sync  # if using uv package manager
```

### Environment Setup
```bash
export AICHAT_API_KEY="your-api-key-here"
# OR
export OPENROUTER_API_KEY="your-openrouter-key-here"  # For OpenRouter
```

## Usage

### Basic Command Structure
```bash
python main.py [character_files] [options]
```

### Game Mode Options
```bash
# Chess game
python main.py --chess character1.json character2.json --delay 5 --max-turns 50

# Tic-Tac-Toe game
python main.py --tic-tac-toe character1.json character2.json --delay 3

# Rock-Paper-Scissors game
python main.py --rock-paper-scissors character1.json character2.json

# Hangman game
python main.py --hangman character1.json character2.json

# Twenty-One (Blackjack variant) game
python main.py --twenty-one character1.json character2.json

# Number Guessing game
python main.py --number-guessing character1.json character2.json

# Word Association game
python main.py --word-association character1.json character2.json

# Connect Four game
python main.py --connect-four character1.json character2.json

# Uno game
python main.py --uno character1.json character2.json
```

### Common Options
- `--delay N`: Wait N seconds between turns (default: 10)
- `--max-turns N`: Maximum conversation turns (default: 100)
- `--similarity N`: Repetition detection threshold (0.0-1.0, default: 0.45)
- `--emergency-threshold N`: Threshold for emergency repetition protocols (default: 0.7)
- `--critical-threshold N`: Threshold for critical repetition protocols (default: 0.9)
- `-o FILE`: Output JSON file for results (default: auto-generated from character names)
- `--resume FILE`: Resume from a saved conversation file
- `--start-turn N`: Starting turn number when resuming (default: 1)
- `--api-endpoint URL`: API endpoint URL (default: https://api.openai.com/v1)
- `--model NAME`: Model name (default: gpt-4o)
- `--api-key KEY`: API key (overrides environment variable)
- `--scenario "DESCRIPTION"`: Custom scenario to guide conversation and trigger lorebook entries

## Game Modes

### Chess Mode
- **Full chess implementation** with complete rule validation
- **Turn-based gameplay** with alternating players
- **Legal move validation** following FIDE rules
- **JSON-formatted moves** required to prevent repetition loops
- **Draw handling** for stalemates and automatic restarts
- **Player assignment**: First character = White, Second character = Black

### Tic-Tac-Toe Mode
- **Standard 3x3 gameplay** with win/draw detection
- **Turn-based system** with alternating X and O
- **Position validation** for [row, col] format moves
- **JSON-formatted moves** with [0-2, 0-2] coordinates

### Rock-Paper-Scissors Mode
- **Simultaneous choice system** for both players
- **Round-based gameplay** with statistics tracking
- **Natural language choice** parsing for rock/paper/scissors

### Hangman Mode
- **Word guessing** with visual hangman display
- **Letter-based gameplay** with limited incorrect guesses
- **JSON-formatted letter** input required

### Twenty-One Mode (Simplified Blackjack)
- **Card drawing** with hit/stand mechanics
- **Score tracking** to stay under 21
- **Dealer AI** following standard rules

### Number Guessing Game
- **Secret number** guessing with high/low feedback
- **Range-based** guessing (typically 1-100)
- **Limited attempts** with feedback on each guess

### Word Association Game
- **Sequential word** connections based on semantic relationships
- **Chain-based** gameplay with evolving themes
- **JSON-formatted** single-word responses

### Connect Four Game
- **Drop-piece** gameplay in 7-column, 6-row grid
- **Win detection** for four-in-a-row patterns
- **Column-based** move system (0-6)

### Uno Game
- **Card matching** based on color or value
- **Special card** handling (Draw Two, Skip, Reverse, Wild)
- **JSON-formatted** move selection

## Features

### Anti-Repetition Mechanisms
1. **Pattern Detection**: Identifies repetitive phrases, sentence structures, and dialogue patterns
2. **Emergency Protocols**: Activates when repetition exceeds threshold levels
3. **Lorebook Integration**: Dynamically triggers relevant lorebook entries based on conversation topics
4. **Environmental Triggers**: Adds environmental events to break repetitious patterns
5. **Token Management**: Sliding window mechanism to maintain context while managing length

### JSON Format Enforcement
- **Strict input validation** for all game modes requiring structured moves
- **Format examples** provided in all prompts
- **Error feedback** when incorrect formats are detected
- **Automatic recovery** to prevent infinite loops

### Autonomous Research Repository
- **Self-maintaining repository** for long-term data collection
- **Domain-specific folders** for various research areas
- **Automated updates** with periodic saving
- **Version tracking** and git integration

### Environmental Events
- **Dynamic scene changes** to break repetitive patterns
- **Thematic events** relevant to conversation context
- **Character-aware triggers** that make sense for the current situation

## Architecture

### Core Components
- `main.py`: Main execution orchestrator
- `character_loader.py`: Character file loading and validation
- `response_generator.py`: Adaptive response generation with repetition detection
- `repetition_detector.py`: Pattern analysis and blocking mechanisms
- `scenario_adapter.py`: Dynamic scenario adaptation and lorebook integration
- `chess_game_module.py`: Complete chess implementation
- `config.py`: Configuration constants and settings
- `utils.py`: Shared utility functions

### Game Modules
- Individual modules for each game type (e.g., `chess_game_module.py`, `tic_tac_toe_game.py`, etc.)
- Each game module implements the complete game logic with validation
- Consistent interface for integration with conversation system

### Data Structures
- **History**: Maintains conversation state with name/content pairs
- **Lorebook**: Contains contextual knowledge and character background
- **Environmental Events**: Dynamic triggers with character and setting relevance
- **Game State**: Per-game tracking of positions, moves, and outcomes

## Repetition Prevention Mechanisms

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

### File Path Handling
- Proper separation of directory paths from filenames
- Maintains hierarchical structure while sanitizing special characters only in filenames
- Prevents unintended character conversion in paths

## Troubleshooting

### Common Issues

#### Infinite Loops
- **Symptoms**: Same character keeps having the same turn repeatedly
- **Cause**: Character repeatedly making invalid moves or not following JSON format
- **Solution**: The system now automatically advances turns after 3 consecutive failures

#### Format Errors
- **Symptoms**: Moves not recognized, JSON parsing failures
- **Cause**: AI characters not following required JSON format
- **Solution**: Enhanced prompts now explicitly require JSON format with examples

#### API Errors
- **Symptoms**: Rate limit exceeded, connection errors
- **Cause**: High-frequency requests to API
- **Solution**: Built-in delay system and emergency responses

#### Character File Issues
- **Symptoms**: File not found, invalid character data
- **Cause**: Incorrect file names or structure
- **Solution**: Validates character files before starting conversation

### Performance Tips

1. **Optimal delay settings**:
   - Lower delays (1-3s) for faster gameplay
   - Higher delays (10-30s) for API rate limiting

2. **Similarity threshold adjustment**:
   - Lower values (0.3-0.4) for more diversity
   - Higher values (0.6-0.8) for more consistency

3. **Turn management**:
   - Use safety limits to prevent extremely long conversations
   - Monitor conversation length for token management

## FAQ

### Q: Why is the game stalling in chess mode?
A: This typically occurs when AI characters repeatedly fail to provide moves in the required JSON format. The system now automatically advances turns after 3 consecutive invalid responses to prevent this issue.

### Q: How do I add new game modes?
A: Create a new game module following the existing patterns (e.g., chess_game_module.py), implement the required methods, and add command-line option integration in main.py.

### Q: What are the JSON format requirements?
A: Each game mode requires specific JSON structure:
- Chess: `{"dialogue": "...", "move": "...", "board_state": "..."}`
- Tic-Tac-Toe: `{"dialogue": "...", "move": "[row, col]", "board_state": "..."}`
- Other games follow similar patterns with appropriate field names

### Q: How do I resume from a previous conversation?
A: Use the `--resume` flag with a saved conversation file and optionally `--start-turn` to specify the turn to resume from.

### Q: Can I change the output file location?
A: Yes, use the `-o` flag to specify a custom output path. The system preserves directory structure while sanitizing filenames.

## Changelog

### v1.2.0 - Current Version
- **Fixed infinite loop issues** in all game modes with automatic turn advancement after 3 consecutive failures
- **Enhanced JSON format enforcement** with more explicit instructions and examples
- **Improved file path handling** to preserve directory separators while sanitizing filenames
- **Added comprehensive error handling** for all game modes to prevent stalling
- **Implemented periodic auto-save** every 10 turns for recovery capability
- **Added autonomous research repository** with domain-specific organization
- **Enhanced environmental triggers** to prevent repetitive patterns
- **Strengthened lorebook integration** for more contextually relevant entries

### v1.1.0 - Previous Release
- **Initial game modes implementation** (chess, tic-tac-toe, rock-paper-scissors, etc.)
- **Basic anti-repetition mechanisms** with similarity detection
- **Character loader with validation**
- **Environmental trigger system**

### v1.0.0 - Original Release
- **Basic conversation system** between AI characters
- **File-based character loading**
- **JSON output generation**
- **Simple repetition detection**

## License
MIT License - See LICENSE file for details.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request