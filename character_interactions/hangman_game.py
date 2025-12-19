#!/usr/bin/env python3
"""
Hangman Game Class
Implements a complete hangman game with word guessing and win validation.
"""

import random
from typing import List, Set, Optional


class HangmanGame:
    """
    A complete hangman game implementation with word guessing
    and win validation for integration with character interactions.
    """
    
    def __init__(self, word_list: Optional[List[str]] = None):
        """Initialize the hangman game with a random word."""
        # Default word list if none provided
        if word_list is None:
            word_list = [
                "python", "chess", "computer", "programming", "algorithm", "function", 
                "variable", "class", "method", "string", "integer", "boolean",
                "database", "framework", "library", "module", "package",
                "keyboard", "monitor", "software", "hardware", "internet",
                "website", "application", "development", "engineer", "science",
                "technology", "knowledge", "experience", "practice", "learning",
                "machine", "artificial", "intelligence", "data", "analysis",
                "network", "security", "encryption", "protocol", "wireless",
                "mobile", "device", "interface", "system", "process", "service",
                "company", "product", "market", "customer", "business"
            ]
        
        self.word_list = [word.lower() for word in word_list]
        self.reset_game()
    
    def reset_game(self):
        """Start a new hangman game with a random word."""
        self.word = random.choice(self.word_list)
        self.correct_letters = set()
        self.incorrect_guesses = 0
        self.max_incorrect = 6  # Traditional hangman
        self.game_over = False
        self.winner = None  # None, 'Player' or 'Hangman'
        self.guessed_letters = set()  # Track all guessed letters
    
    def guess_letter(self, letter: str) -> bool:
        """Process a letter guess. Returns True if successful."""
        letter = letter.lower()
        
        # Validate the guess
        if len(letter) != 1 or not letter.isalpha() or letter in self.guessed_letters:
            return False
        
        # Record the guess
        self.guessed_letters.add(letter)
        
        if letter in self.word:
            self.correct_letters.add(letter)
            
            # Check if the player has guessed the entire word
            if all(l in self.correct_letters for l in self.word if l.isalpha()):
                self.game_over = True
                self.winner = 'Player'
        else:
            self.incorrect_guesses += 1
            
            # Check if the player has exceeded the maximum incorrect guesses
            if self.incorrect_guesses >= self.max_incorrect:
                self.game_over = True
                self.winner = 'Hangman'
        
        return True
    
    def get_current_display(self) -> str:
        """Get the current state of the word with guessed letters filled in."""
        display = ''
        for char in self.word:
            if char in self.correct_letters or not char.isalpha():
                display += char
            else:
                display += '_'
        return ' '.join(display)
    
    def get_guessed_letters(self) -> str:
        """Get a string representation of all guessed letters."""
        if not self.guessed_letters:
            return "None yet"
        return ', '.join(sorted(self.guessed_letters))
    
    def get_remaining_guesses(self) -> int:
        """Get the number of remaining incorrect guesses allowed."""
        return self.max_incorrect - self.incorrect_guesses
    
    def get_hangman_status(self) -> str:
        """Get ASCII art representation of the hangman based on incorrect guesses."""
        stages = [
            """
               ------
               |    |
               |
               |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |    |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   /
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   / \\
               |
            --------
            """
        ]
        return stages[min(self.incorrect_guesses, len(stages) - 1)]
    
    def print_board(self) -> str:
        """Return a string representation of the current game state."""
        board_str = f"Word: {self.get_current_display()}\n"
        board_str += f"Guessed letters: {self.get_guessed_letters()}\n"
        board_str += f"Incorrect guesses: {self.incorrect_guesses}/{self.max_incorrect}\n"
        board_str += f"Hangman status:\n{self.get_hangman_status()}\n"
        return board_str


def test_hangman_game():
    """Test function to verify the hangman game works correctly."""
    print("Testing HangmanGame class...")
    
    # Test with a custom word list
    custom_words = ["python", "hangman", "game"]
    game = HangmanGame(custom_words)
    
    print(f"Target word (for testing): {game.word}")
    print(f"Initial display: {game.get_current_display()}")
    print(f"Remaining incorrect guesses: {game.get_remaining_guesses()}")
    print(f"Hangman status:\n{game.get_hangman_status()}")
    
    # Test a series of guesses
    print("\nTesting letter guesses:")
    letters_to_try = ['p', 'y', 't', 'h', 'o', 'n']  # All letters in 'python'
    
    for letter in letters_to_try:
        print(f"Guessing '{letter}':")
        success = game.guess_letter(letter)
        print(f"  Success: {success}")
        print(f"  Current display: {game.get_current_display()}")
        print(f"  Guessed letters: {game.get_guessed_letters()}")
        print(f"  Remaining incorrect guesses: {game.get_remaining_guesses()}")
        if game.game_over:
            print(f"Game over! Winner: {game.winner}")
            break
        print(f"  Hangman status:\n{game.get_hangman_status()}")
    
    print(f"\nFinal word: {game.word}")
    print(f"Game over: {game.game_over}")
    print(f"Winner: {game.winner}")


if __name__ == "__main__":
    test_hangman_game()