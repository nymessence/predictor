#!/usr/bin/env python3
"""
Number Guessing Game Class
Implements a number guessing game where players try to guess a secret number with feedback.
"""

import random
from typing import Optional


class NumberGuessingGame:
    """
    A number guessing game implementation where players try to guess
    a secret number with high/low feedback.
    """
    
    def __init__(self, min_num: int = 1, max_num: int = 100):
        """Initialize the number guessing game."""
        self.min_num = min_num
        self.max_num = max_num
        self.secret_number = random.randint(min_num, max_num)
        self.attempts = 0
        self.max_attempts = 10  # Limit attempts for challenge
        self.game_over = False
        self.winner = None  # None, 'Player1', 'Player2', or 'draw' (if nobody guesses in max attempts)
        self.last_guess = None
        self.last_feedback = None  # Will be 'too high', 'too low', or 'correct'
        
    def reset_game(self):
        """Reset the game for a new round."""
        self.secret_number = random.randint(self.min_num, self.max_num)
        self.attempts = 0
        self.game_over = False
        self.winner = None
        self.last_guess = None
        self.last_feedback = None
    
    def make_guess(self, number: int) -> str:
        """Process a number guess and return feedback."""
        if self.game_over:
            return "Game is already over."
        
        self.attempts += 1
        self.last_guess = number
        
        if number == self.secret_number:
            self.game_over = True
            self.last_feedback = "correct"
            return f"Correct! The number was {self.secret_number}. It took {self.attempts} attempts."
        elif number < self.secret_number:
            self.last_feedback = "too low"
            if self.attempts >= self.max_attempts:
                self.game_over = True
                self.winner = "draw"  # Nobody guessed correctly
                return f"Too low! The number was {self.secret_number}. Maximum attempts reached, game over."
            return f"Too low! The secret number is greater than {number}. Attempts remaining: {self.max_attempts - self.attempts}"
        else:  # number > self.secret_number
            self.last_feedback = "too high"
            if self.attempts >= self.max_attempts:
                self.game_over = True
                self.winner = "draw"  # Nobody guessed correctly
                return f"Too high! The number was {self.secret_number}. Maximum attempts reached, game over."
            return f"Too high! The secret number is less than {number}. Attempts remaining: {self.max_attempts - self.attempts}" 
    
    def get_game_status(self) -> str:
        """Get current game status."""
        status = f"Secret number is between {self.min_num} and {self.max_num}\n"
        status += f"Attempts used: {self.attempts}/{self.max_attempts}\n"
        if self.last_guess is not None:
            status += f"Last guess: {self.last_guess} ({self.last_feedback})\n"
        return status
    
    def is_valid_guess(self, guess: str) -> bool:
        """Check if a guess is a valid integer within range."""
        try:
            num = int(guess)
            return self.min_num <= num <= self.max_num
        except ValueError:
            return False


def test_number_guessing_game():
    """Test function to verify the number guessing game works correctly."""
    print("Testing NumberGuessingGame class...")
    
    game = NumberGuessingGame(1, 10)  # Small range for testing
    print(f"Secret number is between {game.min_num} and {game.max_num}")
    print(f"Max attempts: {game.max_attempts}")
    print(f"Secret number (for testing): {game.secret_number}")
    
    # Test some guesses
    test_guesses = [5, 8, 7, 6]
    for i, guess in enumerate(test_guesses, 1):
        print(f"\nGuess {i}: {guess}")
        feedback = game.make_guess(guess)
        print(f"Feedback: {feedback}")
        print(f"Game over: {game.game_over}")
        
        if game.game_over:
            print(f"Winner: {game.winner}")
            break


if __name__ == "__main__":
    test_number_guessing_game()