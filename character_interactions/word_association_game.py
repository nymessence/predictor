#!/usr/bin/env python3
"""
Word Association Game Class
Implements a word association game where players take turns saying words related to the previous word.
"""

from typing import List, Optional
import random


class WordAssociationGame:
    """
    A word association game implementation where players take turns
    saying words related to the previous word.
    """
    
    def __init__(self):
        """Initialize the word association game."""
        self.word_chain = []  # List of words in the sequence
        self.round = 1
        self.game_over = False
        self.winner = None  # None, 'Player1', 'Player2', or 'draw'
        self.repeated_words = set()  # Track words that have already been used
        self.invalid_word_count = 0  # Count consecutive invalid words
        self.max_invalid_words = 2  # Max consecutive invalid words before game ends
        self.current_player = 'Player1'  # Start with Player1
        
    def reset_game(self):
        """Reset the game for a new round."""
        self.word_chain = []
        self.round = 1
        self.game_over = False
        self.winner = None
        self.repeated_words = set()
        self.invalid_word_count = 0
        self.current_player = 'Player1'
    
    def submit_word(self, word: str, player: str) -> bool:
        """Submit a word for the word association game. Returns True if valid."""
        word = word.strip().lower()

        if not word or not self._is_valid_word(word):
            self.invalid_word_count += 1
            if self.invalid_word_count >= self.max_invalid_words:
                self.game_over = True
                # Award win to the other player if current makes invalid submission
                self.winner = 'Player2' if player == 'Player1' else 'Player1'
            return False

        if word in self.repeated_words:
            self.invalid_word_count += 1
            if self.invalid_word_count >= self.max_invalid_words:
                self.game_over = True
                # Award win to the other player if current makes invalid submission
                self.winner = 'Player2' if player == 'Player1' else 'Player1'
            return False  # Word already used

        self.word_chain.append(word)
        self.repeated_words.add(word)
        self.invalid_word_count = 0  # Reset invalid counter after valid word

        # Switch player for next turn
        self.current_player = 'Player2' if player == 'Player1' else 'Player1'

        # Check for game end conditions
        # For now, let's say the game continues with turns, and we can end it based on rounds or other criteria
        # We could add a max word chain length as an option
        if len(self.word_chain) >= 20:  # Arbitrary max length for the word chain
            self.game_over = True
            self.winner = 'draw'  # Or could award to last player who completed a valid word

        return True
    
    def _is_valid_word(self, word: str) -> bool:
        """Check if word is valid (non-empty, contains at least one alphabetic character)."""
        # Check if it contains at least one letter and is mostly alphabetic with possible punctuation
        if len(word) == 0:
            return False
        # Check that the word has at least one alphabetic character and only contains alphanumerics or common separators
        return any(c.isalpha() for c in word) and all(c.isalnum() or c in "-' " for c in word)

    def get_last_word(self) -> Optional[str]:
        """Get the last word in the chain."""
        if self.word_chain:
            return self.word_chain[-1]
        return None

    def get_word_chain(self) -> str:
        """Get the current sequence of words."""
        return " -> ".join(self.word_chain)

    def get_current_player(self) -> str:
        """Get the current player."""
        return self.current_player

    def get_game_status(self) -> str:
        """Get current game status information."""
        status = f"Word Chain: {self.get_word_chain()}\n"
        status += f"Current Player: {self.current_player}\n"
        status += f"Words Used: {len(self.word_chain)}\n"
        status += f"Invalid Word Count: {self.invalid_word_count}\n"
        status += f"Game Over: {self.game_over}\n"
        if self.winner:
            status += f"Winner: {self.winner}\n"
        return status


def test_word_association_game():
    """Test function to verify the word association game works correctly."""
    print("Testing WordAssociationGame class...")

    game = WordAssociationGame()
    print(f"Initial status:\n{game.get_game_status()}")

    # Test a sequence of valid words
    words = ["happy", "joy", "celebration", "party", "dance", "music"]

    for i, word in enumerate(words):
        player = 'Player1' if i % 2 == 0 else 'Player2'
        print(f"\n{player} submits word: {word}")
        success = game.submit_word(word, player)
        print(f"Success: {success}")
        print(f"Status:\n{game.get_game_status()}")

        if game.game_over:
            print(f"Game ended! Winner: {game.winner}")
            break

    # Test invalid word handling
    print("\nTesting invalid word submissions after reset...")
    game.reset_game()

    print(f"After reset status:\n{game.get_game_status()}")

    print("\nSubmitting invalid word '123':")
    success = game.submit_word('123', 'Player1')
    print(f"Success: {success}")
    print(f"Status:\n{game.get_game_status()}")

    print("\nSubmitting valid word 'apple':")
    success = game.submit_word('apple', 'Player1')
    print(f"Success: {success}")
    print(f"Status:\n{game.get_game_status()}")

    print("\nSubmitting same word 'apple' again:")
    success = game.submit_word('apple', 'Player2')
    print(f"Success: {success}")
    print(f"Status:\n{game.get_game_status()}")


if __name__ == "__main__":
    test_word_association_game()