#!/usr/bin/env python3
"""
Rock-Paper-Scissors Game Class
Implements a complete rock-paper-scissors game with win validation and game state tracking.
"""

from typing import Optional, Tuple
import random


class RockPaperScissorsGame:
    """
    A complete rock-paper-scissors game implementation with win validation
    and game state tracking for integration with character interactions.
    """
    
    def __init__(self):
        """Initialize the rock-paper-scissors game."""
        self.player1_choice = None  # None, 'rock', 'paper', or 'scissors'
        self.player2_choice = None
        self.game_round = 1
        self.round_over = False
        self.winner = None  # None, 'Player1', 'Player2', or 'draw'
        self.choice_history = []  # List of (player1_choice, player2_choice) tuples
        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0
        self.valid_choices = {'rock', 'paper', 'scissors', 'r', 'p', 's'}
        self.choice_map = {
            'r': 'rock', 'rock': 'rock',
            'p': 'paper', 'paper': 'paper', 
            's': 'scissors', 'scissors': 'scissors'
        }
    
    def is_valid_choice(self, choice: str) -> bool:
        """Check if a choice is valid."""
        return choice.lower() in self.valid_choices
    
    def normalize_choice(self, choice: str) -> Optional[str]:
        """Normalize choice to full form."""
        normalized = choice.lower()
        if normalized in self.choice_map:
            return self.choice_map[normalized]
        return None
    
    def make_choice(self, player: str, choice: str) -> bool:
        """Record a player's choice. Returns True if successful."""
        normalized_choice = self.normalize_choice(choice)
        if not normalized_choice:
            return False  # Invalid choice
        
        if player == 'Player1':
            self.player1_choice = normalized_choice
        elif player == 'Player2':
            self.player2_choice = normalized_choice
        else:
            return False  # Invalid player
        
        # Check if both players have made choices
        if self.player1_choice and self.player2_choice:
            self._determine_round_winner()
            self.choice_history.append((self.player1_choice, self.player2_choice))
            return True
        
        return True  # Choice recorded, waiting for other player
    
    def _determine_round_winner(self):
        """Determine the winner of the round."""
        if not (self.player1_choice and self.player2_choice):
            return
        
        p1 = self.player1_choice
        p2 = self.player2_choice
        
        if p1 == p2:
            self.winner = 'draw'
            self.draws += 1
        elif (p1 == 'rock' and p2 == 'scissors') or \
             (p1 == 'paper' and p2 == 'rock') or \
             (p1 == 'scissors' and p2 == 'paper'):
            self.winner = 'Player1'
            self.player1_wins += 1
        else:
            self.winner = 'Player2'
            self.player2_wins += 1
        
        self.round_over = True
    
    def get_result_message(self) -> str:
        """Get a message describing the result."""
        if not self.round_over:
            return "Round is not complete yet."
        
        if self.winner == 'draw':
            return f"Round #{self.game_round} was a draw! Both players chose {self.player1_choice}."
        else:
            winner_name = "Player 1" if self.winner == 'Player1' else "Player 2"
            choice = self.player1_choice if self.winner == 'Player1' else self.player2_choice
            loser_choice = self.player2_choice if self.winner == 'Player1' else self.player1_choice
            return f"{winner_name} wins round #{self.game_round}! {winner_name} played {choice}, beating {loser_choice}."
    
    def get_game_stats(self) -> str:
        """Get game statistics."""
        return f"Stats - Player 1 Wins: {self.player1_wins}, Player 2 Wins: {self.player2_wins}, Draws: {self.draws}"
    
    def reset_round(self):
        """Reset for the next round."""
        self.player1_choice = None
        self.player2_choice = None
        self.round_over = False
        self.winner = None
        self.game_round += 1
    
    def reset_game(self):
        """Reset the entire game."""
        self.player1_choice = None
        self.player2_choice = None
        self.game_round = 1
        self.round_over = False
        self.winner = None
        self.choice_history = []
        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0


def test_rock_paper_scissors_game():
    """Test function to verify the rock-paper-scissors game works correctly."""
    print("Testing RockPaperScissorsGame class...")
    
    game = RockPaperScissorsGame()
    print(f"Initial stats: {game.get_game_stats()}")
    
    # Test a round
    print("\nRound 1: Player 1 chooses 'rock', Player 2 chooses 'scissors'")
    game.make_choice('Player1', 'rock')
    game.make_choice('Player2', 'scissors')
    
    print(f"Game over: {game.round_over}")
    print(f"Winner: {game.winner}")
    print(f"Result: {game.get_result_message()}")
    print(f"Stats: {game.get_game_stats()}")
    
    # Test draw
    print("\nRound 2: Reset and test a draw")
    game.reset_round()
    print(f"New round: {game.game_round}")
    
    print("Player 1 chooses 'paper', Player 2 chooses 'paper'")
    game.make_choice('Player1', 'paper')
    game.make_choice('Player2', 'paper')
    
    print(f"Game over: {game.round_over}")
    print(f"Winner: {game.winner}")
    print(f"Result: {game.get_result_message()}")
    print(f"Stats: {game.get_game_stats()}")


if __name__ == "__main__":
    test_rock_paper_scissors_game()