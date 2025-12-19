#!/usr/bin/env python3
"""
Tic-Tac-Toe Game Class
Implements a complete tic-tac-toe game with win validation and game state tracking.
"""

from typing import List, Optional, Tuple


class TicTacToeGame:
    """
    A complete tic-tac-toe game implementation with win validation
    and game state tracking for integration with character interactions.
    """
    
    def __init__(self):
        """Initialize the tic-tac-toe board to empty state."""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'  # X starts
        self.game_over = False
        self.winner = None  # None, 'X', 'O', or 'draw'
        self.move_count = 0
    
    def get_board_state(self) -> List[List[str]]:
        """Return the current board state."""
        return [row[:] for row in self.board]
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves (empty positions) on the board."""
        valid_moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == ' ':
                    valid_moves.append((row, col))
        return valid_moves
    
    def is_move_valid(self, row: int, col: int) -> bool:
        """Check if a move is valid (within bounds and on empty position)."""
        if not (0 <= row < 3 and 0 <= col < 3):
            return False
        return self.board[row][col] == ' '
    
    def make_move(self, row: int, col: int) -> bool:
        """Make a move on the board. Returns True if successful."""
        if not self.is_move_valid(row, col) or self.game_over:
            return False
        
        # Make the move
        self.board[row][col] = self.current_player
        self.move_count += 1
        
        # Check for win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        # Check for draw (full board)
        elif self.move_count == 9:
            self.game_over = True
            self.winner = 'draw'
        else:
            # Switch player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return True
    
    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move resulted in a win."""
        player = self.board[row][col]
        
        # Check row
        if all(self.board[row][c] == player for c in range(3)):
            return True
        
        # Check column  
        if all(self.board[r][col] == player for r in range(3)):
            return True
        
        # Check diagonals
        if row == col and all(self.board[i][i] == player for i in range(3)):
            return True
        
        if row + col == 2 and all(self.board[i][2-i] == player for i in range(3)):
            return True
        
        return False
    
    def print_board(self) -> str:
        """Return a string representation of the board."""
        board_str = "   0   1   2\n"
        for i, row in enumerate(self.board):
            board_str += f"{i}  {row[0]} | {row[1]} | {row[2]}\n"
            if i < 2:
                board_str += "  -----------\n"
        return board_str
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        self.move_count = 0


def test_tic_tac_toe_game():
    """Test function to verify the tic-tac-toe game works correctly."""
    print("Testing TicTacToeGame class...")
    
    game = TicTacToeGame()
    print("\nInitial Board:")
    print(game.print_board())
    print(f"Current Player: {game.current_player}")
    print(f"Valid moves count: {len(game.get_valid_moves())}")
    
    # Test a game sequence
    print("\nMaking moves: (0,0), (1,1), (0,1), (1,0), (0,2)")
    moves = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 2)]
    
    for move in moves:
        success = game.make_move(*move)
        print(f"Move {move} successful: {success}")
        if success:
            print(game.print_board())
            print(f"Current Player: {game.current_player}")
        
        if game.game_over:
            print(f"Game Over! Winner: {game.winner}")
            break
    
    # Reset and test a draw
    print("\nTesting draw condition...")
    game.reset_game()
    draw_moves = [(0,0), (0,1), (0,2), (1,0), (1,2), (1,1), (2,0), (2,1), (2,2)]
    for move in draw_moves:
        success = game.make_move(*move)
        if success and game.game_over and game.winner == 'draw':
            print(f"Move {move} resulted in a draw!")
            print(game.print_board())
            break


if __name__ == "__main__":
    test_tic_tac_toe_game()