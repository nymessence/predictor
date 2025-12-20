#!/usr/bin/env python3
"""
Connect Four Game Class
Implements a complete Connect Four game with four-in-a-row win validation and game state tracking.
"""

from typing import List, Optional, Tuple


class ConnectFourGame:
    """
    A complete Connect Four game implementation with four-in-a-row validation
    and game state tracking for integration with character interactions.
    """
    
    def __init__(self):
        """Initialize the connect four board to empty state."""
        self.board = [['.' for _ in range(7)] for _ in range(6)]  # 6 rows x 7 columns
        self.current_player = 'red'  # Red goes first
        self.game_over = False
        self.winner = None  # None, 'red', 'yellow', or 'draw'
        self.move_history = []  # Store column numbers of drops
        self.columns_height = [0 for _ in range(7)]  # Track height of each column (0-6)
        self.last_move = None  # (row, col) of last move
    
    def get_board_state(self) -> List[List[str]]:
        """Return the current board state."""
        return [row[:] for row in self.board]
    
    def get_valid_columns(self) -> List[int]:
        """Return list of columns that are not full."""
        valid_cols = []
        for col in range(7):
            if self.columns_height[col] < 6:  # Column has space
                valid_cols.append(col)
        return valid_cols
    
    def is_column_valid(self, col: int) -> bool:
        """Check if a column is valid for dropping a disc."""
        return 0 <= col < 7 and self.columns_height[col] < 6
    
    def drop_disc(self, col: int) -> bool:
        """Drop a disc in the specified column. Returns True if successful."""
        if not self.is_column_valid(col) or self.game_over:
            return False
        
        # Find the lowest available row in this column
        row = 5 - self.columns_height[col]  # 5 is the bottom row (0-indexed: 0-5), so 0th filled cell goes to row 5

        # Place the disc
        disc = 'R' if self.current_player == 'red' else 'Y'  # R for red, Y for yellow
        self.board[row][col] = disc
        self.columns_height[col] += 1
        self.last_move = (row, col)
        self.move_history.append(col)

        # Check for win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        # Check for draw (board full)
        elif all(height == 6 for height in self.columns_height):
            self.game_over = True
            self.winner = 'draw'
        else:
            # Switch player
            self.current_player = 'yellow' if self.current_player == 'red' else 'red'

        return True
    
    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move resulted in a win (four in a row)."""
        disc = self.board[row][col]
        
        # Check horizontal (left-right)
        count = 1  # Count the current disc
        # Left
        for c in range(col - 1, -1, -1):
            if self.board[row][c] == disc:
                count += 1
            else:
                break
        # Right
        for c in range(col + 1, 7):
            if self.board[row][c] == disc:
                count += 1
            else:
                break
        
        if count >= 4:
            return True
        
        # Check vertical (up-down)
        count = 1
        # Down (higher row numbers)
        for r in range(row + 1, 6):
            if self.board[r][col] == disc:
                count += 1
            else:
                break
        
        if count >= 4:
            return True
        
        # Check diagonal (top-left to bottom-right)
        count = 1
        # Up-left
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0 and self.board[r][c] == disc:
            count += 1
            r -= 1
            c -= 1
        # Down-right
        r, c = row + 1, col + 1
        while r < 6 and c < 7 and self.board[r][c] == disc:
            count += 1
            r += 1
            c += 1
        
        if count >= 4:
            return True
        
        # Check diagonal (top-right to bottom-left)
        count = 1
        # Up-right
        r, c = row - 1, col + 1
        while r >= 0 and c < 7 and self.board[r][c] == disc:
            count += 1
            r -= 1
            c += 1
        # Down-left
        r, c = row + 1, col - 1
        while r < 6 and c >= 0 and self.board[r][c] == disc:
            count += 1
            r += 1
            c -= 1
        
        if count >= 4:
            return True
        
        return False
    
    def print_board(self) -> str:
        """Return a string representation of the board."""
        board_str = "  0   1   2   3   4   5   6\n"
        for r in range(6):
            board_str += f"{r} "
            for c in range(7):
                cell = self.board[r][c]
                board_str += f"{cell} | "  # Show cell + separator
            board_str = board_str.rstrip(' | ') + '\n'  # Remove trailing separator
        return board_str
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.board = [['.' for _ in range(7)] for _ in range(6)]
        self.current_player = 'red'
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.columns_height = [0 for _ in range(7)]
        self.last_move = None


def test_connect_four_game():
    """Test function to verify the connect four game works correctly."""
    print("Testing ConnectFourGame class...")
    
    game = ConnectFourGame()
    print("\nInitial Board:")
    print(game.print_board())
    print(f"Current Player: {game.current_player}")
    print(f"Valid columns: {game.get_valid_columns()}")
    
    # Make some moves to test game mechanics
    print("\nMaking moves in column 3 (red), then column 3 (yellow), etc.")
    moves = [3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 5, 5, 3]  # This should create a vertical win
    
    for move in moves:
        print(f"Dropping {game.current_player}'s disc in column {move}")
        success = game.drop_disc(move)
        print(f"Success: {success}")
        if success:
            print(game.print_board())
            print(f"Current Player: {game.current_player}")
            if game.game_over:
                print(f"Game Over! Winner: {game.winner}")
                break
        else:
            print("Failed to drop disc")


if __name__ == "__main__":
    test_connect_four_game()