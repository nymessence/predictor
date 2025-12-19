#!/usr/bin/env python3
"""
Chess Game Class with Legal Move Validation
Implements a complete chess game with all standard rules, move validation,
and game state tracking for integration with character interactions.
"""

import copy
from typing import List, Tuple, Optional, Dict, Any


class ChessGame:
    """
    A complete chess game implementation with legal move validation
    following standard FIDE chess rules.
    """
    
    def __init__(self):
        """Initialize the chess board to starting position."""
        self.board = self._create_initial_board()
        self.current_player = 'white'  # white starts
        self.move_history = []  # List of moves in algebraic notation
        self.game_over = False
        self.winner = None  # None, 'white', 'black', or 'draw'
        self.castling_rights = {
            'white_kingside': True,
            'white_queenside': True,
            'black_kingside': True,
            'black_queenside': True
        }
        self.en_passant_target = None  # Square where en passant is possible
        self.halfmove_clock = 0  # For 50-move rule
        self.fullmove_number = 1  # Track total moves
        
    def _create_initial_board(self) -> List[List[Optional[str]]]:
        """
        Create the initial 8x8 chess board with all pieces in starting positions.
        Pieces are represented as strings: K=King, Q=Queen, R=Rook, B=Bishop, N=Knight, P=Pawn
        White pieces are uppercase, black pieces are lowercase.
        """
        board = [[None for _ in range(8)] for _ in range(8)]
        
        # Set up pawns
        for col in range(8):
            board[1][col] = 'p'  # Black pawns
            board[6][col] = 'P'  # White pawns
            
        # Set up other pieces
        back_row_black = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
        back_row_white = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        
        for col in range(8):
            board[0][col] = back_row_black[col]  # Black back row
            board[7][col] = back_row_white[col]  # White back row
            
        return board
    
    def get_piece_at(self, row: int, col: int) -> Optional[str]:
        """Get the piece at the specified position."""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None
    
    def is_white_piece(self, piece: Optional[str]) -> bool:
        """Check if the piece belongs to white player."""
        return piece is not None and piece.isupper()
    
    def is_black_piece(self, piece: Optional[str]) -> bool:
        """Check if the piece belongs to black player."""
        return piece is not None and piece.islower()
    
    def is_opponent_piece(self, piece: Optional[str], player: str) -> bool:
        """Check if the piece belongs to the opponent."""
        if piece is None:
            return False
        if player == 'white':
            return self.is_black_piece(piece)
        else:
            return self.is_white_piece(piece)
    
    def is_own_piece(self, piece: Optional[str], player: str) -> bool:
        """Check if the piece belongs to the current player."""
        if piece is None:
            return False
        if player == 'white':
            return self.is_white_piece(piece)
        else:
            return self.is_black_piece(piece)
    
    def get_king_position(self, player: str) -> Optional[Tuple[int, int]]:
        """Find the position of the king for the specified player."""
        king_char = 'K' if player == 'white' else 'k'
        
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == king_char:
                    return (row, col)
        return None
    
    def is_square_attacked(self, row: int, col: int, by_player: str) -> bool:
        """Check if the square at (row, col) is attacked by the specified player."""
        # Check for pawn attacks
        direction = -1 if by_player == 'white' else 1
        for dc in [-1, 1]:  # Diagonal directions
            attack_row, attack_col = row + direction, col + dc
            if 0 <= attack_row < 8 and 0 <= attack_col < 8:
                piece = self.board[attack_row][attack_col]
                if piece and piece.lower() == 'p' and self.is_own_piece(piece, by_player):
                    return True

        # Check for knight attacks
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        for dr, dc in knight_moves:
            attack_row, attack_col = row + dr, col + dc
            if 0 <= attack_row < 8 and 0 <= attack_col < 8:
                piece = self.board[attack_row][attack_col]
                if piece and piece.lower() == 'n' and self.is_own_piece(piece, by_player):
                    return True

        # Check for king attacks (1-square radius)
        king_moves = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        for dr, dc in king_moves:
            attack_row, attack_col = row + dr, col + dc
            if 0 <= attack_row < 8 and 0 <= attack_col < 8:
                piece = self.board[attack_row][attack_col]
                if piece and piece.lower() == 'k' and self.is_own_piece(piece, by_player):
                    return True

        # Check for sliding pieces (rook, bishop, queen, king)
        directions = [
            (-1, 0), (1, 0),  # Vertical (rook-like)
            (0, -1), (0, 1),  # Horizontal (rook-like)
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal (bishop-like)
        ]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                piece = self.board[r][c]
                if piece:
                    if self.is_own_piece(piece, by_player):
                        piece_lower = piece.lower()
                        # Rook-like movement (horizontal/vertical)
                        if dr == 0 or dc == 0:  # Horizontal or vertical
                            if piece_lower in ['r', 'q']:
                                return True
                        # Bishop-like movement (diagonal)
                        else:  # Diagonal
                            if piece_lower in ['b', 'q']:
                                return True
                    break
                r += dr
                c += dc

        return False
    
    def is_in_check(self, player: str) -> bool:
        """Check if the player is in check."""
        king_pos = self.get_king_position(player)
        if king_pos is None:
            return False
        king_row, king_col = king_pos
        
        opponent = 'white' if player == 'black' else 'black'
        return self.is_square_attacked(king_row, king_col, opponent)
    
    def _get_piece_valid_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid moves for a specific piece at (row, col)."""
        piece = self.board[row][col]
        if not piece:
            return []
            
        piece_lower = piece.lower()
        player = 'white' if piece.isupper() else 'black'
        
        moves = []
        
        if piece_lower == 'p':  # Pawn
            direction = -1 if piece.isupper() else 1  # White pawns move up, black move down
            start_row = 6 if piece.isupper() else 1
            
            # Move forward one square
            if 0 <= row + direction < 8 and self.board[row + direction][col] is None:
                moves.append((row + direction, col))
                
                # Move forward two squares from starting position
                if row == start_row and self.board[row + 2 * direction][col] is None:
                    moves.append((row + 2 * direction, col))
            
            # Captures
            for dc in [-1, 1]:  # Diagonal captures
                new_row, new_col = row + direction, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target_piece = self.board[new_row][new_col]
                    # Regular capture
                    if target_piece and self.is_opponent_piece(target_piece, player):
                        moves.append((new_row, new_col))
                    # En passant capture
                    elif (new_row, new_col) == self.en_passant_target:
                        moves.append((new_row, new_col))
        
        elif piece_lower == 'r':  # Rook
            # Horizontal and vertical movement
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    target_piece = self.board[r][c]
                    if target_piece is None:
                        moves.append((r, c))
                    elif self.is_opponent_piece(target_piece, player):
                        moves.append((r, c))
                        break
                    else:
                        break
                    r += dr
                    c += dc
        
        elif piece_lower == 'n':  # Knight
            knight_moves = [
                (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)
            ]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target_piece = self.board[new_row][new_col]
                    if target_piece is None or self.is_opponent_piece(target_piece, player):
                        moves.append((new_row, new_col))
        
        elif piece_lower == 'b':  # Bishop
            # Diagonal movement
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    target_piece = self.board[r][c]
                    if target_piece is None:
                        moves.append((r, c))
                    elif self.is_opponent_piece(target_piece, player):
                        moves.append((r, c))
                        break
                    else:
                        break
                    r += dr
                    c += dc
        
        elif piece_lower == 'q':  # Queen
            # Combination of rook and bishop movements
            directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),  # Rook-like
                (-1, -1), (-1, 1), (1, -1), (1, 1)  # Bishop-like
            ]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    target_piece = self.board[r][c]
                    if target_piece is None:
                        moves.append((r, c))
                    elif self.is_opponent_piece(target_piece, player):
                        moves.append((r, c))
                        break
                    else:
                        break
                    r += dr
                    c += dc
        
        elif piece_lower == 'k':  # King
            king_moves = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
            for dr, dc in king_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target_piece = self.board[new_row][new_col]
                    if target_piece is None or self.is_opponent_piece(target_piece, player):
                        moves.append((new_row, new_col))
            
            # Castling moves
            if not self.is_in_check(player):
                # Kingside castling
                if self.castling_rights[f'{player}_kingside']:
                    # Check if squares are empty and not attacked
                    rook_col = 7 if piece.isupper() else 7
                    king_col = 4 if piece.isupper() else 4
                    rook_pos = (row, rook_col)
                    
                    # Find the rook position properly for each side
                    rook_col = 7
                    king_col = 4
                    rook_row = 7 if piece.isupper() else 0
                    
                    if (self.board[row][5] is None and 
                        self.board[row][6] is None and
                        self.board[rook_row][rook_col] and 
                        self.board[rook_row][rook_col].lower() == 'r' and
                        self.is_own_piece(self.board[rook_row][rook_col], player)):
                        
                        # Check that squares are not attacked
                        if (not self.is_square_attacked(row, 5, player) and
                            not self.is_square_attacked(row, 6, player)):
                            moves.append((row, 6))  # Kingside castle
                
                # Queenside castling  
                if self.castling_rights[f'{player}_queenside']:
                    rook_row = 7 if piece.isupper() else 0
                    rook_col = 0
                    
                    if (self.board[row][1] is None and 
                        self.board[row][2] is None and 
                        self.board[row][3] is None and
                        self.board[rook_row][rook_col] and 
                        self.board[rook_row][rook_col].lower() == 'r' and
                        self.is_own_piece(self.board[rook_row][rook_col], player)):
                        
                        # Check that squares are not attacked
                        if (not self.is_square_attacked(row, 2, player) and
                            not self.is_square_attacked(row, 3, player)):
                            moves.append((row, 2))  # Queenside castle
        
        return moves
    
    def get_valid_moves(self, player: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all valid moves for a player in format ((from_row, from_col), (to_row, to_col))."""
        moves = []

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and self.is_own_piece(piece, player):
                    piece_moves = self._get_piece_valid_moves(row, col)
                    for move in piece_moves:
                        from_pos = (row, col)
                        to_pos = move

                        # Test if this move would leave the king in check
                        if self._would_leave_king_in_check(from_pos, to_pos, player):
                            continue

                        moves.append((from_pos, to_pos))

        return moves

    def _debug_get_all_moves_no_check_validation(self, player: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Debugging method to get all moves without check validation."""
        moves = []

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and self.is_own_piece(piece, player):
                    piece_moves = self._get_piece_valid_moves(row, col)
                    for move in piece_moves:
                        from_pos = (row, col)
                        to_pos = move
                        moves.append((from_pos, to_pos))

        return moves
    
    def _would_leave_king_in_check(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], player: str) -> bool:
        """Test if a move would leave the player's own king in check."""
        # Make a temporary move
        temp_board = [row[:] for row in self.board]
        temp_castling = self.castling_rights.copy()
        temp_en_passant = self.en_passant_target
        temp_halfmove = self.halfmove_clock

        from_row, from_col = from_pos
        to_row, to_col = to_pos
        piece = temp_board[from_row][from_col]

        # Perform the move on temp board
        captured_piece = temp_board[to_row][to_col]
        temp_board[to_row][to_col] = piece
        temp_board[from_row][from_col] = None

        # Handle special moves for temporary board
        # En passant capture
        if piece.lower() == 'p' and (to_row, to_col) == self.en_passant_target:
            # Remove the captured pawn in en passant
            pawn_row = from_row
            captured_pawn_col = to_col
            temp_board[pawn_row][captured_pawn_col] = None

        # Update castling rights for temporary board if needed
        # Reset castling rights if king or rook moves
        if piece and piece.lower() == 'k':
            if player == 'white':
                temp_castling['white_kingside'] = False
                temp_castling['white_queenside'] = False
            else:
                temp_castling['black_kingside'] = False
                temp_castling['black_queenside'] = False
        elif piece and piece.lower() == 'r':
            # Check if it's a rook that affects castling rights
            # Check the original positions of rooks
            if from_pos == (7, 0):  # White queenside rook
                if player == 'white':
                    temp_castling['white_queenside'] = False
            elif from_pos == (7, 7):  # White kingside rook
                if player == 'white':
                    temp_castling['white_kingside'] = False
            elif from_pos == (0, 0):  # Black queenside rook
                if player == 'black':
                    temp_castling['black_queenside'] = False
            elif from_pos == (0, 7):  # Black kingside rook
                if player == 'black':
                    temp_castling['black_kingside'] = False

        # See if king is in check on temp board
        test_game = copy.deepcopy(self)
        test_game.board = temp_board
        test_game.castling_rights = temp_castling
        test_game.en_passant_target = temp_en_passant  # This might need to be updated differently

        return test_game.is_in_check(player)
    
    def is_move_legal(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], player: str) -> bool:
        """Check if a move is legal for the current player."""
        if self.game_over:
            return False
        
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Check if positions are valid
        if not (0 <= from_row < 8 and 0 <= from_col < 8 and 0 <= to_row < 8 and 0 <= to_col < 8):
            return False
        
        # Check if there's a piece at the starting position
        piece = self.board[from_row][from_col]
        if not piece:
            return False
        
        # Check if it's the right player's piece
        if not self.is_own_piece(piece, player):
            return False
        
        # Check if destination has own piece
        dest_piece = self.board[to_row][to_col]
        if dest_piece and self.is_own_piece(dest_piece, player):
            return False
        
        # Get piece-specific valid moves
        valid_moves = self._get_piece_valid_moves(from_row, from_col)
        if to_pos not in valid_moves:
            return False
        
        # Check if this move would leave own king in check
        if self._would_leave_king_in_check(from_pos, to_pos, player):
            return False
        
        return True
    
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Make a move on the board. Returns True if successful."""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Validate the move
        if not self.is_move_legal(from_pos, to_pos, self.current_player):
            return False
        
        # Get the piece to move
        piece = self.board[from_row][from_col]
        captured_piece = self.board[to_row][to_col]
        
        # Handle special moves
        piece_moved = piece
        move_notation = self._get_algebraic_notation(from_pos, to_pos, captured_piece)
        
        # Handle castling
        if piece.lower() == 'k' and abs(from_col - to_col) == 2:
            # This is a castling move
            if to_col == 6:  # Kingside
                # Move the rook: from (row, 7) to (row, 5)
                rook_pos = (from_row, 7)
                new_rook_pos = (from_row, 5)
                self.board[from_row][5] = self.board[from_row][7]
                self.board[from_row][7] = None
            elif to_col == 2:  # Queenside
                # Move the rook: from (row, 0) to (row, 3)
                self.board[from_row][3] = self.board[from_row][0]
                self.board[from_row][0] = None
        
        # Handle en passant capture
        elif piece.lower() == 'p' and (to_row, to_col) == self.en_passant_target:
            # Capture the pawn that moved two squares
            pawn_row = from_row
            captured_pawn_col = to_col
            self.board[pawn_row][captured_pawn_col] = None  # Remove the captured pawn
        
        # Update the board
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        
        # Update castling rights
        if piece_moved.lower() == 'k':
            # King moved - lose castling rights
            if self.current_player == 'white':
                self.castling_rights['white_kingside'] = False
                self.castling_rights['white_queenside'] = False
            else:
                self.castling_rights['black_kingside'] = False
                self.castling_rights['black_queenside'] = False
        elif piece_moved.lower() == 'r':
            # Check if it was a rook that had castling rights
            if from_pos == (7, 0):  # White queenside rook
                self.castling_rights['white_queenside'] = False
            elif from_pos == (7, 7):  # White kingside rook
                self.castling_rights['white_kingside'] = False
            elif from_pos == (0, 0):  # Black queenside rook
                self.castling_rights['black_queenside'] = False
            elif from_pos == (0, 7):  # Black kingside rook
                self.castling_rights['black_kingside'] = False
        
        # Handle pawn promotion - automatically promote to queen for simplicity
        if piece_moved.lower() == 'p':
            if (piece_moved.isupper() and to_row == 0) or (piece_moved.islower() and to_row == 7):
                # Promote to queen
                self.board[to_row][to_col] = 'Q' if piece_moved.isupper() else 'q'
        
        # Set en passant target
        self.en_passant_target = None
        if (piece_moved.lower() == 'p' and 
            abs(from_row - to_row) == 2):  # Pawn moved two squares
            # En passant is possible on the next move to the square behind
            ep_row = (from_row + to_row) // 2
            self.en_passant_target = (ep_row, from_col)
        
        # Update move history and counters
        self.move_history.append(move_notation)
        self.halfmove_clock += 1
        
        # If a pawn was moved or a piece was captured, reset halfmove clock
        if piece_moved.lower() == 'p' or captured_piece is not None:
            self.halfmove_clock = 0
        
        # Switch players
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        if self.current_player == 'white':
            self.fullmove_number += 1
        
        # Check for game end conditions
        self._check_game_end()
        
        return True
    
    def _get_algebraic_notation(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], captured_piece: Optional[str]) -> str:
        """Convert a move to algebraic notation."""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        piece = self.board[to_row][to_col]  # After the move, the piece that moved is at to_pos
        # But we need the piece that was at from_pos, so get it directly
        piece = self.get_piece_at(from_pos[0], from_pos[1])
        
        # Column letters (a-h)
        col_names = 'abcdefgh'
        # Row numbers (1-8, but board index is 0-7)
        from_square = col_names[from_col] + str(8 - from_row)
        to_square = col_names[to_col] + str(8 - to_row)
        
        # Special handling for different piece types
        if piece.lower() == 'p':  # Pawn
            if captured_piece:  # Pawn capture
                notation = col_names[from_col] + 'x' + to_square
                # Check for promotion
                if (piece.isupper() and to_row == 0) or (piece.islower() and to_row == 7):
                    notation += '=Q'  # Auto-promote to queen
                return notation
            else:  # Regular pawn move
                notation = to_square
                # Check for promotion
                if (piece.isupper() and to_row == 0) or (piece.islower() and to_row == 7):
                    notation += '=Q'  # Auto-promote to queen
                return notation
        else:
            # Non-pawn pieces
            piece_symbol = piece.upper() if piece.lower() != 'k' else 'K'  # King symbol is K
            if captured_piece:
                return piece_symbol + 'x' + to_square
            else:
                return piece_symbol + to_square
    
    def _check_game_end(self):
        """Check if the game has ended and update game state."""
        # Check if current player has any valid moves
        valid_moves = self.get_valid_moves(self.current_player)
        
        if not valid_moves:
            # No moves available - check if in check (checkmate) or not (stalemate)
            if self.is_in_check(self.current_player):
                # Checkmate - current player loses
                self.game_over = True
                self.winner = 'black' if self.current_player == 'white' else 'white'
            else:
                # Stalemate - draw
                self.game_over = True
                self.winner = 'draw'
        elif self.halfmove_clock >= 100:
            # 50-move rule (100 half-moves) - draw
            self.game_over = True
            self.winner = 'draw'
        else:
            # Check for insufficient material (simplified check)
            pieces = []
            for row in range(8):
                for col in range(8):
                    piece = self.board[row][col]
                    if piece:
                        pieces.append(piece)
            
            # Kings only - draw
            if len(pieces) == 2:
                self.game_over = True
                self.winner = 'draw'
            # King + minor piece vs king - draw
            elif len(pieces) == 3:
                piece_types = [p.lower() for p in pieces if p.lower() != 'k']
                if piece_types and all(pt in ['b', 'n'] for pt in piece_types):
                    self.game_over = True
                    self.winner = 'draw'
    
    def get_board_fen(self) -> str:
        """Get current board position in Forsyth-Edwards Notation."""
        fen_rows = []
        for row in self.board:
            fen_row = ""
            empty_count = 0
            for piece in row:
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += piece
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        
        # Simplified FEN - just the board part
        return "/".join(fen_rows)
    
    def print_board(self) -> str:
        """Return a string representation of the board."""
        board_str = "  a b c d e f g h\n"
        for i, row in enumerate(self.board):
            board_str += f"{8-i} "
            for piece in row:
                if piece:
                    board_str += piece + " "
                else:
                    board_str += ". "
            board_str += f"{8-i}\n"
        board_str += "  a b c d e f g h"
        return board_str
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.board = self._create_initial_board()
        self.current_player = 'white'
        self.move_history = []
        self.game_over = False
        self.winner = None
        self.castling_rights = {
            'white_kingside': True,
            'white_queenside': True,
            'black_kingside': True,
            'black_queenside': True
        }
        self.en_passant_target = None
        self.halfmove_clock = 0
        self.fullmove_number = 1


def test_chess_game():
    """Test function to verify the chess game works correctly."""
    print("Testing ChessGame class...")
    
    # Test initial board setup
    game = ChessGame()
    print("\nInitial Board:")
    print(game.print_board())
    print(f"Current Player: {game.current_player}")
    print(f"White's valid moves count: {len(game.get_valid_moves('white'))}")

    # For debugging
    print(f"White's moves without check validation: {len(game._debug_get_all_moves_no_check_validation('white'))}")

    white_king_pos = game.get_king_position('white')
    black_king_pos = game.get_king_position('black')
    print(f"White king position: {white_king_pos}")
    print(f"Black king position: {black_king_pos}")
    print(f"Is white in check: {game.is_in_check('white')}")
    print(f"Is black in check: {game.is_in_check('black')}")
    
    # Test basic pawn move
    print("\nMaking move: e2 to e4 (white)")
    success = game.make_move((6, 4), (4, 4))  # e2 to e4
    print(f"Move successful: {success}")
    print(game.print_board())
    print(f"Current Player: {game.current_player}")
    
    # Test black response
    print("\nMaking move: e7 to e5 (black)")
    success = game.make_move((1, 4), (3, 4))  # e7 to e5
    print(f"Move successful: {success}")
    print(game.print_board())
    print(f"Current Player: {game.current_player}")
    print(f"Move History: {game.move_history}")
    
    # Test invalid move
    print("\nTesting invalid move: e4 to e6 (should fail - not a legal move)")
    success = game.make_move((4, 4), (2, 4))  # e4 to e6 (pawn can't jump)
    print(f"Move successful: {success}")
    
    # Test capturing
    print("\nMaking moves to set up capture...")
    game.reset_game()
    # 1. e4
    game.make_move((6, 4), (4, 4))
    # 2. e5
    game.make_move((1, 4), (3, 4))
    # 3. f4
    game.make_move((6, 5), (4, 5))
    # 4. d5
    game.make_move((1, 3), (3, 3))
    # 5. exd5 (capture)
    success = game.make_move((4, 4), (3, 3))
    print(f"Capture move successful: {success}")
    print("Board after capture e.p.:")
    print(game.print_board())
    print(f"Move History: {game.move_history}")
    
    # Test check detection
    print("\nSetting up check position...")
    game.reset_game()
    # Reset and set up a simpler test
    # Clear the board first
    for r in range(8):
        for c in range(8):
            game.board[r][c] = None
    # Put kings in safe positions
    game.board[7][4] = 'K'  # White king at e1
    game.board[0][3] = 'k'  # Black king at d8 (different from white king's position)
    # Put a black queen that could threaten white king
    game.board[1][4] = 'q'  # Black queen at e7
    print("Board with potential check:")
    print(game.print_board())
    print(f"Is white in check: {game.is_in_check('white')}")
    print(f"Is black in check: {game.is_in_check('black')}")
    
    print("\nChess game tests completed!")


if __name__ == "__main__":
    test_chess_game()