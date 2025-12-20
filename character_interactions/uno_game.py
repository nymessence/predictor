#!/usr/bin/env python3
"""
Uno Card Game Class  
Implements a simplified Uno card game with color/number matching, action cards, and win validation.
"""

import random
from typing import List, Optional, Tuple, Dict


class Card:
    """Represents a Uno card."""
    def __init__(self, color: str, value: str):
        self.color = color  # 'red', 'blue', 'green', 'yellow', 'wild'
        self.value = value  # '0', '1', '2', ..., '9', 'skip', 'reverse', '+2', '+4', 'wild'
        
        # Calculate card value for scoring (special cards)
        if value in ['skip', 'reverse', '+2']:
            self.points = 20
        elif value in ['+4', 'wild']:
            self.points = 50
        else:
            try:
                self.points = int(value)  # Number cards are worth their face value
            except ValueError:
                self.points = 0  # For other special cases
    
    def __str__(self):
        if self.color == 'wild':
            return f"{self.value.upper()}"
        else:
            return f"{self.color.title()} {self.value}"
    
    def __repr__(self):
        return str(self)
    
    def matches(self, other_card: 'Card') -> bool:
        """Check if this card matches another card (color or value)."""
        return (self.color == other_card.color or 
                self.value == other_card.value or 
                self.color == 'wild' or 
                other_card.color == 'wild')


class Deck:
    """A Uno deck of cards."""
    def __init__(self):
        self.cards = []
        colors = ['red', 'blue', 'green', 'yellow']
        values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'skip', 'reverse', '+2']
        
        # Create the regular cards (two of each card for each color, except 0 which has one of each)
        for color in colors:
            # Add one zero card for each color
            self.cards.append(Card(color, '0'))
            
            # Add two of each other card for each color
            for value in values[1:]:  # Skip '0'
                self.cards.append(Card(color, value))
                self.cards.append(Card(color, value))
        
        # Add wild cards (4 of each)
        for _ in range(4):
            self.cards.append(Card('wild', 'wild'))
            self.cards.append(Card('wild', '+4'))
        
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def deal_card(self) -> Optional[Card]:
        """Deal a card from the deck."""
        if self.cards:
            return self.cards.pop()
        return None
    
    def is_empty(self) -> bool:
        """Check if the deck is empty."""
        return len(self.cards) == 0


class UnoGame:
    """
    A simplified Uno game implementation with color/value matching,
    action cards, and win validation for integration with character interactions.
    """
    
    def __init__(self):
        """Initialize the Uno game."""
        self.deck = Deck()
        self.discard_pile = []
        self.player1_hand = []
        self.player2_hand = []
        self.current_player = 'Player1'  # Start with Player1
        self.game_direction = 1  # 1 for forward, -1 for reverse
        self.game_over = False
        self.winner = None  # None, 'Player1', 'Player2' or 'draw'
        self.special_effects = {}  # Track effects like skip, etc.
        self.top_card = None
        self.reset_game()
    
    def reset_game(self):
        """Reset the game to initial state."""
        # Reset deck and discard pile
        self.deck = Deck()
        self.discard_pile = []
        
        # Deal 7 cards to each player
        self.player1_hand = []
        self.player2_hand = []
        for _ in range(7):
            self.player1_hand.append(self.deck.deal_card())
            self.player2_hand.append(self.deck.deal_card())
        
        # Draw the first discard card (make sure it's not a special card initially)
        first_card = self.deck.deal_card()
        while first_card and first_card.value in ['skip', 'reverse', '+2', '+4', 'wild']:
            # If it's a special card, put it back and draw another
            self.deck.cards.insert(0, first_card)
            self.deck.shuffle()
            first_card = self.deck.deal_card()
        
        self.discard_pile.append(first_card)
        self.top_card = first_card
        self.current_player = 'Player1'
        self.game_direction = 1
        self.game_over = False
        self.winner = None
        self.special_effects = {}
    
    def get_current_player_hand(self) -> List[Card]:
        """Get the current player's hand."""
        return self.player1_hand if self.current_player == 'Player1' else self.player2_hand
    
    def get_other_player_hand_size(self) -> int:
        """Get the size of the other player's hand."""
        return len(self.player2_hand) if self.current_player == 'Player1' else len(self.player1_hand)
    
    def can_play_card(self, card: Card) -> bool:
        """Check if a card can be played on the top card."""
        if self.top_card is None:
            return True  # If no top card, any card can be played (shouldn't happen in normal gameplay)
        return card.matches(self.top_card)
    
    def play_card(self, card_index: int) -> bool:
        """Play a card from current player's hand. Returns True if successful."""
        player_hand = self.player1_hand if self.current_player == 'Player1' else self.player2_hand
        
        if card_index < 0 or card_index >= len(player_hand):
            return False  # Invalid card index
        
        card = player_hand[card_index]
        
        # Check if the card can be played
        if not self.can_play_card(card):
            return False
        
        # Play the card
        played_card = player_hand.pop(card_index)
        self.discard_pile.append(played_card)
        self.top_card = played_card

        # Process special card effects
        self._process_special_card(played_card)

        # Check for win (player with no cards left)
        if len(player_hand) == 0:
            self.game_over = True
            self.winner = self.current_player

        # Switch player (taking into account skip effects)
        if 'skip_next' in self.special_effects:
            del self.special_effects['skip_next']
            # Skip the next player - effectively skip to the player after next player
            # With 2 players this means it goes back to the current player, skipping the other
            self.current_player = self.current_player  # Stay with the same player (skip other player)
        else:
            # Regular player switch regardless of direction (simplified for 2-player game)
            self.current_player = 'Player2' if self.current_player == 'Player1' else 'Player1'

        return True

    def _process_special_card(self, card: Card):
        """Process any special card effects."""
        if card.value == 'skip':
            self.special_effects['skip_next'] = True
        elif card.value == 'reverse':
            # Reverse game direction
            self.game_direction *= -1
        elif card.value == '+2':
            # In simplified uno: next player draws 2 cards and their turn is skipped
            self.special_effects['draw_two_and_skip'] = True
        elif card.value == '+4':
            # Wild +4: next player draws 4 cards and their turn is skipped
            self.special_effects['draw_four_and_skip'] = True
        elif card.value == 'wild':
            # Wild card changes the color - player should specify new color (handled outside)
            pass
    
    def draw_card(self) -> bool:
        """Draw a card from the deck to current player's hand."""
        new_card = self.deck.deal_card()
        if new_card:
            if self.current_player == 'Player1':
                self.player1_hand.append(new_card)
            else:
                self.player2_hand.append(new_card)
            return True
        return False
    
    def get_hand_for_player(self, player: str) -> str:
        """Get string representation of a player's hand."""
        hand = self.player1_hand if player == 'Player1' else self.player2_hand
        return ', '.join([str(card) for card in hand])
    
    def print_top_card(self) -> str:
        """Return string representation of the top card."""
        if self.top_card:
            return f"Top card: {self.top_card}"
        else:
            return "No top card (should not happen during gameplay)"
    
    def get_game_state(self) -> Dict[str, any]:
        """Get comprehensive game state information."""
        return {
            'current_player': self.current_player,
            'top_card': str(self.top_card) if self.top_card else None,
            'current_player_hand': self.get_hand_for_player(self.current_player),
            'current_player_hand_size': len(self.get_current_player_hand()),
            'other_player_hand_size': self.get_other_player_hand_size(),
            'game_over': self.game_over,
            'winner': self.winner,
            'deck_size': len(self.deck.cards),
            'discard_pile_size': len(self.discard_pile)
        }


def test_uno_game():
    """Test function to verify the uno game works correctly."""
    print("Testing UnoGame class...")
    
    game = UnoGame()
    print(f"\nInitial Top Card: {game.print_top_card()}")
    print(f"Player1 Hand: {game.get_hand_for_player('Player1')}")
    print(f"Player2 Hand: {game.get_hand_for_player('Player2')}")
    print(f"Current Player: {game.current_player}")
    
    # Test getting game state
    state = game.get_game_state()
    print(f"\nGame state: {state}")
    
    # Find a playable card for the current player
    current_hand = game.get_current_player_hand()
    playable_cards = []
    for i, card in enumerate(current_hand):
        if game.can_play_card(card):
            playable_cards.append((i, card))
    
    if playable_cards:
        print(f"\nPlayable cards for {game.current_player}: {playable_cards}")
        idx, card = playable_cards[0]
        print(f"Playing card at index {idx}: {card}")
        success = game.play_card(idx)
        print(f"Play successful: {success}")
        print(f"New top card: {game.print_top_card()}")
        print(f"Next player: {game.current_player}")
    else:
        print(f"No playable cards for {game.current_player}")


if __name__ == "__main__":
    test_uno_game()