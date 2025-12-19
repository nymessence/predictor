#!/usr/bin/env python3
"""
Twenty-One Game Class (Simplified Blackjack)
Implements a simplified card game similar to Blackjack where players aim to get as close to 21 as possible without going over.
"""

import random
from typing import List, Optional, Tuple


class Card:
    """Represents a playing card."""
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        # Calculate card value for twenty-one
        if rank in ['J', 'Q', 'K']:
            self.value = 10
        elif rank == 'A':
            self.value = 11  # Aces start as 11, but can be adjusted to 1 if needed
        else:
            self.value = int(rank)
    
    def __str__(self):
        return f"{self.rank} of {self.suit}"
    
    def __repr__(self):
        return str(self)


class Deck:
    """A deck of playing cards."""
    def __init__(self):
        self.cards = []
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(suit, rank))
        
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def deal_card(self) -> Optional[Card]:
        """Deal a card from the deck."""
        if self.cards:
            return self.cards.pop()
        return None


class TwentyOneGame:
    """
    A simplified twenty-one (blackjack-style) game implementation with card dealing
    and hit/stand mechanics for integration with character interactions.
    """
    
    def __init__(self):
        """Initialize the twenty-one game."""
        self.deck = None
        self.player_hand = []
        self.dealer_hand = []
        self.game_round = 1
        self.game_over = False
        self.player_stands = False
        self.winner = None  # None, 'Player', 'Dealer', or 'draw'
        self.dealer_stands = False
        self.player_busted = False
        self.dealer_busted = False
        self.player_wins = 0
        self.dealer_wins = 0
        self.draws = 0
        self.reset_game()
    
    def reset_game(self):
        """Start a new game with fresh deck and hands."""
        self.deck = Deck()
        self.player_hand = []
        self.dealer_hand = []
        self.player_stands = False
        self.game_over = False
        self.winner = None
        self.dealer_stands = False
        self.player_busted = False
        self.dealer_busted = False
        
        # Deal initial cards
        self.player_hand.append(self.deck.deal_card())
        self.dealer_hand.append(self.deck.deal_card())
        self.player_hand.append(self.deck.deal_card())
        self.dealer_hand.append(self.deck.deal_card())
        
        # Check for natural win
        if self.get_player_score() == 21 and self.get_dealer_score() == 21:
            self.winner = 'draw'
            self.game_over = True
        elif self.get_player_score() == 21:
            self.winner = 'Player'
            self.game_over = True
        elif self.get_dealer_score() == 21:
            self.winner = 'Dealer'
            self.game_over = True
    
    def get_player_score(self) -> int:
        """Calculate the player's hand value, properly handling aces."""
        return self._calculate_score(self.player_hand)
    
    def get_dealer_score(self) -> int:
        """Calculate the dealer's hand value, properly handling aces."""
        return self._calculate_score(self.dealer_hand)
    
    def _calculate_score(self, hand: List[Card]) -> int:
        """Calculate score for a hand, adjusting aces as needed."""
        score = sum(card.value for card in hand)
        aces = sum(1 for card in hand if card.rank == 'A')
        
        # Adjust for aces if score is over 21
        while score > 21 and aces > 0:
            score -= 10  # Convert an ace from 11 to 1
            aces -= 1
        
        return score
    
    def player_hit(self) -> bool:
        """Player chooses to hit (take another card). Returns True if successful."""
        if self.game_over or self.player_stands:
            return False

        card = self.deck.deal_card()
        if card:
            self.player_hand.append(card)

            # Check if player busts
            if self.get_player_score() > 21:
                self.player_busted = True
                self.game_over = True
                self.winner = 'Dealer'

            # Check for 21
            elif self.get_player_score() == 21:
                # Player has 21, now dealer plays
                self.player_stand()

            return True
        return False
    
    def player_stand(self):
        """Player chooses to stand (end their turn)."""
        self.player_stands = True
        
        # Dealer plays according to standard rules
        self._dealer_play()
    
    def _dealer_play(self):
        """Dealer plays according to standard rules (hit on 16, stand on 17+)."""
        while not self.dealer_stands and not self.dealer_busted:
            dealer_score = self.get_dealer_score()
            
            if dealer_score < 17:
                # Dealer must hit
                card = self.deck.deal_card()
                if card:
                    self.dealer_hand.append(card)
                    
                    # Check if dealer busts
                    if self.get_dealer_score() > 21:
                        self.dealer_busted = True
                        self.game_over = True
                        self.winner = 'Player'
                else:
                    # No more cards in deck
                    self.dealer_stands = True
            elif dealer_score >= 17:
                # Dealer stands on 17+
                self.dealer_stands = True
                break
        
        # If both players stand, determine winner by comparison
        if self.player_stands and self.dealer_stands:
            self._determine_game_winner()
    
    def _determine_game_winner(self):
        """Determine the winner after both players finish their turns."""
        if self.game_over:
            return  # Already determined
            
        player_score = self.get_player_score()
        dealer_score = self.get_dealer_score()
        
        if player_score > 21:
            self.winner = 'Dealer'
        elif dealer_score > 21:
            self.winner = 'Player'
        elif player_score > dealer_score:
            self.winner = 'Player'
        elif dealer_score > player_score:
            self.winner = 'Dealer'
        else:  # Scores are equal
            self.winner = 'draw'
        
        self.game_over = True
    
    def get_player_hand_str(self) -> str:
        """Get string representation of player's hand."""
        return ', '.join([str(card) for card in self.player_hand])

    def get_dealer_hand_str(self, hide_first: bool = False) -> str:
        """Get string representation of dealer's hand."""
        if hide_first and len(self.dealer_hand) > 0:
            hidden_hand = ["[Hidden Card]"] + [str(card) for card in self.dealer_hand[1:]]
            return ', '.join(hidden_hand)
        else:
            return ', '.join([str(card) for card in self.dealer_hand])
    
    def get_game_state_info(self) -> dict:
        """Get comprehensive game state information."""
        return {
            'player_hand': self.get_player_hand_str(),
            'player_score': self.get_player_score(),
            'dealer_hand_visible': self.get_dealer_hand_str(hide_first=True),
            'dealer_hand_full': self.get_dealer_hand_str(hide_first=False),
            'dealer_score': self.get_dealer_score(),
            'game_round': self.game_round,
            'game_over': self.game_over,
            'winner': self.winner,
            'player_stands': self.player_stands,
            'dealer_stands': self.dealer_stands,
            'player_busted': self.player_busted,
            'dealer_busted': self.dealer_busted,
            'player_wins': self.player_wins,
            'dealer_wins': self.dealer_wins,
            'draws': self.draws
        }

    def reset_round(self):
        """Reset for a new round."""
        self.reset_game()
        self.game_round += 1


def test_twenty_one_game():
    """Test function to verify the twenty-one game works correctly."""
    print("Testing TwentyOneGame class...")

    game = TwentyOneGame()
    print(f"Initial player hand: {game.get_player_hand_str()}")
    print(f"Player score: {game.get_player_score()}")
    print(f"Dealer's visible card: {game.get_dealer_hand_str(hide_first=True)}")
    print(f"Dealer's full hand: {game.get_dealer_hand_str(hide_first=False)}")
    print(f"Dealer's score: {game.get_dealer_score()}")

    # Check if game is already over due to natural 21
    if game.game_over:
        print(f"Game ended due to natural 21! Winner: {game.winner}")
        return

    # Test a few player moves
    print(f"\nPlayer hits...")
    success = game.player_hit()
    print(f"Hit successful: {success}")
    print(f"Player's new hand: {game.get_player_hand_str()}")
    print(f"Player's new score: {game.get_player_score()}")

    print(f"\nPlayer stands...")
    game.player_stand()
    print(f"After player stands:")
    print(f"Dealer's full hand: {game.get_dealer_hand_str(hide_first=False)}")
    print(f"Dealer's score: {game.get_dealer_score()}")
    print(f"Game over: {game.game_over}")
    print(f"Winner: {game.winner}")

    # Test reset
    print("\nTesting round reset...")
    game.reset_round()
    print(f"Post-reset player hand: {game.get_player_hand_str()}")
    print(f"Post-reset score: {game.get_player_score()}")


if __name__ == "__main__":
    test_twenty_one_game()