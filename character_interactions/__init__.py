"""
Dynamic AI Character Conversation System - Anti-Repetition Enhanced Edition
Modular package structure for better maintainability and AI patching
"""

__version__ = "2.0.0"
__author__ = "AI Character System"

# Import main components for easy access
from .character_loader import load_character_generic
from .response_generator import generate_response_adaptive
from .repetition_detector import detect_repetition_patterns
from .environmental_triggers import generate_environmental_trigger
from .api_client import make_api_call

__all__ = [
    'load_character_generic',
    'generate_response_adaptive',
    'detect_repetition_patterns',
    'generate_environmental_trigger',
    'make_api_call'
]
