"""
Configuration constants for the Dynamic AI Character Conversation System
"""

import os

# --- API Configuration ---
API_KEY = os.environ.get("AICHAT_API_KEY")
BASE_URL = "https://api.llm7.io/v1"

# --- Model Configuration ---
MODEL_NAME = "magistral-small-2509"
API_ERROR_MAX_RETRIES = 999

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")  # Separate API key for OpenRouter

# Make these available for global updates
__all__ = ['API_KEY', 'BASE_URL', 'MODEL_NAME', 'API_ERROR_MAX_RETRIES',
           'MAX_TURNS', 'DELAY_SECONDS', 'DEFAULT_SIMILARITY_THRESHOLD',
           'CRITICAL_REPETITION_THRESHOLD', 'EMERGENCY_REPETITION_THRESHOLD',
           'OPENROUTER_API_KEY']

# --- Context Window Management ---
MODEL_CONTEXT_WINDOW = 32000
GENERATION_BUFFER = 1024
MAX_TOTAL_CONTEXT = MODEL_CONTEXT_WINDOW - GENERATION_BUFFER
MAX_SYSTEM_TOKENS = 2048
MAX_LOREBOOK_TOKENS = 8198
MAX_HISTORY_TOKENS = 12288

# --- Conversation Configuration ---
MAX_TURNS = 100
DELAY_SECONDS = 3  # Reduced from 40 to 3 seconds for faster game progression
DEFAULT_SIMILARITY_THRESHOLD = 0.45
CRITICAL_REPETITION_THRESHOLD = 0.75
EMERGENCY_REPETITION_THRESHOLD = 0.9
MAX_ACTION_DESCRIPTIONS = 1

# --- Voice Analysis Descriptions ---
FORMALITY_DESC = {
    "very formal": "highly formal and regal in speech",
    "formal": "formal and proper in speech",
    "neutral": "balanced and natural in speech",
    "casual": "casual and conversational in speech",
    "very casual": "very informal and relaxed in speech"
}

STYLE_DESC = {
    "direct": "direct and to the point",
    "descriptive": "rich in descriptive language",
    "poetic": "poetic and metaphorical",
    "humorous": "witty and humorous",
    "philosophical": "thoughtful and introspective",
    "conversational": "natural and flowing in dialogue"
}
