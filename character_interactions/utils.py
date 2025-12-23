"""
API client with retry logic and error handling
Plus utility functions for text processing
"""

import time
import re
from typing import Optional, List, Dict
from openai import OpenAI
from difflib import SequenceMatcher
from config import API_ERROR_MAX_RETRIES, DELAY_SECONDS  # Only import fixed configs
from api_client import get_client, BASE_URL, API_KEY, MODEL_NAME


# Client will be obtained dynamically using get_client()


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculates the similarity ratio between two strings using difflib's SequenceMatcher.
    Returns a float between 0.0 (no match) and 1.0 (perfect match).
    """
    # Compare strings after converting to lowercase
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count (approximately 4 characters per token for English)
    """
    if not text:
        return 0
    # More accurate estimation: ~0.75 tokens per word
    word_count = len(text.split())
    char_count = len(text)
    # Use whichever gives a more conservative (higher) estimate
    return max(int(word_count * 0.75), int(char_count / 4))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens
    """
    if not text:
        return ""
    
    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text
    
    # Calculate approximate character limit
    ratio = max_tokens / current_tokens
    char_limit = int(len(text) * ratio * 0.95)  # 0.95 for safety margin
    
    truncated = text[:char_limit]
    
    # Try to end at a sentence boundary
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    boundary = max(last_period, last_newline)
    if boundary > char_limit * 0.7:  # Only use boundary if it's not too far back
        truncated = truncated[:boundary + 1]
    
    return truncated.strip()


def trim_history_adaptive(history: List[Dict], max_tokens: int) -> List[Dict]:
    """
    Adaptively trim conversation history to fit within token budget
    Keeps most recent messages and removes repetitive content
    """
    if not history:
        return []
    
    # Always keep the last 6 messages
    min_keep = min(6, len(history))
    recent_history = history[-min_keep:]
    
    # Calculate tokens for recent messages
    total_tokens = sum(estimate_tokens(str(msg.get('content', ''))) 
                      for msg in recent_history if isinstance(msg, dict))
    
    if total_tokens <= max_tokens:
        # If we have room, try to add more messages from earlier
        remaining_tokens = max_tokens - total_tokens
        extended_history = []
        
        for msg in reversed(history[:-min_keep]):
            if not isinstance(msg, dict):
                continue
            msg_tokens = estimate_tokens(str(msg.get('content', '')))
            if msg_tokens <= remaining_tokens:
                extended_history.insert(0, msg)
                remaining_tokens -= msg_tokens
            else:
                break
        
        return extended_history + recent_history
    
    # If recent messages exceed budget, truncate them
    trimmed = []
    token_budget = max_tokens
    
    for msg in reversed(recent_history):
        if not isinstance(msg, dict):
            continue
        
        content = str(msg.get('content', ''))
        msg_tokens = estimate_tokens(content)
        
        if msg_tokens <= token_budget:
            trimmed.insert(0, msg)
            token_budget -= msg_tokens
        else:
            # Truncate this message to fit
            truncated_content = truncate_to_tokens(content, token_budget)
            if truncated_content:
                trimmed.insert(0, {
                    'name': msg.get('name', 'Unknown'),
                    'content': truncated_content
                })
            break
    
    return trimmed


def remove_repetitive_phrases(text: str, history: List[Dict], threshold: float = 0.7) -> str:
    """
    Remove phrases from text that are too similar to recent history
    """
    if not history or not text:
        return text
    
    # Get recent content
    recent_contents = [str(msg.get('content', '')) for msg in history[-5:] 
                      if isinstance(msg, dict)]
    
    # Split text into sentences
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    filtered_sentences = []
    for sentence in sentences:
        # Check if this sentence is too similar to any recent content
        is_repetitive = False
        for recent in recent_contents:
            if calculate_similarity(sentence, recent) > threshold:
                is_repetitive = True
                break
        
        if not is_repetitive:
            filtered_sentences.append(sentence)
    
    result = ' '.join(filtered_sentences)
    
    # If we filtered out everything, return the original
    return result if result.strip() else text


def make_api_call(prompt: str, max_tokens: int = 150, temperature: float = 0.7,
                 stop: Optional[List[str]] = None, verbose: bool = False) -> str:
    """Generic API call with delay and comprehensive error handling"""
    for attempt in range(API_ERROR_MAX_RETRIES):
        try:
            if verbose and attempt > 0:
                print(f"🔄 API retry attempt {attempt + 1}/{API_ERROR_MAX_RETRIES}")

            # Get the current client with updated configuration
            client = get_client()
            resp = client.chat.completions.create(
                model=MODEL_NAME,  # Use runtime model name
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )
            content = resp.choices[0].message.content.strip()

            # Apply delay after successful API call
            if DELAY_SECONDS > 0:
                if verbose:
                    print(f"⏳ Waiting {DELAY_SECONDS} seconds after API call...")
                time.sleep(DELAY_SECONDS)

            return content

        except Exception as e:
            error_msg = str(e)[:150]
            print(f"⚠️  API error (attempt {attempt + 1}): {error_msg}")
            if attempt < API_ERROR_MAX_RETRIES - 1:
                time.sleep(1.5 ** attempt)
            else:
                raise

    return "I'm having trouble responding right now. Please try again."
