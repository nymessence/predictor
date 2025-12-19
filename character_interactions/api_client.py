"""
API client with retry logic and error handling
"""

import time
from typing import Optional, List
from openai import OpenAI
from config import BASE_URL, API_KEY, MODEL_NAME, API_ERROR_MAX_RETRIES, DELAY_SECONDS


def get_client():
    """Get OpenAI client with current configuration"""
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def make_api_call(prompt: str, max_tokens: int = 150, temperature: float = 0.7, 
                 stop: Optional[List[str]] = None, verbose: bool = False) -> str:
    """Generic API call with delay and comprehensive error handling"""
    client = get_client()
    
    for attempt in range(API_ERROR_MAX_RETRIES):
        try:
            if verbose and attempt > 0:
                print(f"🔄 API retry attempt {attempt + 1}/{API_ERROR_MAX_RETRIES}")
            
            resp = client.chat.completions.create(
                model=MODEL_NAME,
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
