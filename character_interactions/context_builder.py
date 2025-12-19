"""
Context and system prompt building with anti-repetition integration
"""

from typing import Dict, List, Tuple, Optional
from utils import truncate_to_tokens, trim_history_adaptive
# FIX: Changed from 'repetition_detector' to 'repitition_detector' to match filename
from repitition_detector import generate_anti_repetition_guidance
from config import (FORMALITY_DESC, STYLE_DESC, MAX_LOREBOOK_TOKENS, 
                   MAX_HISTORY_TOKENS, DEFAULT_SIMILARITY_THRESHOLD)


def generate_system_prompt(character_name: str, persona: str, voice_analysis: Dict, 
                          private_agenda: str, scenario_context: Optional[str] = None) -> str:
    """Generate an enhanced system prompt with anti-repetition features"""
    formality_text = FORMALITY_DESC.get(voice_analysis['formality'], "balanced in formality")
    style_text = STYLE_DESC.get(voice_analysis['style'], "versatile in expression")
    
    scenario_text = f"\nCURRENT SCENARIO:\n{scenario_context}" if scenario_context else ""
    
    characteristics = ', '.join(voice_analysis['characteristics']) if voice_analysis['characteristics'] else 'authentic personality'
    
    return f"""You are {character_name}, a character with depth and complexity.

CHARACTER CORE:
"{persona[:400]}"

VOICE & STYLE:
- {formality_text}
- {style_text}
- Key traits: {characteristics}

PRIVATE MOTIVATION:
{private_agenda}
{scenario_text}

RESPONSE PRINCIPLES:
1. PROGRESSION: Every response must advance the conversation or deepen the relationship
2. SPECIFICITY: React to specific details from the previous message
3. VOICE CONSISTENCY: Match your character's established voice and style
4. VARIETY: Vary sentence structure, action descriptions, and emotional expression
5. SHOW DON'T TELL: Reveal personality through actions, reactions, and specific details
6. INTERNAL DEPTH: Share genuine thoughts and feelings that align with your character
7. ENVIRONMENTAL AWARENESS: React to environmental triggers within 1-2 responses

ANTI-REPETITION RULES (ABSOLUTE):
- NEVER repeat the same opening phrases (e.g., "I lean back", "I chuckle softly")
- NEVER copy or closely paraphrase the previous speaker's exact words
- MAXIMUM ONE action/emotional description per response (enclosed in *asterisks*)
- Vary response length dramatically (mix short 20-word responses with longer 80-word responses)
- If conversation stalls, introduce completely new elements: memories, observations, questions
- ALWAYS react to environmental triggers within 2 responses
- NEVER let your speech patterns be influenced by the other character's style
- COMPLETE SENTENCES ONLY - no sentence fragments unless for dramatic effect

You are {character_name}. Act, think, and respond as they would. Your character integrity depends on authentic, varied dialogue."""


def build_context_adaptive(current: Dict, other: Dict, history: List[Dict], 
                          lorebook_entries: Optional[List[str]] = None, 
                          repetition_data: Optional[Dict] = None, 
                          similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Tuple[str, str, str]:
    """Build adaptive context with emergency anti-repetition integration"""
    
    # Generate system prompt if not already present
    if not current.get('system_prompt'):
        system_prompt = generate_system_prompt(
            current['name'],
            current['persona'],
            current['voice_analysis'],
            current['private_agenda'],
            current.get('scenario_context')
        )
    else:
        system_prompt = current['system_prompt']
    
    # Add AGGRESSIVE anti-repetition guidance when needed
    if repetition_data and repetition_data.get('repetition_score', 0) > 0.4:
        anti_repetition_guidance = generate_anti_repetition_guidance(
            repetition_data, current['name'], other['name']
        )
        system_prompt += anti_repetition_guidance
    
    # Build lorebook context safely
    lorebook_context = ""
    if lorebook_entries and isinstance(lorebook_entries, list):
        lorebook_context = "\n".join([str(entry) for entry in lorebook_entries[:3] if entry])
        lorebook_context = truncate_to_tokens(lorebook_context, MAX_LOREBOOK_TOKENS)
    
    # Trim history with AGGRESSIVE pattern removal
    trimmed_history = trim_history_adaptive(history, MAX_HISTORY_TOKENS)
    
    # Build history text with character differentiation
    history_text = "\n".join([
        f"[Turn {i+1}] {h['name']}: {h['content']}" 
        for i, h in enumerate(trimmed_history[-12:])
        if isinstance(h, dict) and h.get('name') and h.get('content')
    ])
    
    return system_prompt, lorebook_context, history_text
