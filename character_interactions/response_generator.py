#!/usr/bin/env python3
"""
Response generation with adaptive anti-repetition strategies
"""

import random
from typing import Dict, List, Optional
from openai import OpenAI
from config import DELAY_SECONDS  # Only import fixed config items
from context_builder import build_context_adaptive
from repitition_detector import detect_repetition_patterns
from environmental_triggers import generate_environmental_trigger
from character_loader import extract_lorebook_entries
from api_client import get_client, update_config
from scenario_adapter import enforce_scenario_constraints, check_scenario_consistency


def generate_response_adaptive(current: Dict, other: Dict, history: List[Dict], 
                              turn: int, enable_environmental: bool = True,
                              similarity_threshold: float = 0.45, 
                              verbose: bool = False, scenario_context: Optional[str] = None,
                              lorebook_entries: Optional[List[str]] = None) -> str:
    """Generate character response with adaptive anti-repetition strategies"""
    
    # Detect repetition patterns
    repetition_data = detect_repetition_patterns(history, similarity_threshold)
    repetition_score = repetition_data.get('repetition_score', 0.0)
    
    if verbose:
        print(f"📊 Repetition score: {repetition_score:.2f}")
        if repetition_data.get('issues'):
            print(f"⚠️  Issues: {', '.join(repetition_data['issues'][:2])}")
    
    # Extract lorebook entries
    if lorebook_entries is None:
        lorebook_entries = extract_lorebook_entries(
            current.get('raw_data', {}), 
            history[-5:] if len(history) >= 5 else history
        )
    
    # Build context with anti-repetition guidance
    system_prompt, lorebook_context, history_text = build_context_adaptive(
        current, other, history, lorebook_entries, repetition_data, similarity_threshold
    )
    
    # Add scenario context and enforce scenario constraints if provided
    if scenario_context:
        # Add scenario context to system prompt
        scenario_text = f"\nCUSTOM SCENARIO CONTEXT:\n{scenario_context}"
        system_prompt += scenario_text

        # SPECIAL ENFORCING: If scenario involves a structured game, emphasize JSON requirements
        game_modes = ['chess', 'tic-tac-toe', 'hangman', 'twenty-one', 'rock-paper-scissors', 'connect-four', 'uno', 'number guessing', 'word association']
        if any(mode in scenario_context.lower() for mode in game_modes):
            json_enforcement = """
            CRITICAL: You are in a structured game mode. You MUST respond in the required JSON format.
            For chess: {"dialogue": "your thoughts", "move": "e4", "board_state": "visualization after move"}
            For tic-tac-toe: {"dialogue": "your thoughts", "move": "[row, col]", "board_state": "visualization after move"}
            For other games: follow the specific JSON structure required for that game.
            FAILURE TO USE PROPER JSON FORMAT WILL RESULT IN TURN ADVANCEMENT WITHOUT YOUR MOVE BEING PROCESSED.
            """
            system_prompt += json_enforcement

        # Enforce scenario constraints
        system_prompt = enforce_scenario_constraints(system_prompt, scenario_context, current['name'])
    
    # Environmental trigger injection
    environmental_trigger = ""
    if enable_environmental and turn > 3 and random.random() < 0.15:
        environmental_trigger = generate_environmental_trigger(
            current, other, current.get('scenario_context'), history
        )
        if verbose:
            print(f"🌪️  Environmental trigger: {environmental_trigger[:50]}...")
    
    # Build the prompt
    prompt_parts = [system_prompt]
    
    if lorebook_context:
        prompt_parts.append(f"\n### RELEVANT LORE ###\n{lorebook_context}")
    
    prompt_parts.append(f"\n### CONVERSATION HISTORY ###\n{history_text}")
    
    if environmental_trigger:
        prompt_parts.append(f"\n### ENVIRONMENTAL EVENT ###\n{environmental_trigger}")
    
    # Get last message for context
    last_message = ""
    if history:
        last_msg = history[-1]
        if isinstance(last_msg, dict) and last_msg.get('content'):
            last_message = last_msg['content']
    
    prompt_parts.append(f"\n### RESPOND AS {current['name'].upper()} ###")
    prompt_parts.append(f"Previous message: {last_message}")
    prompt_parts.append(f"\nYour response (as {current['name']}):")
    
    full_prompt = "\n".join(prompt_parts)
    
    # Adjust temperature based on repetition score
    temperature = 0.7
    if repetition_score > 0.6:
        temperature = min(0.95, 0.7 + (repetition_score * 0.4))
        if verbose:
            print(f"🔥 Increased temperature to {temperature:.2f}")
    
    # Make API call
    try:
        client = get_client()
        # Import the runtime model name from api_client
        from api_client import MODEL_NAME as RUNTIME_MODEL_NAME

        resp = client.chat.completions.create(
            model=RUNTIME_MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=350,
            temperature=temperature,
            stop=[f"\n{other['name']}:", "\n\n", "[Turn"]
        )
        
        content = resp.choices[0].message.content.strip()
        
        # Clean up response
        content = content.replace(f"{current['name']}:", "").strip()
        content = content.replace(f"[{current['name']}]", "").strip()
        
        # Emergency validation
        if len(content.split()) < 5:
            if verbose:
                print("⚠️  Response too short, triggering emergency response")
            return generate_emergency_response(current, other, history, repetition_data, turn)
        
        # Scenario consistency check if scenario is provided
        if scenario_context and turn > 1:  # Skip check for first turn (greeting)
            consistency_check = check_scenario_consistency(content, scenario_context, current['name'])
            if verbose:
                print(f"🔍 Scenario consistency: {'✅' if consistency_check['consistent'] else '❌'}")
                if consistency_check['elements_referenced']:
                    print(f"   Elements referenced: {', '.join(consistency_check['elements_referenced'][:2])}")
                if consistency_check['issues']:
                    print(f"   Issues: {', '.join(consistency_check['issues'][:2])}")
            
            # If response is inconsistent with scenario, try to fix it
            if not consistency_check['consistent'] and consistency_check['issues']:
                if verbose:
                    print("🔄 Attempting to fix scenario inconsistency...")
                try:
                    # Generate a corrected response
                    correction_prompt = f"""Fix this response to be consistent with the scenario:

SCENARIO:
{scenario_context}

ORIGINAL RESPONSE:
{content}

ISSUES:
{', '.join(consistency_check['issues'])}

Provide a corrected response that addresses these issues while maintaining the character's voice:"""
                    
                    corrected_resp = make_api_call(
                        correction_prompt,
                        max_tokens=200,
                        temperature=0.7,
                        verbose=False
                    )
                    
                    # Clean up corrected response
                    corrected_resp = corrected_resp.strip()
                    corrected_resp = corrected_resp.replace(f"{current['name']}:", "").strip()
                    corrected_resp = corrected_resp.replace(f"[{current['name']}]", "").strip()
                    
                    # Use corrected response if it's valid
                    if len(corrected_resp.split()) >= 5:
                        content = corrected_resp
                        if verbose:
                            print("✅ Scenario inconsistency corrected")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Scenario correction failed: {e}")
        
        return content
        
    except Exception as e:
        print(f"❌ API error in generate_response_adaptive: {e}")
        return generate_emergency_response(current, other, history, repetition_data, turn)


def generate_emergency_response(current: Dict, other: Dict, history: List[Dict],
                                repetition_data: Dict, turn: int) -> str:
    """Generate emergency fallback response with maximum diversity"""
    
    emergency_templates = [
        f"*{current['name']} pauses, their expression shifting as a new thought emerges* Tell me, {other['name']}, what brought you here today?",
        f"*A memory surfaces unexpectedly* You know, {other['name']}, I once encountered something similar... *trails off thoughtfully*",
        f"*{current['name']}'s gaze sharpens with sudden intensity* Wait. There's something you're not telling me, isn't there?",
        f"*Unexpectedly changes the subject* Forgive me, but I can't help wondering - what do you truly want from this conversation?",
        f"*{current['name']} lets out a quiet laugh* This is strange. I feel like we're circling around something important without saying it directly.",
        f"*Leans forward with renewed curiosity* {other['name']}, answer me this: if you could change one thing about your current situation, what would it be?",
        f"*{current['name']}'s expression becomes thoughtful* I realize I've been talking without truly listening. What matters most to you in all of this?",
        f"*A distant sound draws their attention briefly before refocusing* Sorry, where were we? Actually, never mind that - tell me about yourself. The real you, not what you think I want to hear."
    ]
    
    # Select based on turn number for variety
    template_index = (turn * 3) % len(emergency_templates)
    response = emergency_templates[template_index]
    
    # Add environmental element occasionally
    if random.random() < 0.3:
        env_additions = [
            " *The atmosphere shifts subtly*",
            " *A chill runs through the air*",
            " *Light filters through differently now*",
            " *Time seems to slow for a moment*"
        ]
        response += random.choice(env_additions)
    
    return response
