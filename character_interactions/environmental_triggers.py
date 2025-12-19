"""
Environmental trigger generation for dynamic conversation events
"""

import random
import re
from typing import Dict, List, Optional
from api_client import make_api_call


FALLBACK_TRIGGERS = [
    "*A sudden explosion rocks the building, causing debris to rain from the ceiling*",
    "*An alarm blares loudly throughout the facility, red emergency lights flashing*",
    "*A stranger bursts through the door, bleeding and gasping for help*",
    "*The floor gives way beneath you, dropping you into a hidden chamber below*",
    "*A holographic message appears in mid-air, displaying urgent classified information*",
    "*Time seems to slow down as everything around you freezes in place*",
    "*The walls begin to shift and rearrange themselves, changing the entire layout*",
    "*A portal tears open in the air, revealing a completely different world beyond*",
    "*All technology in the room suddenly powers down, plunging you into darkness*",
    "*A mysterious figure appears behind you, their voice whispering directly in your ear*"
]


def generate_environmental_trigger(current_character: Optional[Dict] = None, 
                                  other_character: Optional[Dict] = None, 
                                  scenario_context: Optional[str] = None, 
                                  history: Optional[List[Dict]] = None) -> str:
    """Generate contextually relevant environmental triggers using AI"""
    try:
        if current_character and other_character:
            # Build comprehensive character context
            character_context = f"""
CHARACTER CONTEXT FOR ENVIRONMENTAL TRIGGERS:
Current Speaker: {current_character.get('name', 'Unknown')}
Character Description: {current_character.get('persona', '')[:500]}
Voice Analysis: {current_character.get('voice_analysis', {})}
Private Agenda: {current_character.get('private_agenda', '')}
Other Character: {other_character.get('name', 'Unknown')}
Their Description: {other_character.get('persona', '')[:300]}
"""
            
            # Generate scenario context if missing
            if not scenario_context and history:
                recent_history = " ".join([h.get('content', '')[:100] for h in history[-3:] if isinstance(h, dict)])
                scenario_context = f"Recent conversation context: {recent_history[:300]}"
            
            if scenario_context:
                character_context += f"\nCurrent Scenario Context: {scenario_context[:400]}"
            
            prompt = f"""You are a master storyteller creating environmental triggers for a dynamic character conversation. Generate 8 contextually relevant environmental triggers that:

1. Match the tone, setting, and themes of the character descriptions and current scenario
2. Are subtle but meaningful interruptions that could shift conversation dynamics
3. Are written EXACTLY in the format "*[description]*" with asterisks
4. Vary between: physical phenomena, sounds, arrivals of people/messages, atmospheric changes, and symbolic events
5. Reflect the character's world, personality, and current emotional tone
6. Are 5-20 words long and use vivid, sensory language
7. Avoid generic triggers - make them specific to this character's reality

{character_context}

Examples of excellent triggers for different contexts:
"*Ancient stone walls groan as if remembering forgotten battles*"
"*A messenger in tattered robes stumbles through the door, bleeding*"
"*The scent of burning incense suddenly intensifies, making the air thick*"
"*Distant thunder rumbles, echoing the tension in the room*"
"*A shadow detaches itself from the corner, taking a human form*"
"*The holographic displays flicker, showing corrupted Imperial data*"
"*A rare night-blooming flower suddenly opens on the windowsill*"

Generate 8 unique environmental triggers in the exact format "*description*" that fit this context:"""
            
            try:
                resp = make_api_call(
                    prompt,
                    max_tokens=350,
                    temperature=0.85,
                    stop=["\n\n", "Examples:", "Generate"],
                    verbose=False
                )
                
                content = resp.strip()
                generated_triggers = []
                
                # Extract triggers using regex pattern
                trigger_matches = re.findall(r'\*([^*]+)\*', content)
                for match in trigger_matches:
                    cleaned_trigger = f"*{match.strip()}*"
                    # Validate trigger quality
                    if (len(cleaned_trigger) > 15 and 
                        len(cleaned_trigger) < 120 and 
                        any(word in cleaned_trigger.lower() for word in [
                            'wind', 'light', 'sound', 'shadow', 'air', 'ground', 
                            'door', 'message', 'temperature', 'music', 'creature', 
                            'clock', 'wall', 'window', 'scent'
                        ])):
                        generated_triggers.append(cleaned_trigger)
                
                if generated_triggers and len(generated_triggers) >= 3:
                    return random.choice(generated_triggers)
                    
            except Exception as e:
                print(f"⚠️  AI trigger generation failed: {e}")
        
        return random.choice(FALLBACK_TRIGGERS)
        
    except Exception as e:
        print(f"⚠️  Environmental trigger generation failed: {e}")
        return random.choice(FALLBACK_TRIGGERS)
