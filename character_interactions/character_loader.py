"""
Character loading and voice analysis with support for any JSON format
"""

import os
import json
import re
from typing import Dict, Any
from api_client import make_api_call


def analyze_character_voice(persona: str, character_name: str) -> Dict[str, Any]:
    """Dynamically analyze character voice from persona text"""
    try:
        prompt = f"""Analyze the voice and speaking style of {character_name} based on this persona description:
"{persona[:500]}"
Provide a concise analysis in this format:
- Formality level: [very formal/formal/neutral/casual/very casual]
- Speaking style: [direct/descriptive/poetic/humorous/philosophical/conversational]
- Key characteristics: [2-3 specific traits like "uses metaphors", "speaks in riddles", "direct and blunt", etc.]"""
        
        analysis = make_api_call(prompt, max_tokens=150, temperature=0.3, verbose=False)
        
        # Parse the analysis
        formality = "neutral"
        style = "conversational"
        characteristics = []
        
        if "Formality level:" in analysis:
            formality_line = analysis.split("Formality level:")[1].split("\n")[0].strip()
            formality = formality_line.split("/")[0].strip().lower()
        
        if "Speaking style:" in analysis:
            style_line = analysis.split("Speaking style:")[1].split("\n")[0].strip()
            style = style_line.split("/")[0].strip().lower()
        
        if "Key characteristics:" in analysis:
            chars_line = analysis.split("Key characteristics:")[1].split("\n")[0].strip()
            characteristics = [char.strip() for char in chars_line.split(",") if char.strip()]
        
        return {
            "formality": formality,
            "style": style,
            "characteristics": characteristics[:3]
        }
    except Exception as e:
        print(f"⚠️  Voice analysis warning: {e}")
        return {"formality": "neutral", "style": "conversational", "characteristics": []}


def generate_private_agenda(character_name: str, persona: str, voice_analysis: Dict) -> str:
    """Generate dynamic private agenda based on character analysis"""
    try:
        prompt = f"""Create a subtle private agenda for {character_name} based on their persona:
Persona: "{persona[:400]}"
Voice Analysis: {voice_analysis}

Requirements:
- One sentence maximum
- Must create internal tension or secret motivation
- Must be specific to this character's nature
- Should NOT be about "trusting others" or "finding common ground"
- Should reflect their unique perspective and goals

Example formats:
"If they're from the capital, I need to determine if they know about the assassination plot."
"As a wanderer who's seen too much, I'm searching for someone who understands what lies beyond the stars."
"A ruler must always test potential allies - I need to see if they break under pressure."
"""
        agenda = make_api_call(prompt, max_tokens=75, temperature=0.7, verbose=False)
        return agenda.strip('"').strip("'").strip()
        
    except Exception as e:
        print(f"⚠️  Agenda generation warning: {e}")
        if "ruler" in persona.lower() or any(x in character_name.lower() for x in ["queen", "empress", "king"]):
            return f"As a ruler, I must determine if this stranger poses any threat to my domain."
        elif any(x in persona.lower() for x in ["wanderer", "traveler", "rogue"]):
            return "I'm always on the lookout for useful information or opportunities."
        else:
            return f"{character_name} has their own hidden motivations that they keep carefully guarded."


def generate_missing_scenario(character_data: Dict, character_name: str) -> str:
    """Generate missing scenario context from character description"""
    try:
        persona = character_data.get('persona', '')
        if not persona or len(persona) < 50:
            return f"{character_name} exists in a dynamic world where their actions shape the narrative."
        
        prompt = f"""Based on this character, generate a brief 2-sentence scenario context:

Character: {character_name}
Description: "{persona[:600]}"

Requirements:
- 2 sentences maximum
- Describe their current location and situation
- Include 1-2 sensory details
- Set up potential for interaction
- Match the tone and themes of their character
"""
        scenario = make_api_call(prompt, max_tokens=75, temperature=0.7, verbose=False)
        return scenario.strip()
        
    except Exception as e:
        print(f"⚠️  Scenario generation warning: {e}")
        return f"{character_name} exists in their own world, shaped by their experiences and goals."


def extract_system_prompt(character_data: Dict) -> str:
    """Extract system prompt from character data, trying multiple paths"""
    paths_to_check = [
        ('system_prompt',),
        ('data', 'system_prompt'),
        ('character', 'system_prompt'),
        ('prompt',),
        ('data', 'prompt'),
        ('system', 'prompt'),
        ('system', 'system_prompt')
    ]
    
    for path in paths_to_check:
        current = character_data
        try:
            for key in path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    raise KeyError
            if isinstance(current, str) and current.strip():
                return current.strip()
        except (KeyError, TypeError, AttributeError):
            continue
    
    return ""


def load_character_generic(filepath: str) -> Dict[str, Any]:
    """Load ANY character file format with robust features"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                character_data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                content = f.read()
                character_data = {'raw_content': content}
        
        print(f"🔍 Analyzing character file: {filepath}")
        
        # Extract character name
        character_name = "Unknown Character"
        name_paths = [
            ('name',), ('data', 'name'), ('character', 'name'),
            ('personality', 'name'), ('system_prompt',), ('description',)
        ]
        
        for path in name_paths:
            current = character_data
            try:
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        raise KeyError
                if isinstance(current, str) and current.strip():
                    character_name = current.strip()
                    break
            except (KeyError, TypeError, AttributeError):
                continue
        
        if character_name == "Unknown Character":
            character_name = os.path.splitext(os.path.basename(filepath))[0].replace('_', ' ').title()
            print(f"   ⚠️  Using filename as character name: {character_name}")
        
        print(f"   ✓ Character name: {character_name}")
        
        # Extract persona
        persona = ""
        persona_sources = [
            ('description',), ('personality',), ('system_prompt',),
            ('data', 'description'), ('data', 'personality'),
            ('character', 'description'), ('scenario',), ('raw_content',)
        ]
        
        for path in persona_sources:
            current = character_data
            try:
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        raise KeyError
                if isinstance(current, str) and len(current.strip()) > 20:
                    persona = current.strip()
                    break
            except (KeyError, TypeError, AttributeError):
                continue
        
        if not persona:
            persona = f"{character_name} is a character with their own motivations and personality."
            print(f"   ⚠️  Creating minimal persona")
        
        print(f"   ✓ Persona length: {len(persona)} characters")
        
        # Generate scenario if missing
        scenario_context = None
        if not any(key in character_data for key in ['scenario', 'context', 'setting', 'world']):
            print(f"   🌍 Generating missing scenario context...")
            scenario_context = generate_missing_scenario(character_data, character_name)
            character_data['generated_scenario'] = scenario_context
            print(f"   ✓ Generated scenario")
        
        # Extract greeting
        greeting = ""
        greeting_sources = [
            ('first_mes',), ('greeting',), ('data', 'first_mes'),
            ('data', 'greeting'), ('messages', 0, 'content'), ('conversation_start',)
        ]
        
        for path in greeting_sources:
            current = character_data
            try:
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    elif isinstance(current, list) and isinstance(key, int) and key < len(current):
                        current = current[key]
                    else:
                        raise KeyError
                if isinstance(current, str) and len(current.strip()) > 5:
                    greeting = current.strip()
                    greeting = re.sub(r'\{\{user\}\}|\{\{char\}\}|\{\{User\}\}|\{\{Char\}\}|\<USER\>|\<BOT\>', 
                                    'you', greeting)
                    break
            except (KeyError, TypeError, AttributeError, IndexError):
                continue
        
        if not greeting:
            greeting = f"*{character_name} looks up as you approach* Hello. I am {character_name}."
            print(f"   ⚠️  Creating default greeting")
        
        print(f"   ✓ Greeting extracted")
        
        # Analyze character voice
        print(f"   🎭 Analyzing character voice...")
        voice_analysis = analyze_character_voice(persona, character_name)
        print(f"   ✓ Voice analysis: {voice_analysis}")
        
        # Generate private agenda
        print(f"   🤖 Generating private agenda...")
        private_agenda = generate_private_agenda(character_name, persona, voice_analysis)
        print(f"   ✓ Private agenda generated")
        
        # System prompt (will be built in context_builder)
        system_prompt = extract_system_prompt(character_data)
        
        return {
            "name": character_name,
            "persona": persona,
            "greeting": greeting,
            "private_agenda": private_agenda,
            "voice_analysis": voice_analysis,
            "system_prompt": system_prompt,
            "scenario_context": scenario_context,
            "raw_data": character_data
        }
        
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        
        character_name = os.path.splitext(os.path.basename(filepath))[0].replace('_', ' ').title()
        print(f"   ⚠️  Creating fallback character: {character_name}")
        
        return {
            "name": character_name,
            "persona": f"{character_name} is a character with their own unique personality.",
            "greeting": f"*{character_name} regards you with interest* Hello. I am {character_name}.",
            "private_agenda": "I want to understand who you are and what you want.",
            "voice_analysis": {"formality": "neutral", "style": "conversational", "characteristics": []},
            "system_prompt": "",
            "scenario_context": None,
            "raw_data": {}
        }


def extract_lorebook_entries(character_data: Dict, recent_messages: list, max_entries: int = 3) -> list:
    """Safely extract lorebook entries from ANY character format"""
    try:
        lorebook_entries = []
        paths_to_check = [
            ('character_book', 'entries'), ('data', 'character_book', 'entries'),
            ('lore', 'entries'), ('book', 'entries'), ('entries',),
            ('data', 'entries'), ('personality',), ('scenario',)
        ]
        
        for path in paths_to_check:
            current = character_data
            try:
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        raise KeyError
                
                if isinstance(current, list):
                    lorebook_entries.extend([str(entry) for entry in current if entry])
                elif isinstance(current, dict) and 'entries' in current:
                    entries = current['entries']
                    if isinstance(entries, list):
                        lorebook_entries.extend([str(entry) for entry in entries if entry])
                elif isinstance(current, str) and current.strip():
                    lorebook_entries.append(current)
                elif isinstance(current, dict):
                    for key, value in current.items():
                        if isinstance(value, str) and len(value) > 20:
                            lorebook_entries.append(f"{key}: {value}")
            except (KeyError, TypeError, AttributeError):
                continue
        
        # Filter by relevance if we have recent messages
        if lorebook_entries and recent_messages:
            recent_text = " ".join([str(m.get('content', '')) for m in recent_messages[-3:] 
                                  if isinstance(m, dict)]).lower()
            scored_entries = []
            
            for entry in lorebook_entries:
                if not isinstance(entry, str) or len(entry) < 10:
                    continue
                relevance = sum(1 for word in recent_text.split() 
                              if len(word) > 3 and word in entry.lower())
                if relevance > 0:
                    scored_entries.append((relevance, entry))
            
            scored_entries.sort(reverse=True)
            relevant_entries = [entry for _, entry in scored_entries[:max_entries]]
            if relevant_entries:
                return relevant_entries
        
        return lorebook_entries[:max_entries] if lorebook_entries else []
        
    except Exception as e:
        print(f"⚠️  Lorebook extraction warning: {e}")
        return []
