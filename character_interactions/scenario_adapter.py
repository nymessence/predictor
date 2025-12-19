#!/usr/bin/env python3
"""
Scenario adaptation module for AI-powered message modification based on custom scenarios
"""

import re
from typing import Dict, List, Optional
from api_client import make_api_call


def adapt_character_message(character: Dict, scenario: str, turn: int = 1) -> str:
    """
    Adapt a character's message (typically greeting) based on a custom scenario using AI
    
    Args:
        character: Character dictionary with name, persona, greeting, etc.
        scenario: Custom scenario description
        turn: Current turn number (typically 1 for initial greeting)
    
    Returns:
        Adapted message that fits the scenario context
    """
    try:
        # Build context for adaptation
        character_name = character['name']
        original_greeting = character['greeting']
        persona = character['persona']
        
        # Create adaptation prompt
        adaptation_prompt = f"""Adapt the character's greeting to fit the custom scenario while maintaining their core personality.

CHARACTER:
Name: {character_name}
Persona: "{persona[:300]}"
Original Greeting: "{original_greeting}"

CUSTOM SCENARIO:
{scenario}

ADAPTATION REQUIREMENTS:
1. Keep the character's core personality and voice intact
2. Adapt the greeting to fit the scenario context meaningfully
3. Make the scenario relevant and integrated into the response
4. Maintain the same length and structure as the original
5. Add scenario-specific details that make sense for the character
6. Keep it natural and in-character

RESPONSE FORMAT:
Return ONLY the adapted greeting message. No explanations, no quotes, just the message.

Adapted greeting:"""

        # Make API call to adapt the message
        adapted_message = make_api_call(
            adaptation_prompt,
            max_tokens=200,
            temperature=0.7,
            verbose=False
        )
        
        # Clean up the response
        adapted_message = adapted_message.strip()
        
        # Remove any prefix/suffix artifacts
        adapted_message = re.sub(r'^Adapted greeting:\s*', '', adapted_message)
        adapted_message = re.sub(r'^"\s*', '', adapted_message)
        adapted_message = re.sub(r'\s*"$', '', adapted_message)
        
        # Validate the adapted message
        if len(adapted_message.strip()) < 10:
            print(f"⚠️  Adapted message too short, using original: {adapted_message}")
            return original_greeting
        
        return adapted_message
        
    except Exception as e:
        print(f"⚠️  Message adaptation failed: {e}")
        return character['greeting']


def enforce_scenario_constraints(system_prompt: str, scenario: str, character_name: str) -> str:
    """
    Add scenario enforcement constraints to the system prompt
    
    Args:
        system_prompt: Original system prompt
        scenario: Custom scenario description
        character_name: Name of the character being prompted
    
    Returns:
        Enhanced system prompt with scenario enforcement
    """
    scenario_enforcement = f"""

SCENARIO ENFORCEMENT (ABSOLUTE):
You are in the scenario: "{scenario}"
- EVERY response must be consistent with this scenario
- React to scenario elements and incorporate them into your dialogue
- Maintain scenario-appropriate knowledge and awareness
- Do not contradict or ignore scenario context
- If the scenario involves specific locations, technologies, or situations, acknowledge them
- Keep the conversation grounded in the scenario reality

CHARACTER-SPECIFIC SCENARIO ROLE:
As {character_name} in this scenario:
- Your responses should reflect your understanding of the scenario context
- Incorporate scenario-specific elements that are relevant to your character
- Maintain consistency with both your personality and the scenario
"""

    return system_prompt + scenario_enforcement


def check_scenario_consistency(response: str, scenario: str, character_name: str) -> Dict:
    """
    Check if a response is consistent with the custom scenario
    
    Args:
        response: Character's response to check
        scenario: Custom scenario description
        character_name: Name of the character who made the response
    
    Returns:
        Dictionary with consistency check results
    """
    try:
        consistency_prompt = f"""Check if this character response is consistent with the custom scenario.

SCENARIO:
{scenario}

CHARACTER RESPONSE:
{response}

ANALYSIS REQUIREMENTS:
1. Does the response acknowledge or reference elements from the scenario?
2. Is the response consistent with the scenario context?
3. Does the character's reaction make sense within the scenario?
4. Are there any contradictions with the scenario?

Return analysis in this format:
Scenario Consistency: [CONSISTENT/NEUTRAL/INCONSISTENT]
Issues: [List any issues, or "None" if consistent]
Scenario Elements Referenced: [List scenario elements mentioned, or "None"]

Analysis:"""

        analysis = make_api_call(
            consistency_prompt,
            max_tokens=150,
            temperature=0.3,
            verbose=False
        )
        
        # Parse the analysis
        result = {
            'consistent': True,
            'issues': [],
            'elements_referenced': []
        }
        
        if 'Scenario Consistency: INCONSISTENT' in analysis:
            result['consistent'] = False
        elif 'Scenario Consistency: NEUTRAL' in analysis:
            result['consistent'] = True  # Neutral is still acceptable
        
        # Extract issues
        if 'Issues:' in analysis:
            issues_part = analysis.split('Issues:')[1].split('\n')[0].strip()
            if issues_part != 'None':
                result['issues'] = [issue.strip() for issue in issues_part.split(',')]
        
        # Extract referenced elements
        if 'Scenario Elements Referenced:' in analysis:
            elements_part = analysis.split('Scenario Elements Referenced:')[1].split('\n')[0].strip()
            if elements_part != 'None':
                result['elements_referenced'] = [elem.strip() for elem in elements_part.split(',')]
        
        return result
        
    except Exception as e:
        print(f"⚠️  Scenario consistency check failed: {e}")
        return {'consistent': True, 'issues': [], 'elements_referenced': []}


def generate_scenario_guidance(scenario: str, character_name: str) -> str:
    """
    Generate scenario-specific guidance for a character
    
    Args:
        scenario: Custom scenario description
        character_name: Name of the character
    
    Returns:
        Scenario-specific guidance text
    """
    try:
        guidance_prompt = f"""Generate scenario-specific guidance for this character in the given scenario.

SCENARIO:
{scenario}

CHARACTER:
{character_name}

Generate 2-3 specific guidance points that would help this character respond appropriately in this scenario. Focus on:
1. What the character should be aware of in this scenario
2. How the character should typically react to scenario elements
3. Any scenario-specific knowledge or context they should have

Return in this format:
Scenario Guidance:
- [Guidance point 1]
- [Guidance point 2]
- [Guidance point 3]

Guidance:"""

        guidance = make_api_call(
            guidance_prompt,
            max_tokens=150,
            temperature=0.5,
            verbose=False
        )
        
        # Extract guidance points
        guidance_points = []
        if 'Scenario Guidance:' in guidance:
            guidance_part = guidance.split('Scenario Guidance:')[1]
            lines = guidance_part.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-') and len(line) > 1:
                    guidance_points.append(line[1:].strip())
        
        return "\n".join(guidance_points) if guidance_points else ""
        
    except Exception as e:
        print(f"⚠️  Scenario guidance generation failed: {e}")
        return ""