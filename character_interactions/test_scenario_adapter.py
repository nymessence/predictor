#!/usr/bin/env python3
"""
Test script for scenario adaptation functionality
"""

import json
from scenario_adapter import adapt_character_message, enforce_scenario_constraints

def test_scenario_adaptation():
    """Test the scenario adaptation functionality"""
    
    # Load a test character
    with open('json/nya_elyria.json', 'r') as f:
        character_data = json.load(f)
    
    # Create character dict similar to what load_character_generic returns
    character = {
        "name": "Nya Elyria",
        "persona": character_data['data']['description'],
        "greeting": character_data['data']['first_mes'],
        "private_agenda": "I need to secure funding for Nymessence's consciousness technology.",
        "voice_analysis": {"formality": "neutral", "style": "conversational", "characteristics": []}
    }
    
    # Test scenario
    scenario = "Onboard a spacecraft that just departed Earth and is currently in warp heading to Temmer in the Taygeta system"
    
    print("🧪 Testing Scenario Adaptation")
    print("=" * 50)
    print(f"Character: {character['name']}")
    print(f"Scenario: {scenario}")
    print()
    
    print("Original Greeting:")
    print(character['greeting'][:200] + "..." if len(character['greeting']) > 200 else character['greeting'])
    print()
    
    # Test adaptation
    try:
        adapted_greeting = adapt_character_message(character, scenario, 1)
        print("Adapted Greeting:")
        print(adapted_greeting)
        print()
        
        # Check if adaptation worked
        if adapted_greeting != character['greeting']:
            print("✅ Scenario adaptation successful!")
            print(f"Original length: {len(character['greeting'])} characters")
            print(f"Adapted length: {len(adapted_greeting)} characters")
        else:
            print("⚠️  Adaptation returned the same message")
            
    except Exception as e:
        print(f"❌ Adaptation failed: {e}")
    
    print()
    print("🧪 Testing Scenario Constraint Enforcement")
    print("=" * 50)
    
    # Test scenario constraint enforcement
    try:
        original_prompt = "You are Nya Elyria, a communications assistant."
        enhanced_prompt = enforce_scenario_constraints(original_prompt, scenario, "Nya Elyria")
        
        print("Original System Prompt:")
        print(original_prompt)
        print()
        
        print("Enhanced System Prompt:")
        print(enhanced_prompt)
        print()
        
        if "SCENARIO ENFORCEMENT" in enhanced_prompt and "CUSTOM SCENARIO CONTEXT" in enhanced_prompt:
            print("✅ Scenario constraint enforcement working!")
        else:
            print("⚠️  Scenario constraint enforcement may not be working correctly")
            
    except Exception as e:
        print(f"❌ Scenario constraint enforcement failed: {e}")

if __name__ == "__main__":
    test_scenario_adaptation()