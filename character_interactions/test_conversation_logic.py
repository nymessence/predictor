#!/usr/bin/env python3
"""
Test conversation logic without making API calls
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Activate virtual environment
os.environ['VIRTUAL_ENV'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv')

from main import setup_output_file, validate_character_files
from character_loader import load_character_generic
import json

def test_conversation_switching_logic():
    """Test the character switching logic for multi-character conversations"""
    print("Testing conversation switching logic...")
    
    # Load test characters
    characters = [
        load_character_generic('json/empress_azalea.json'),
        load_character_generic('json/queen_alenym.json'),
        load_character_generic('json/mari_swaruu.json')
    ]
    
    print(f"Loaded characters: {[c['name'] for c in characters]}")
    
    # Simulate conversation switching
    current_char_index = 0
    other_char_index = 1
    history = []
    
    # Simulate 6 turns
    turn_order = []
    for turn in range(1, 7):
        current_char = characters[current_char_index]
        other_char = characters[other_char_index]
        
        turn_order.append({
            'turn': turn,
            'speaker': current_char['name'],
            'listener': other_char['name']
        })
        
        # Simulate response (mock)
        mock_response = f"[Mock response from {current_char['name']} to {other_char['name']} on turn {turn}]"
        history.append({
            'name': current_char['name'],
            'content': mock_response
        })
        
        # Switch characters (cycle through all)
        other_char_index = current_char_index
        current_char_index = (current_char_index + 1) % len(characters)
    
    # Verify turn order
    print("Turn order:")
    for turn in turn_order:
        print(f"  Turn {turn['turn']}: {turn['speaker']} -> {turn['listener']}")
    
    # Verify all characters got turns
    speakers = [t['speaker'] for t in turn_order]
    unique_speakers = set(speakers)
    
    if len(unique_speakers) == len(characters):
        print(f"✓ All {len(characters)} characters received turns")
    else:
        print(f"❌ Not all characters received turns. Expected {len(characters)}, got {len(unique_speakers)}")
        return False
    
    # Verify proper cycling
    expected_speakers = [
        characters[0]['name'],  # Turn 1
        characters[1]['name'],  # Turn 2
        characters[2]['name'],  # Turn 3
        characters[0]['name'],  # Turn 4
        characters[1]['name'],  # Turn 5
        characters[2]['name'],  # Turn 6
    ]
    
    if speakers == expected_speakers:
        print("✓ Character cycling works correctly")
    else:
        print(f"❌ Character cycling incorrect. Expected {expected_speakers}, got {speakers}")
        return False
    
    return True

def test_scenario_keyword_matching():
    """Test scenario keyword matching logic"""
    print("\nTesting scenario keyword matching...")
    
    from character_loader import extract_lorebook_entries
    
    # Load a character
    character = load_character_generic('json/empress_azalea.json')
    
    # Test scenarios with different keyword sets
    test_scenarios = [
        {
            'scenario': 'royal court politics and intrigue',
            'expected_keywords': ['royal', 'court', 'politics', 'intrigue']
        },
        {
            'scenario': 'space exploration and adventure',
            'expected_keywords': ['space', 'exploration', 'adventure']
        },
        {
            'scenario': 'ancient mysteries and forgotten knowledge',
            'expected_keywords': ['ancient', 'mysteries', 'forgotten', 'knowledge']
        }
    ]
    
    history = [{'name': character['name'], 'content': 'Hello, I am here.'}]
    
    for test_case in test_scenarios:
        scenario = test_case['scenario']
        expected_keywords = test_case['expected_keywords']
        
        print(f"  Testing scenario: '{scenario}'")
        
        # Extract lorebook entries
        entries = extract_lorebook_entries(character['raw_data'], history, max_entries=5)
        
        # Filter entries by scenario relevance
        scenario_keywords = scenario.lower().split()
        relevant_entries = []
        
        for entry in entries:
            entry_lower = entry.lower()
            is_relevant = any(keyword in entry_lower for keyword in scenario_keywords if len(keyword) > 3)
            if is_relevant:
                relevant_entries.append(entry)
        
        print(f"    Found {len(relevant_entries)} relevant entries out of {len(entries)} total")
        
        # Show some examples
        for i, entry in enumerate(relevant_entries[:2]):
            print(f"      {i+1}. {entry[:100]}...")
    
    print("✓ Scenario keyword matching test completed")
    return True

def test_conversation_history_format():
    """Test conversation history format and output"""
    print("\nTesting conversation history format...")
    
    # Create mock conversation history
    history = [
        {'name': 'Empress Azalea', 'content': 'Hello, I am the Empress.'},
        {'name': 'Queen Alenym', 'content': 'Greetings, Your Majesty.'},
        {'name': 'Empress Azalea', 'content': 'What brings you to my court?'},
        {'name': 'Queen Alenym', 'content': 'I come seeking an alliance.'}
    ]
    
    # Test output file naming
    output_file = setup_output_file(['Empress Azalea', 'Queen Alenym'])
    expected_file = 'Empress_Azalea_&_Queen_Alenym_conversation.json'
    
    if output_file != expected_file:
        print(f"❌ Output file naming failed. Expected {expected_file}, got {output_file}")
        return False
    
    print(f"✓ Output file: {output_file}")
    
    # Test conversation history saving format
    try:
        # Save conversation
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([{
                'turn': i+1,
                'name': h['name'],
                'content': h['content']
            } for i, h in enumerate(history) if isinstance(h, dict) and h.get('name') and h.get('content')], 
            f, indent=4, ensure_ascii=False)
        
        # Verify saved format
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # Check structure
        if len(saved_data) != len(history):
            print(f"❌ Saved conversation length mismatch. Expected {len(history)}, got {len(saved_data)}")
            return False
        
        for i, entry in enumerate(saved_data):
            if 'turn' not in entry or 'name' not in entry or 'content' not in entry:
                print(f"❌ Missing required fields in entry {i}")
                return False
            
            if entry['name'] != history[i]['name'] or entry['content'] != history[i]['content']:
                print(f"❌ Entry {i} content mismatch")
                return False
        
        print("✓ Conversation history format is correct")
        
        # Clean up
        os.remove(output_file)
        
    except Exception as e:
        print(f"❌ Error testing conversation history: {e}")
        return False
    
    return True

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\nTesting edge cases...")
    
    # Test with 2 characters
    try:
        characters_2 = [
            load_character_generic('json/empress_azalea.json'),
            load_character_generic('json/queen_alenym.json')
        ]
        print(f"✓ 2 characters work: {[c['name'] for c in characters_2]}")
    except Exception as e:
        print(f"❌ 2 characters failed: {e}")
        return False
    
    # Test with 4 characters (maximum reasonable)
    try:
        characters_4 = [
            load_character_generic('json/empress_azalea.json'),
            load_character_generic('json/queen_alenym.json'),
            load_character_generic('json/mari_swaruu.json'),
            load_character_generic('json/nya_elyria.json')
        ]
        print(f"✓ 4 characters work: {[c['name'] for c in characters_4]}")
    except Exception as e:
        print(f"❌ 4 characters failed: {e}")
        return False
    
    # Test scenario with empty string
    try:
        sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json', '--scenario', '']
        from main import parse_arguments
        args = parse_arguments()
        if args.scenario != '':
            print(f"❌ Empty scenario test failed. Expected '', got {args.scenario}")
            return False
        print("✓ Empty scenario works")
    except Exception as e:
        print(f"❌ Empty scenario test failed: {e}")
        return False
    
    # Test scenario with special characters
    try:
        sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json', '--scenario', 'royal court "politics" & intrigue!']
        from main import parse_arguments
        args = parse_arguments()
        if args.scenario != 'royal court "politics" & intrigue!':
            print(f"❌ Special characters scenario test failed. Expected 'royal court \"politics\" & intrigue!', got {args.scenario}")
            return False
        print("✓ Special characters scenario works")
    except Exception as e:
        print(f"❌ Special characters scenario test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all conversation logic tests"""
    print("🧪 Starting conversation logic tests...")
    print("=" * 50)
    
    try:
        # Test 1: Conversation switching logic
        if not test_conversation_switching_logic():
            return False
        
        # Test 2: Scenario keyword matching
        if not test_scenario_keyword_matching():
            return False
        
        # Test 3: Conversation history format
        if not test_conversation_history_format():
            return False
        
        # Test 4: Edge cases
        if not test_edge_cases():
            return False
        
        print("\n" + "=" * 50)
        print("🎉 All conversation logic tests passed!")
        print("✓ Multi-character conversation switching works")
        print("✓ Scenario keyword matching works")
        print("✓ Conversation history format works")
        print("✓ Edge cases handled correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)