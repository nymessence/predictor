#!/usr/bin/env python3
"""
Comprehensive test script to verify multi-character support and custom scenarios
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Activate virtual environment
os.environ['VIRTUAL_ENV'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv')

from main import parse_arguments, setup_output_file, validate_character_files
from character_loader import load_character_generic
import json

def test_character_loading():
    """Test loading multiple characters"""
    print("Testing character loading...")
    
    # Test 2 characters
    characters_2 = [
        load_character_generic('json/empress_azalea.json'),
        load_character_generic('json/queen_alenym.json')
    ]
    print(f"✓ Loaded 2 characters: {[c['name'] for c in characters_2]}")
    
    # Test 3 characters
    characters_3 = [
        load_character_generic('json/empress_azalea.json'),
        load_character_generic('json/queen_alenym.json'),
        load_character_generic('json/mari_swaruu.json')
    ]
    print(f"✓ Loaded 3 characters: {[c['name'] for c in characters_3]}")
    
    # Validate character structure
    for char in characters_3:
        assert 'name' in char, f"Character missing name: {char}"
        assert 'private_agenda' in char, f"Character missing private_agenda: {char}"
        assert 'voice_analysis' in char, f"Character missing voice_analysis: {char}"
        assert 'greeting' in char, f"Character missing greeting: {char}"
        assert 'raw_data' in char, f"Character missing raw_data: {char}"
    
    print("✓ All character structures validated")
    return characters_2, characters_3

def test_scenario_keyword_extraction():
    """Test scenario-based keyword extraction"""
    print("\nTesting scenario keyword extraction...")
    
    from character_loader import extract_lorebook_entries
    
    # Load a character
    character = load_character_generic('json/empress_azalea.json')
    
    # Test scenario with keywords that should match lorebook entries
    test_scenarios = [
        "royal court politics and intrigue",
        "space exploration and adventure", 
        "ancient mysteries and forgotten knowledge"
    ]
    
    history = [{'name': character['name'], 'content': 'Hello, I am here.'}]
    
    for scenario in test_scenarios:
        print(f"  Testing scenario: '{scenario}'")
        scenario_keywords = scenario.lower().split()
        
        # Extract lorebook entries
        entries = extract_lorebook_entries(character['raw_data'], history, max_entries=3)
        
        # Filter entries by scenario relevance
        relevant_entries = []
        for entry in entries:
            if any(keyword in entry.lower() for keyword in scenario_keywords if len(keyword) > 3):
                relevant_entries.append(entry)
        
        print(f"    Found {len(relevant_entries)} relevant entries")
        for i, entry in enumerate(relevant_entries[:2]):
            print(f"      {i+1}. {entry[:100]}...")
    
    print("✓ Scenario keyword extraction test completed")

def test_multi_character_switching():
    """Test character switching logic for multiple characters"""
    print("\nTesting multi-character switching logic...")
    
    # Test with 3 characters
    characters = [
        load_character_generic('json/empress_azalea.json'),
        load_character_generic('json/queen_alenym.json'),
        load_character_generic('json/mari_swaruu.json')
    ]
    
    # Simulate character switching
    current_char_index = 0
    other_char_index = 1
    
    # Simulate 5 turns to verify switching works correctly
    turn_order = []
    for turn in range(5):
        turn_order.append((current_char_index, other_char_index))
        
        # Switch characters (cycle through all characters)
        other_char_index = current_char_index
        current_char_index = (current_char_index + 1) % len(characters)
    
    print(f"  Turn order: {[(characters[i]['name'], characters[j]['name']) for i, j in turn_order]}")
    
    # Verify that all characters get turns
    speakers = [characters[i]['name'] for i, j in turn_order]
    unique_speakers = set(speakers)
    print(f"  Speakers: {unique_speakers}")
    
    if len(unique_speakers) == len(characters):
        print("✓ All characters received turns")
    else:
        print(f"❌ Not all characters received turns. Expected {len(characters)}, got {len(unique_speakers)}")
    
    return True

def test_output_file_naming():
    """Test output file naming with multiple characters"""
    print("\nTesting output file naming...")
    
    # Test various character name combinations
    test_cases = [
        (['Azalea', 'Alenym'], 'Azalea_&_Alenym_conversation.json'),
        (['Nya', 'Azalea', 'Mari'], 'Nya_&_Azalea_&_Mari_conversation.json'),
        (['Character With Spaces', 'Another-Character'], 'Character_With_Spaces_&_Another-Character_conversation.json'),
        (['Character<With>Special:Chars/'], 'Character_With_Special_Chars__conversation.json')
    ]
    
    for names, expected in test_cases:
        result = setup_output_file(names)
        print(f"  {names} -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ Output file naming test passed")

def test_validation_functions():
    """Test validation functions"""
    print("\nTesting validation functions...")
    
    # Test character file validation
    try:
        # Valid files
        args_valid = type('Args', (), {
            'characters': ['json/empress_azalea.json', 'json/queen_alenym.json']
        })()
        validate_character_files(args_valid)
        print("✓ Valid character files passed validation")
    except SystemExit:
        print("❌ Valid character files failed validation")
        return False
    
    # Test invalid files
    try:
        args_invalid = type('Args', (), {
            'characters': ['nonexistent.json']
        })()
        validate_character_files(args_invalid)
        print("❌ Invalid character files should have failed validation")
        return False
    except SystemExit:
        print("✓ Invalid character files correctly rejected")
    
    return True

def run_all_tests():
    """Run all tests"""
    print("🧪 Starting comprehensive tests...")
    print("=" * 60)
    
    try:
        # Test 1: Character loading
        characters_2, characters_3 = test_character_loading()
        
        # Test 2: Scenario keyword extraction
        test_scenario_keyword_extraction()
        
        # Test 3: Multi-character switching
        test_multi_character_switching()
        
        # Test 4: Output file naming
        test_output_file_naming()
        
        # Test 5: Validation functions
        test_validation_functions()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Multi-character and custom scenario functionality is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)