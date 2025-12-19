#!/usr/bin/env python3
"""
Functional test script to verify core functionality without API calls
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Activate virtual environment
os.environ['VIRTUAL_ENV'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv')

from main import parse_arguments, setup_output_file, validate_character_files
import json

def test_argument_parsing():
    """Test argument parsing for multi-character and scenarios"""
    print("Testing argument parsing...")
    
    # Test 2 characters with scenario
    sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json', '--scenario', 'royal court politics']
    args = parse_arguments()
    assert len(args.characters) == 2, f"Expected 2 characters, got {len(args.characters)}"
    assert args.scenario == 'royal court politics', f"Expected scenario 'royal court politics', got {args.scenario}"
    print(f"✓ 2 characters with scenario: {args.characters}, scenario: {args.scenario}")
    
    # Test 3 characters with scenario
    sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json', 'json/mari_swaruu.json', '--scenario', 'space adventure']
    args = parse_arguments()
    assert len(args.characters) == 3, f"Expected 3 characters, got {len(args.characters)}"
    assert args.scenario == 'space adventure', f"Expected scenario 'space adventure', got {args.scenario}"
    print(f"✓ 3 characters with scenario: {args.characters}, scenario: {args.scenario}")
    
    # Test 4 characters (edge case)
    sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json', 'json/mari_swaruu.json', 'json/nya_elyria.json']
    args = parse_arguments()
    assert len(args.characters) == 4, f"Expected 4 characters, got {len(args.characters)}"
    print(f"✓ 4 characters: {len(args.characters)} characters loaded")
    
    print("✓ Argument parsing test passed")

def test_output_file_naming():
    """Test output file naming with various character combinations"""
    print("\nTesting output file naming...")
    
    test_cases = [
        (['Empress Azalea', 'Queen Alenym'], 'Empress_Azalea_&_Queen_Alenym_conversation.json'),
        (['Mari Swaruu', 'Nya Elyria'], 'Mari_Swaruu_&_Nya_Elyria_conversation.json'),
        (['Azalea', 'Alenym', 'Mari'], 'Azalea_&_Alenym_&_Mari_conversation.json'),
        (['Character With Spaces', 'Another-Character'], 'Character_With_Spaces_&_Another-Character_conversation.json'),
        (['Character<With>Special:Chars/'], 'Character_With_Special_Chars__conversation.json')
    ]
    
    for names, expected in test_cases:
        result = setup_output_file(names)
        print(f"  Input: {names}")
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")
        assert result == expected, f"Expected {expected}, got {result}"
        print("  ✓ Match")
    
    print("✓ Output file naming test passed")

def test_character_file_validation():
    """Test character file validation"""
    print("\nTesting character file validation...")
    
    # Test valid files
    try:
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
    
    # Test mixed valid/invalid files
    try:
        args_mixed = type('Args', (), {
            'characters': ['json/empress_azalea.json', 'nonexistent.json']
        })()
        validate_character_files(args_mixed)
        print("❌ Mixed valid/invalid files should have failed validation")
        return False
    except SystemExit:
        print("✓ Mixed valid/invalid files correctly rejected")
    
    return True

def test_json_file_structure():
    """Test that character JSON files have the expected structure"""
    print("\nTesting JSON file structure...")
    
    character_files = [
        'json/empress_azalea.json',
        'json/queen_alenym.json', 
        'json/mari_swaruu.json'
    ]
    
    for char_file in character_files:
        if not os.path.exists(char_file):
            print(f"❌ Character file not found: {char_file}")
            return False
        
        try:
            with open(char_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required fields - account for nested structure
            def get_nested_value(obj, paths):
                for path in paths:
                    current = obj
                    try:
                        for key in path:
                            if isinstance(current, dict) and key in current:
                                current = current[key]
                            else:
                                raise KeyError
                        return current
                    except (KeyError, TypeError, AttributeError):
                        continue
                return None
            
            # Check name field
            name_paths = [('name',), ('data', 'name'), ('character', 'name')]
            name = get_nested_value(data, name_paths)
            if not name:
                print(f"❌ Missing required field 'name' in {char_file}")
                return False
            
            # Check description/persona field
            desc_paths = [('description',), ('data', 'description'), ('personality',)]
            description = get_nested_value(data, desc_paths)
            if not description or len(description.strip()) < 20:
                print(f"❌ Missing or invalid required field 'description' in {char_file}")
                return False
            
            # Check greeting field
            greeting_paths = [('first_mes',), ('greeting',), ('data', 'first_mes')]
            greeting = get_nested_value(data, greeting_paths)
            if not greeting or len(greeting.strip()) < 5:
                print(f"❌ Missing or invalid required field 'greeting' in {char_file}")
                return False
            
            print(f"✓ {char_file}: {name} - valid structure")
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {char_file}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error reading {char_file}: {e}")
            return False
    
    return True

def test_scenario_parameter_handling():
    """Test scenario parameter handling"""
    print("\nTesting scenario parameter handling...")
    
    # Test with scenario
    sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json', '--scenario', 'test scenario']
    args = parse_arguments()
    assert args.scenario == 'test scenario', f"Expected 'test scenario', got {args.scenario}"
    print(f"✓ Scenario parameter: '{args.scenario}'")
    
    # Test without scenario
    sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json']
    args = parse_arguments()
    assert args.scenario is None, f"Expected None, got {args.scenario}"
    print(f"✓ No scenario parameter: {args.scenario}")
    
    # Test with empty scenario
    sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json', '--scenario', '']
    args = parse_arguments()
    assert args.scenario == '', f"Expected empty string, got {args.scenario}"
    print(f"✓ Empty scenario parameter: '{args.scenario}'")
    
    return True

def run_all_tests():
    """Run all functional tests"""
    print("🧪 Starting functional tests...")
    print("=" * 50)
    
    try:
        # Test 1: Argument parsing
        test_argument_parsing()
        
        # Test 2: Output file naming
        test_output_file_naming()
        
        # Test 3: Character file validation
        if not test_character_file_validation():
            return False
        
        # Test 4: JSON file structure
        if not test_json_file_structure():
            return False
        
        # Test 5: Scenario parameter handling
        if not test_scenario_parameter_handling():
            return False
        
        print("\n" + "=" * 50)
        print("🎉 All functional tests passed!")
        print("✓ Multi-character support working")
        print("✓ Custom scenario parameter working")
        print("✓ File validation working")
        print("✓ Output file naming working")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)