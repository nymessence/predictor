#!/usr/bin/env python3
"""
Basic test script to verify the new argument parsing without making API calls
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import parse_arguments, setup_output_file, validate_character_files

def test_argument_parsing():
    """Test the new argument parsing"""
    print("Testing argument parsing...")
    
    # Test 2 characters
    sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json', '--scenario', 'test scenario']
    args = parse_arguments()
    print(f"Characters: {args.characters}")
    print(f"Scenario: {args.scenario}")
    
    # Test 3 characters
    sys.argv = ['main.py', 'json/empress_azalea.json', 'json/queen_alenym.json', 'json/mari_swaruu.json', '--scenario', 'multi-character test']
    args = parse_arguments()
    print(f"Characters: {args.characters}")
    print(f"Scenario: {args.scenario}")
    
    # Test output file naming
    output = setup_output_file(['Character A', 'Character B', 'Character C'])
    print(f"Output file: {output}")
    
    print("✓ Argument parsing test passed!")

if __name__ == "__main__":
    test_argument_parsing()