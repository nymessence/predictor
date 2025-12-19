#!/usr/bin/env python3
"""
Convert AI predictions JSON file to Markdown format
"""

import argparse
import json
import os


def json_to_markdown(json_file_path, markdown_file_path):
    """
    Convert AI predictions JSON file to Markdown format
    
    Args:
        json_file_path (str): Path to the input JSON file
        markdown_file_path (str): Path to the output Markdown file
    """
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Extract information from the JSON
    topic = data.get('topic', 'AI Predictions')
    start_year = data.get('start_year', '')
    end_year = data.get('end_year', '')
    
    # Start building the markdown content
    markdown_content = f"# {topic} Predictions\n\n"
    markdown_content += f"**Time Period:** {start_year} - {end_year}\n\n"
    
    # Add the predictions for each year
    predictions = data.get('predictions', {})
    
    # Sort years numerically to ensure chronological order
    sorted_years = sorted(predictions.keys(), key=int)
    
    for year in sorted_years:
        prediction_text = predictions[year]
        
        # Add the year as a header
        markdown_content += f"## {year}\n\n"
        
        # Add the prediction text
        markdown_content += f"{prediction_text}\n\n"
        
        # Add a horizontal rule between years for better readability
        markdown_content += "---\n\n"

    # Write the markdown content to the output file
    with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_content)

    print(f"Successfully converted '{json_file_path}' to '{markdown_file_path}'")


def main():
    parser = argparse.ArgumentParser(description="Convert AI predictions JSON file to Markdown format")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", required=True, help="Output Markdown file path")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    # Convert the file
    json_to_markdown(args.input, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())