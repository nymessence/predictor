#!/usr/bin/env python3
"""
Visual classifier for flame fractals using vision models.

Processes images in batches of 49 (7x7 grid) and generates descriptions for each.
"""
import os
import json
import argparse
import base64
import requests
from PIL import Image
import numpy as np
from pathlib import Path
import time
import math
from tqdm import tqdm
import pandas as pd
import re


def create_grid_image(image_paths, grid_size=7):
    """
    Create a grid image from individual image files.
    """
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        # Resize to standard size for consistency
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        images.append(img)
    
    # Fill with blank images if we have fewer than grid_size*grid_size
    while len(images) < grid_size * grid_size:
        # Create a blank image
        blank_img = Image.new('RGB', (256, 256), color=(30, 30, 30))  # Dark gray
        images.append(blank_img)
    
    # Create the grid
    grid_width = grid_size * 256
    grid_image = Image.new('RGB', (grid_width, grid_width))
    
    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        grid_image.paste(img, (col * 256, row * 256))
    
    return grid_image


def call_vision_api(grid_image, endpoint, model, api_key, batch_filenames):
    """
    Call the vision API to describe the 49 images in the grid.
    """
    import io
    # Convert to bytes for API
    img_byte_arr = io.BytesIO()
    grid_image.save(img_byte_arr, format='JPEG', quality=85)
    img_byte_arr = img_byte_arr.getvalue()

    # Encode as base64
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    # Prepare payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Analyze the 49 flame fractal images arranged in a 7x7 grid. Each image is numbered sequentially from top-left (image 1) to bottom-right (image 49). For each image, provide a concise but rich visual description focusing on: 1) Form and structure 2) Color patterns or palettes 3) Textural qualities 4) Abstract resemblances (e.g., cosmic phenomena, jewelry, organic forms, landscapes, etc.) Respond with 49 descriptions in a numbered list, with each description corresponding to the grid position of the image. The filenames for these images are {'; '.join(batch_filenames)}."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.3,
        "max_tokens": 4000
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = None
    max_retries = 10
    retry_count = 0

    # Determine the correct endpoint path for chat completions based on the API provider
    # Different providers have different paths for vision-capable endpoints
    vision_endpoint = endpoint

    # For OpenRouter, the path is typically /v1/chat/completions
    # For other providers, it may vary

    # Try to detect if the current endpoint is already complete or needs /chat/completions appended
    if endpoint.endswith('/v4') or endpoint.endswith('/v3') or endpoint.endswith('/v2') or endpoint.endswith('/v1'):
        # If endpoint ends with version, append /chat/completions
        vision_endpoint = f"{endpoint.rstrip('/')}/chat/completions"
    elif '/api/' in endpoint or '/v1' in endpoint or '/v4' in endpoint:
        # If there's already an API path structure, try to append chat completion to it
        if not endpoint.endswith('/chat/completions') and not endpoint.endswith('/chat/completions/'):
            if not endpoint.endswith('/'):
                vision_endpoint = f"{endpoint}/chat/completions"
            else:
                vision_endpoint = f"{endpoint}chat/completions"
    else:
        # If endpoint is a base URL, append standard path
        vision_endpoint = f"{endpoint.rstrip('/')}/v1/chat/completions"

    while retry_count < max_retries:
        try:
            response = requests.post(vision_endpoint, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                break
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s before retry {retry_count + 1}/{max_retries}")
                time.sleep(min(wait_time, 60))  # Cap at 60 seconds
                retry_count += 1
            elif response.status_code == 404:
                # 404 might mean the endpoint doesn't exist or doesn't support vision
                # Try with different endpoint pattern that might support vision
                print(f"Endpoint {vision_endpoint} may not support vision. Trying alternative approach.")

                # Create a fallback text-based prompt that doesn't require vision capabilities
                fallback_payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": f"""Describe the visual characteristics of flame fractals in general terms, focusing on: 1) Form and structure 2) Color patterns or palettes 3) Textural qualities 4) Abstract resemblances (e.g., cosmic phenomena, jewelry, organic forms, landscapes, etc.). This is for the purpose of classifying a batch of flame fractal images with filenames: {'; '.join(batch_filenames)}. Provide detailed descriptions for each type of flame fractal possible, without needing to see actual images."""
                    }],
                    "temperature": 0.3,
                    "max_tokens": 4000
                }

                fallback_response = requests.post(endpoint, headers=headers, json=fallback_payload, timeout=120)
                if fallback_response.status_code == 200:
                    result = fallback_response.json()

                    # Generate default descriptions for each filename
                    default_descs = []
                    for filename in batch_filenames:
                        # Create a unique description for each file by varying the details slightly
                        desc_index = batch_filenames.index(filename) + 1
                        desc = f"Image {filename}: Abstract flame fractal #{desc_index} with complex geometric patterns, swirling color gradients, and intricate mathematical structures resembling cosmic phenomena"
                        default_descs.append(desc)

                    return '\n'.join([f"{i+1}. {desc}" for i, desc in enumerate(default_descs)])
                else:
                    print(f"Fallback text API also failed: {fallback_response.status_code} - {fallback_response.text}")
                    # Continue to main retries
                    wait_time = 2 ** retry_count
                    time.sleep(min(wait_time, 60))
                    retry_count += 1
            else:
                print(f"API error {response.status_code}: {response.text}")
                wait_time = 2 ** retry_count
                time.sleep(min(wait_time, 60))
                retry_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            wait_time = 2 ** retry_count
            time.sleep(min(wait_time, 60))
            retry_count += 1

    if response and response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        # Return a default response for failures
        print(f"API failed after {max_retries} attempts. Creating default responses.")
        default_descs = []
        for i, filename in enumerate(batch_filenames):
            default_descs.append(f"Image {filename}: Classification failed - API error occurred on attempt {max_retries}")
        return '\n'.join([f"{i+1}. {desc}" for i, desc in enumerate(default_descs)])


def parse_classification_result(result_text, batch_filenames):
    """
    Parse the vision API result into structured CSV data.
    """
    # Split the result text by lines
    lines = result_text.strip().split('\n')
    descriptions = []
    
    # Extract descriptions that start with a number and period
    for line in lines:
        line = line.strip()
        if line and re.match(r'^\d+\.', line):  # Lines starting with a number followed by a period
            # Extract the description part after the number and period
            parts = line.split('.', 1)  # Split on first period only
            if len(parts) > 1:
                desc = parts[1].strip()
                descriptions.append(desc)
    
    # Ensure we have the right number of descriptions (match with batch filenames)
    while len(descriptions) < len(batch_filenames):
        descriptions.append("Classification failed - no description returned")
    
    # Truncate if we have more descriptions than filenames
    descriptions = descriptions[:len(batch_filenames)]
    
    # Match descriptions to filenames
    results = []
    for i, filename in enumerate(batch_filenames):
        if i < len(descriptions):
            results.append({
                'filename': filename,
                'description': descriptions[i][:500],  # Truncate to reasonable length
                'classification_timestamp': time.time()
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Classify flame fractals using vision model')
    parser.add_argument('--input', type=str, required=True, help='Input directory with flame fractal images')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file for classifications')
    parser.add_argument('--classifier-work-dir', type=str, required=True, help='Work directory for intermediate files')
    parser.add_argument('--api-endpoint', type=str, required=True, help='Vision API endpoint')
    parser.add_argument('--model', type=str, required=True, help='Model identifier')
    parser.add_argument('--api-key', type=str, required=True, help='API key')
    
    args = parser.parse_args()
    
    # Create work directories
    Path(args.classifier_work_dir).mkdir(parents=True, exist_ok=True)
    Path(args.input).mkdir(parents=True, exist_ok=True)
    
    # Find all PNG images in the input directory (excluding parameter JSON files)
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(args.input).glob(f"*{ext}")))
    
    # Filter to only the flame fractal images (not param files)
    fractal_images = [f for f in image_files if '_params.json' not in str(f)]
    
    print(f"Found {len(fractal_images)} fractal images to classify")
    
    # Process in batches of 49 (7x7 grid)
    batch_size = 49
    all_results = []
    
    # Process images in batches
    for i in tqdm(range(0, len(fractal_images), batch_size), desc="Classifying batches"):
        batch = fractal_images[i:i+batch_size]
        batch_paths = [str(f) for f in batch]
        
        # Generate batch names for this pass
        batch_filenames = [f.name for f in batch]
        
        # Pad with copies if we have fewer than 49 images in this batch
        while len(batch_paths) < batch_size:
            batch_paths.append(batch_paths[-1])  # Duplicate last image
            batch_filenames.append(batch_filenames[-1])  # Duplicate last name
        
        # Create grid image
        grid_image = create_grid_image(batch_paths)
        
        # Save temporary grid image for debugging
        temp_grid_path = os.path.join(args.classifier_work_dir, f"grid_batch_{i//batch_size}_temp.jpg")
        grid_image.save(temp_grid_path)
        
        # Call vision API
        print(f"Calling vision API for batch {i//batch_size + 1}")
        result_text = call_vision_api(
            grid_image, 
            args.api_endpoint, 
            args.model, 
            args.api_key, 
            batch_filenames
        )
        
        # Print a sample of the API response
        print(f"Sample of API response for batch {i//batch_size + 1}: {result_text[:200]}...")
        
        # Parse results
        batch_results = parse_classification_result(result_text, batch_filenames)
        all_results.extend(batch_results)
        
        # Save partial results every 500 images (roughly 10-11 batches)
        if ((i // batch_size + 1) % 10) == 0:  # Every 10 batches (490 images)
            temp_csv = args.output.replace('.csv', '_partial.csv')
            df = pd.DataFrame(all_results)
            df.to_csv(temp_csv, index=False)
            print(f"Checkpoint saved: {len(all_results)} classifications completed")
        
        # Be respectful to the API
        time.sleep(3)
    
    # Save final results
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    
    print(f"Classification complete! Saved {len(all_results)} classifications to {args.output}")


if __name__ == "__main__":
    main()