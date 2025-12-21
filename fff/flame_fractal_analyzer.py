#!/usr/bin/env python3
"""
Flame fractal analyzer and parameter refinement tool.

Analyzes classified images to identify visually compelling motifs and refines parameters.
"""
import os
import json
import pandas as pd
import argparse
from pathlib import Path
import csv


def analyze_classifications(csv_path):
    """
    Analyze the classification CSV to identify visually compelling or recurring motifs.
    """
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} classifications for analysis")
    
    # Identify interesting image categories based on description keywords
    interesting_images = []
    
    for idx, row in df.iterrows():
        desc = row['description'].lower() if pd.notna(row['description']) else ''
        
        # Keywords for identifying visually compelling motifs
        jewelry_keywords = ['jewel', 'pendant', 'ornament', 'gem', 'crystal', 'diamond', 'necklace', 'bracelet', 'ring', 'brooch']
        nature_keywords = ['landscape', 'organic', 'flower', 'leaf', 'tree', 'natural', 'growth', 'cellular', 'tree-like', 'leaf-like']
        cosmic_keywords = ['nebula', 'galaxy', 'star', 'cosmic', 'space', 'universe', 'stellar', 'celestial', 'aurora', 'cosmos']
        symmetry_keywords = ['symmetrical', 'symmetric', 'balanced', 'radial', 'circular', 'geometric', 'pattern', 'repeating']
        chaotic_keywords = ['chaotic', 'complex', 'intricate', 'detailed', 'dynamic', 'flowing', 'swirling', 'spiral']
        
        # Determine if image is interesting based on matching keywords
        is_interesting = (
            any(kw in desc for kw in jewelry_keywords) or
            any(kw in desc for kw in nature_keywords) or
            any(kw in desc for kw in cosmic_keywords) or
            any(kw in desc for kw in symmetry_keywords) or
            any(kw in desc for kw in chaotic_keywords)
        )
        
        if is_interesting:
            interesting_images.append({
                'filename': row['filename'],
                'description': row['description'],
                'is_jewelry_like': any(kw in desc for kw in jewelry_keywords),
                'is_nature_like': any(kw in desc for kw in nature_keywords),
                'is_cosmic_like': any(kw in desc for kw in cosmic_keywords),
                'is_symmetric': any(kw in desc for kw in symmetry_keywords),
                'is_complex': any(kw in desc for kw in chaotic_keywords)
            })
    
    print(f"Identified {len(interesting_images)} potentially interesting images")
    return interesting_images, df


def extract_parameters_for_images(interesting_images, base_dir):
    """
    Extract parameters for interesting images.
    """
    interesting_with_params = []
    
    for img_info in interesting_images:
        filename = img_info['filename']
        # Find corresponding parameter file
        param_filename = filename.replace('.png', '_params.json')
        param_path = Path(base_dir) / param_filename
        
        if param_path.exists():
            try:
                with open(param_path, 'r') as f:
                    params = json.load(f)
                
                # Add parameters to image info
                img_info['parameters'] = params
                img_info['seed'] = params.get('seed', 'unknown')
                interesting_with_params.append(img_info)
            except Exception as e:
                print(f"Error loading parameters for {param_filename}: {e}")
        else:
            print(f"Parameter file not found for {filename}")
    
    return interesting_with_params


def perturb_parameters(params, perturbation_factor=0.1):
    """
    Create slightly perturbed parameters from existing ones.
    """
    import random
    
    new_params = {'transforms': [], 'seed': f"perturbed_{params['seed']}"}
    
    # Copy over basic parameters
    for key in ['num_points', 'gamma', 'brightness', 'contrast', 'hue_rotation']:
        if key in params:
            val = params[key]
            if isinstance(val, (int, float)):
                # Add small random perturbation
                if isinstance(val, int):
                    perturbation = int(random.uniform(-abs(val)*perturbation_factor, abs(val)*perturbation_factor))
                    new_params[key] = val + perturbation
                else:  # float
                    perturbation = random.uniform(-abs(val)*perturbation_factor, abs(val)*perturbation_factor)
                    new_params[key] = val + perturbation
            else:
                new_params[key] = val
        else:
            # Use defaults
            if key == 'num_points':
                new_params[key] = 2000000
            elif key == 'gamma':
                new_params[key] = 2.0
            elif key == 'brightness':
                new_params[key] = 1.0
            elif key == 'contrast':
                new_params[key] = 1.0
            elif key == 'hue_rotation':
                new_params[key] = random.uniform(0.0, 360.0)
    
    # Perturb transforms
    for transform in params['transforms']:
        new_transform = {}
        for key in ['a', 'b', 'c', 'd', 'e', 'f', 'color', 'weight']:
            if key in transform:
                val = transform[key]
                if isinstance(val, (int, float)):
                    if isinstance(val, int):
                        perturbation = int(random.uniform(-abs(val)*perturbation_factor, abs(val)*perturbation_factor))
                        new_transform[key] = val + perturbation
                    else:  # float
                        perturbation = random.uniform(-abs(val)*perturbation_factor, abs(val)*perturbation_factor)
                        new_transform[key] = val + perturbation
                else:
                    new_transform[key] = val
            else:
                # Set default values for missing keys
                if key in ['a', 'b', 'c', 'd', 'e', 'f']:
                    new_transform[key] = random.uniform(-1.0, 1.0)
                elif key == 'color':
                    new_transform[key] = random.uniform(0.0, 1.0)
                elif key == 'weight':
                    new_transform[key] = random.uniform(0.5, 1.5)
        
        # For variations, apply similar logic
        if 'variations' in transform:
            new_variations = {}
            for var_name, weight in transform['variations'].items():
                if isinstance(weight, (int, float)):
                    if isinstance(weight, int):
                        perturbation = int(random.uniform(-abs(weight)*perturbation_factor, abs(weight)*perturbation_factor))
                        new_variations[var_name] = max(0, weight + perturbation)  # Ensure non-negative
                    else:  # float
                        perturbation = random.uniform(-abs(weight)*perturbation_factor, abs(weight)*perturbation_factor)
                        new_variations[var_name] = max(0, weight + perturbation)  # Ensure non-negative
                else:
                    new_variations[var_name] = weight
            new_transform['variations'] = new_variations
        else:
            new_transform['variations'] = {}
        
        new_params['transforms'].append(new_transform)
    
    return new_params


def generate_second_dataset(interesting_param_sets, output_dir, num_per_interesting=10):
    """
    Generate a second dataset using refined/perturbed parameters.
    """
    import uuid
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(interesting_param_sets) * num_per_interesting} renders for second dataset...")
    
    all_new_params = []
    
    for orig_params in interesting_param_sets:
        for i in range(num_per_interesting):
            new_seed = f"refined_{orig_params['seed']}_{i}"
            new_params = perturb_parameters(orig_params, 0.15)  # Slightly larger perturbation
            new_params['seed'] = new_seed
            
            all_new_params.append(new_params)
    
    # Save all new parameter sets to a file (so they can be used by the generator)
    params_file = Path(output_dir) / "refined_parameters.json"
    with open(params_file, 'w') as f:
        json.dump(all_new_params, f, indent=2)
    
    print(f"Saved {len(all_new_params)} refined parameter sets to {params_file}")
    return all_new_params


def create_documentation_report(interesting_with_params, output_path):
    """
    Create detailed documentation about the findings.
    """
    report_content = f"""
# Flame Fractal Analysis Report

## Summary
Analyzed {len(interesting_with_params)} potentially interesting flame fractals from a larger dataset.

## Interesting Categories Found
- Jewelry/Pendant-like: Images with ornamental, crystalline, or gem-like qualities
- Landscape/Organic: Images with natural, flowing, or biological appearances
- Cosmic/Nebula: Images with celestial, starry, or galactic qualities
- Symmetrical/Geometric: Images with balanced, radial, or patterned structures
- Chaotic/Complex: Images with intricate, dynamic, or swirling patterns

## Top Examples

"""
    
    for i, img in enumerate(interesting_with_params[:20]):  # Show top 20
        category = []
        if img['is_jewelry_like']: category.append('jewelry')
        if img['is_nature_like']: category.append('nature')
        if img['is_cosmic_like']: category.append('cosmic')
        if img['is_symmetric']: category.append('symmetric')
        if img['is_complex']: category.append('complex')
        
        report_content += f"### Image {i+1}: {img['filename']}\n"
        report_content += f"- **Categories**: {', '.join(category)}\n"
        report_content += f"- **Description**: {img['description']}\n\n"
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    print(f"Analysis report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze flame fractal classifications and refine parameters')
    parser.add_argument('--input-csv', type=str, required=True, help='Input CSV file with classifications')
    parser.add_argument('--input-base-dir', type=str, required=True, help='Base directory where image and param files are located')
    parser.add_argument('--output-refined-params-dir', type=str, required=True, help='Output directory for refined parameters')
    parser.add_argument('--output-report', type=str, required=True, help='Output analysis report file')
    parser.add_argument('--refine-count', type=int, default=10, help='Number of refined variants to create per interesting image')
    
    args = parser.parse_args()
    
    # Analyze classifications to identify interesting images
    interesting_images, df = analyze_classifications(args.input_csv)
    
    # Extract parameters for interesting images
    interesting_with_params = extract_parameters_for_images(interesting_images, args.input_base_dir)
    
    print(f"Found parameters for {len(interesting_with_params)} interesting images")
    
    # Extract just the parameter dictionaries
    interesting_param_dicts = [img['parameters'] for img in interesting_with_params]
    
    # Generate second dataset with perturbed parameters
    refined_params = generate_second_dataset(interesting_param_dicts, args.output_refined_params_dir, args.refine_count)
    
    # Create analysis report
    create_documentation_report(interesting_with_params, args.output_report)
    
    print("Analysis and parameter refinement complete!")


if __name__ == "__main__":
    main()