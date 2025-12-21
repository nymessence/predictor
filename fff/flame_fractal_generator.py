#!/usr/bin/env python3
"""
Flame Fractal Generator

Generates random flame fractals with deterministic parameters saved for reproducibility.
Based on the flame fractal algorithm (flame equation variations).
"""
import os
import json
import argparse
import math
import numpy as np
import random
from PIL import Image
import uuid
from tqdm import tqdm
import time


def apply_variation(x, y, var_name):
    """
    Apply a specific flame fractal variation to the coordinates.
    """
    try:
        if var_name == 'linear':
            return x, y
        elif var_name == 'sinusoidal':
            return math.sin(x), math.sin(y)
        elif var_name == 'spherical':
            r = x*x + y*y
            r = max(1e-6, r)  # Avoid division by zero
            x_new = x / r
            y_new = y / r
            return x_new, y_new
        elif var_name == 'swirl':
            r = x*x + y*y
            sin_r = math.sin(r)
            cos_r = math.cos(r)
            x_new = x*sin_r - y*cos_r
            y_new = x*cos_r + y*sin_r
            return x_new, y_new
        elif var_name == 'horseshoe':
            r = math.sqrt(max(0, x*x + y*y))
            if r == 0:
                return 0.0, 0.0
            x_new = (x - y) * (x + y) / r
            y_new = 2 * x * y / r
            return x_new, y_new
        elif var_name == 'polar':
            theta = math.atan2(y, x)
            r = math.sqrt(max(0, x*x + y*y))
            x_new = theta / math.pi
            y_new = r - 1
            return x_new, y_new
        elif var_name == 'handkerchief':
            r = math.sqrt(max(0, x*x + y*y))
            theta = math.atan2(y, x)
            x_new = r * math.sin(theta + r)
            y_new = r * math.cos(theta - r)
            return x_new, y_new
        elif var_name == 'heart':
            r = math.sqrt(max(0, x*x + y*y))
            theta = math.atan2(y, x)
            x_new = r * math.sin(theta * r)
            y_new = -r * math.cos(theta * r)
            return x_new, y_new
        elif var_name == 'disc':
            r = math.sqrt(max(0, x*x + y*y))
            theta = math.atan2(y, x)
            if r > 0:
                theta_prime = theta / math.pi
                x_new = theta_prime * math.sin(math.pi * r)
                y_new = theta_prime * math.cos(math.pi * r)
            else:
                x_new = 0.0
                y_new = 0.0
            return x_new, y_new
        elif var_name == 'spiral':
            r = math.sqrt(max(0, x*x + y*y))
            theta = math.atan2(y, x)
            if r == 0:
                return 0.0, 0.0
            x_new = (1/r) * (math.cos(theta) + math.sin(r))
            y_new = (1/r) * (math.sin(theta) - math.cos(r))
            return x_new, y_new
        elif var_name == 'hyperbolic':
            r = math.sqrt(max(0, x*x + y*y))
            if r == 0:
                return 0.0, 0.0
            theta = math.atan2(y, x)
            x_new = math.sin(theta) / r
            y_new = r * math.cos(theta)
            return x_new, y_new
        else:
            # Unknown variation - just return original values
            return x, y
    except:
        # Handle any mathematical errors gracefully
        return x, y


def render_flame(params, width, height, max_iter=1000000):
    """
    Render a flame fractal based on the given parameters.
    """
    # Create density matrix to accumulate points
    density = np.zeros((height, width, 3), dtype=np.float64)
    
    # Define coordinate ranges
    min_x, max_x = -2.0, 2.0
    min_y, max_y = -2.0, 2.0
    
    # Initialize starting point
    x, y = 0.0, 0.0
    color_idx = 0.0
    
    for i in range(min(max_iter, params.get('num_points', 1000000))):
        # Select a transform randomly based on weighted probabilities
        weights = [t.get('weight', 1.0) for t in params['transforms']]
        total_weight = sum(weights)
        if total_weight > 0:
            # Normalize weights
            weights = [w / total_weight for w in weights]
        else:
            # If no weights, make them all equal
            weights = [1.0 / len(params['transforms']) for _ in params['transforms']]
        
        # Choose transform based on weights
        rand_val = random.random()
        cumulative = 0.0
        idx = 0
        for j, w in enumerate(weights):
            cumulative += w
            if rand_val <= cumulative:
                idx = j
                break
        
        transform = params['transforms'][idx]
        
        # Apply affine transformation
        new_x = transform['a']*x + transform['b']*y + transform['e']
        new_y = transform['c']*x + transform['d']*y + transform['f']
        
        # Apply variations
        for var_name, weight in transform.get('variations', {}).items():
            if weight > 0:
                var_x, var_y = apply_variation(new_x, new_y, var_name)
                new_x = new_x * (1.0 - weight) + var_x * weight
                new_y = new_y * (1.0 - weight) + var_y * weight
        
        x, y = new_x, new_y
        color_idx = (color_idx + 0.3) % 1.0  # Cycle through color indices
        
        # Map to pixel coordinates
        px = int((x - min_x) / (max_x - min_x) * width)
        py = int((y - min_y) / (max_y - min_y) * height)
        
        # Check bounds
        if 0 <= px < width and 0 <= py < height:
            # Use HSV color cycling for interesting effects
            hue = (color_idx * 360.0 + params.get('hue_rotation', 0)) % 360.0
            saturation = 0.8
            value = 1.0
            # Convert HSV to RGB approximately
            h_i = int(hue // 60) % 6
            f = hue / 60 - h_i
            p = value * (1 - saturation)
            q = value * (1 - f * saturation)
            t = value * (1 - (1 - f) * saturation)
            
            if h_i == 0:
                r, g, b = value, t, p
            elif h_i == 1:
                r, g, b = q, value, p
            elif h_i == 2:
                r, g, b = p, value, t
            elif h_i == 3:
                r, g, b = p, q, value
            elif h_i == 4:
                r, g, b = t, p, value
            else:
                r, g, b = value, p, q
            
            # Add to density
            density[py, px, 0] += r * 255
            density[py, px, 1] += g * 255
            density[py, px, 2] += b * 255
    
    # Apply gamma correction and normalize
    gamma = params.get('gamma', 2.0)
    brightness = params.get('brightness', 1.0)
    contrast = params.get('contrast', 1.0)
    
    density = np.power(density, 1.0/gamma)
    max_density = np.max(density)
    if max_density > 0:
        density = (density / max_density) * 255.0 * brightness * contrast
    
    # Clip values and convert to uint8
    density = np.clip(density, 0, 255)
    img_array = density.astype(np.uint8)
    
    # Create final image
    img = Image.fromarray(img_array, 'RGB')
    return np.array(img)


def generate_fractal_params(seed):
    """
    Generate deterministic flame fractal parameters based on a seed.
    """
    random.seed(seed)
    
    # Number of transforms (functions) in the flame fractal
    num_transforms = random.randint(2, 6)
    
    transforms = []
    for i in range(num_transforms):
        # Affine transformation parameters
        a = random.uniform(-1.5, 1.5)
        b = random.uniform(-1.5, 1.5)
        c = random.uniform(-1.5, 1.5)
        d = random.uniform(-1.5, 1.5)
        e = random.uniform(-1.5, 1.5)
        f = random.uniform(-1.5, 1.5)
        
        # Color weight
        color = random.uniform(0.0, 1.0)
        
        # Variation weights (simple variations for efficiency)
        variations = {}
        variation_list = [
            'linear', 'sinusoidal', 'spherical', 'swirl', 'horseshoe', 
            'polar', 'handkerchief', 'heart', 'disc', 'spiral', 'hyperbolic'
        ]
        # Select random 2-4 variations to use
        selected_variations = random.sample(variation_list, random.randint(2, min(4, len(variation_list))))
        
        for var in selected_variations:
            variations[var] = random.uniform(0.0, 1.0) if random.random() > 0.3 else 0.0
        
        transforms.append({
            'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f,
            'color': color,
            'variations': variations,
            'weight': random.uniform(0.5, 2.0)  # Transform weight/probability
        })
    
    # Additional parameters
    num_points = random.randint(1000000, 5000000)  # Number of points to plot
    gamma = random.uniform(1.0, 4.0)
    brightness = random.uniform(0.5, 2.0)
    contrast = random.uniform(0.5, 2.0)
    hue_rotation = random.uniform(0.0, 360.0)
    
    params = {
        'num_transforms': num_transforms,
        'transforms': transforms,
        'num_points': num_points,
        'gamma': gamma,
        'brightness': brightness,
        'contrast': contrast,
        'hue_rotation': hue_rotation
    }
    
    return params


def save_image_with_params(img_array, output_dir, filename, params, seed):
    """
    Save image and parameters to JSON file.
    """
    # Create image
    img = Image.fromarray(img_array, 'RGB')
    
    # Save image
    img_path = os.path.join(output_dir, f"{filename}.png")
    img.save(img_path)
    
    # Save parameters
    params_path = os.path.join(output_dir, f"{filename}_params.json")
    with open(params_path, 'w') as f:
        json.dump({
            'seed': seed,
            'parameters': params
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate random flame fractals')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--resolution', type=int, default=256, help='Image resolution')
    parser.add_argument('--renders', type=int, default=100, help='Number of fractals to render')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Generating {args.renders} flame fractals at {args.resolution}x{args.resolution} resolution...")
    
    for i in tqdm(range(args.renders), desc="Rendering"):
        seed = f"{uuid.uuid4()}-{i}"
        
        # Generate parameters
        params = generate_fractal_params(seed)
        
        # Render fractal
        try:
            img_array = render_flame(params, args.resolution, args.resolution, max_iter=min(1000000, params['num_points']))
            save_image_with_params(img_array, args.output, f"fractal_{i:05d}", params, seed)
        except Exception as e:
            print(f"Error rendering fractal {i}: {e}")
            import traceback
            traceback.print_exc()
            # Still save parameters even if rendering failed
            params_path = os.path.join(args.output, f"fractal_{i:05d}_params.json")
            with open(params_path, 'w') as f:
                json.dump({
                    'seed': seed,
                    'parameters': params,
                    'error': str(e)
                }, f, indent=2)
        
        # Periodic checkpoint every 500 renders
        if (i + 1) % 500 == 0:
            print(f"Checkpoint at {i+1}/{args.renders} renders")
    
    print("Rendering complete!")


if __name__ == "__main__":
    main()