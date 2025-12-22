#!/usr/bin/env python3
"""
Flame Fractal Generator (Robust Version)
Handles NaN/Infinity errors and ensures numerical stability.
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

def apply_variation(x, y, var_name):
    """
    Apply a specific flame fractal variation with safeguards against 
    division by zero or extremely small numbers.
    """
    try:
        r_sq = x*x + y*y
        r = math.sqrt(max(1e-9, r_sq))
        theta = math.atan2(y, x)

        if var_name == 'linear':
            return x, y
        elif var_name == 'sinusoidal':
            return math.sin(x), math.sin(y)
        elif var_name == 'spherical':
            return x / r_sq, y / r_sq
        elif var_name == 'swirl':
            sin_r2 = math.sin(r_sq)
            cos_r2 = math.cos(r_sq)
            return x*sin_r2 - y*cos_r2, x*cos_r2 + y*sin_r2
        elif var_name == 'horseshoe':
            return (x - y) * (x + y) / r, 2 * x * y / r
        elif var_name == 'polar':
            return theta / math.pi, r - 1.0
        elif var_name == 'handkerchief':
            return r * math.sin(theta + r), r * math.cos(theta - r)
        elif var_name == 'heart':
            return r * math.sin(theta * r), -r * math.cos(theta * r)
        elif var_name == 'disc':
            return (theta / math.pi) * math.sin(math.pi * r), (theta / math.pi) * math.cos(math.pi * r)
        elif var_name == 'spiral':
            return (1.0/r) * (math.cos(theta) + math.sin(r)), (1.0/r) * (math.sin(theta) - math.cos(r))
        elif var_name == 'hyperbolic':
            return math.sin(theta) / r, r * math.cos(theta)
        else:
            return x, y
    except (ZeroDivisionError, ValueError, OverflowError):
        return x, y

def render_flame(params, width, height, max_iter=1000000):
    """
    Render a flame fractal with coordinate validation.
    """
    density = np.zeros((height, width, 3), dtype=np.float64)
    
    # Standard view window
    min_x, max_x = -2.0, 2.0
    min_y, max_y = -2.0, 2.0
    
    # Chaos game state
    x, y = random.uniform(-1, 1), random.uniform(-1, 1)
    color_idx = 0.0
    
    transforms = params['transforms']
    weights = [t.get('weight', 1.0) for t in transforms]
    
    # Burn-in: skip first 20 iterations to settle onto the attractor
    for _ in range(20):
        t = random.choices(transforms, weights=weights)[0]
        nx = t['a']*x + t['b']*y + t['e']
        ny = t['c']*x + t['d']*y + t['f']
        for v_name, v_weight in t.get('variations', {}).items():
            if v_weight > 0:
                vx, vy = apply_variation(nx, ny, v_name)
                nx = nx * (1.0 - v_weight) + vx * v_weight
                ny = ny * (1.0 - v_weight) + vy * v_weight
        x, y = nx, ny

    for i in range(max_iter):
        t = random.choices(transforms, weights=weights)[0]
        
        # 1. Transform
        new_x = t['a']*x + t['b']*y + t['e']
        new_y = t['c']*x + t['d']*y + t['f']
        
        # 2. Variations
        for var_name, weight in t.get('variations', {}).items():
            if weight > 0:
                var_x, var_y = apply_variation(new_x, new_y, var_name)
                new_x = new_x * (1.0 - weight) + var_x * weight
                new_y = new_y * (1.0 - weight) + var_y * weight
        
        x, y = new_x, new_y
        
        # --- ROBUSTNESS CHECK ---
        if not math.isfinite(x) or not math.isfinite(y):
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            continue

        color_idx = (color_idx + t['color']) / 2.0
        
        # 3. Plotting
        if min_x <= x <= max_x and min_y <= y <= max_y:
            px = int((x - min_x) / (max_x - min_x) * (width - 1))
            py = int((y - min_y) / (max_y - min_y) * (height - 1))
            
            if 0 <= px < width and 0 <= py < height:
                # Approximate HSV to RGB
                hue = (color_idx * 360.0 + params.get('hue_rotation', 0)) % 360.0
                h_i = int(hue // 60) % 6
                f = hue / 60 - h_i
                v, s = 1.0, 0.8
                p, q, t_val = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
                
                rgb = [(v,t_val,p), (q,v,p), (p,v,t_val), (p,q,v), (t_val,p,v), (v,p,q)][h_i]
                
                density[py, px, 0] += rgb[0]
                density[py, px, 1] += rgb[1]
                density[py, px, 2] += rgb[2]

    # Post-processing: Log-density mapping & Gamma
    density = np.log1p(density) # Stabilizes brightness
    max_val = np.max(density)
    if max_val > 0:
        density = (density / max_val)
    
    # Apply user gamma/contrast
    density = np.power(density, 1.0 / params.get('gamma', 2.2))
    img_array = (np.clip(density * 255 * params.get('brightness', 1.0), 0, 255)).astype(np.uint8)
    return img_array

def generate_fractal_params(seed):
    random.seed(seed)
    num_transforms = random.randint(2, 5)
    
    transforms = []
    for _ in range(num_transforms):
        # Slightly tighter bounds (-1.0 to 1.0) for better stability
        transforms.append({
            'a': random.uniform(-1.0, 1.0), 'b': random.uniform(-1.0, 1.0),
            'c': random.uniform(-1.0, 1.0), 'd': random.uniform(-1.0, 1.0),
            'e': random.uniform(-1.5, 1.5), 'f': random.uniform(-1.5, 1.5),
            'color': random.random(),
            'weight': random.uniform(0.1, 1.0),
            'variations': {v: random.random() for v in random.sample(
                ['linear', 'sinusoidal', 'spherical', 'swirl', 'polar', 'heart'], 
                random.randint(1, 3)
            )}
        })
    
    return {
        'transforms': transforms,
        'num_points': random.randint(1000000, 3000000),
        'gamma': random.uniform(1.8, 2.5),
        'brightness': 1.2,
        'hue_rotation': random.uniform(0, 360)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--renders', type=int, default=10)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    for i in tqdm(range(args.renders), desc="Rendering"):
        current_seed = f"{uuid.uuid4()}"
        params = generate_fractal_params(current_seed)
        
        try:
            img_data = render_flame(params, args.resolution, args.resolution, max_iter=params['num_points'])
            img = Image.fromarray(img_data, 'RGB')
            img.save(os.path.join(args.output, f"fractal_{i:05d}.png"))
            
            with open(os.path.join(args.output, f"fractal_{i:05d}_params.json"), 'w') as f:
                json.dump({'seed': current_seed, 'params': params}, f, indent=2)
        except Exception as e:
            print(f"\nSkipping fractal {i} due to error: {e}")

if __name__ == "__main__":
    main()
