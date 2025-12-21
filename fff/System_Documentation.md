# Flame Fractal Generation and Analysis System

## Overview

The Flame Fractal Generation and Analysis System is a complete pipeline for generating, classifying, and refining flame fractal images using AI character interactions. The system creates complex fractal art through AI-guided generation, analyzes the visual properties using vision models, and refines parameters to produce more compelling results.

## Components

### 1. Flame Fractal Generator (`flame_fractal_generator.py`)
The generator creates flame fractals using iterated function systems with multiple transformations. Each fractal is generated with a deterministic seed for reproducibility.

**Features:**
- Random parameter generation based on seed
- Deterministic rendering for reproducible results
- Simultaneous parameter storage with image files
- Variable resolution support (default: 256x256)
- Approximately 1 million to 5 million points per render

**Parameters Generated:**
- Number of transforms: 2-6 affine transformations
- Affine coefficients (a, b, c, d, e, f) per transform
- Color weights for each transform
- Variation weights for visual effects (sinusoidal, spherical, swirl, etc.)
- Rendering parameters (gamma, brightness, contrast, hue rotation)

**Usage:**
```bash
uv run fff/flame_fractal_generator.py --output fff/renders_1 --resolution 256 --renders 10000
```

### 2. Flame Fractal Classifier (`flame_fractal_classifier.py`)
The classifier processes fractal images in batches of 49 (7x7 grids) using vision models to generate detailed descriptions of visual properties.

**Features:**
- Batch processing in 7x7 grids (49 images per API call)
- Detailed visual descriptions focusing on form, structure, color, texture
- Pattern recognition for visual motifs (jewelry-like, cosmic, organic, etc.)
- Token management to handle context limits
- Structured CSV output with filename and description

**Analysis Categories:**
- Jewelry/Pendant-like: Ornamental, crystalline, gem-like qualities
- Landscape/Organic: Natural, flowing, biological appearances
- Cosmic/Nebula: Celestial, starry, galactic qualities
- Symmetrical/Geometric: Balanced, radial, patterned structures
- Chaotic/Complex: Intricate, dynamic, swirling patterns

**Usage:**
```bash
uv run fff/flame_fractal_classifier.py --input fff/renders_1 --output fff/classification_map.csv --classifier-work-dir fff/renders_1_classifier --api-endpoint "https://api.z.ai/api/paas/v4" --model "glm-4.6v-flash" --api-key $Z_AI_API_KEY
```

### 3. Flame Fractal Analyzer (`flame_fractal_analyzer.py`)
The analyzer reviews classified images to identify visually compelling specimens and creates refined parameter sets for enhanced generation.

**Features:**
- Pattern matching for interesting visual motifs
- Parameter perturbation for variation generation
- Refinement algorithm to enhance desirable properties
- Detailed analysis reports
- Structured parameter output for generator consumption

**Refinement Process:**
1. Identifies fractals with jewelry-like, cosmic, organic, symmetric, or complex properties
2. Extracts parameters for each interesting specimen
3. Creates perturbed variations with slight parameter adjustments (±10-15%)
4. Generates multiple refined versions of each interesting fractal
5. Outputs refined parameter sets for second-generation rendering

**Usage:**
```bash
uv run fff/flame_fractal_analyzer.py --input-csv fff/classification_map.csv --input-base-dir fff/renders_1 --output-refined-params-dir fff/renders_2 --output-report fff/reports/analysis_report.md --refine-count 10
```

## Pipeline Workflow

### Phase 1: Large-Scale Random Generation
1. Generate 10,000 fractals with random parameters
2. Store each fractal as PNG with corresponding JSON parameter file
3. Use deterministic seeds for reproducibility
4. Maintain aspect ratio and consistent rendering parameters

### Phase 2: Visual Classification and Mapping
1. Group images in 7x7 grids (49 per batch)
2. Submit grids to vision model with structured request
3. Extract detailed visual descriptions for each square
4. Map descriptions to original filenames
5. Store results in structured CSV format
6. Maintain checkpoint system every 1,000 classified images

### Phase 3: Parameter Refinement and Second Generation
1. Analyze CSV for fractals with interesting visual properties
2. Identify motifs matching target categories
3. Extract parameters for interesting specimens
4. Generate perturbed parameter sets (10 variations per interesting fractal)
5. Create refined parameter files for second-generation rendering
6. Validate parameter integrity and format

### Phase 4: Documentation and Tooling
1. Comprehensive analysis report creation
2. Parameter format documentation
3. Motif category definitions
4. Reproduction instructions
5. Troubleshooting guidance

## Technical Implementation

### Algorithmic Foundation
The flame fractal algorithm uses iterated function systems with a chaos game approach:
- Starts with initial point (typically origin)
- Randomly selects transformation based on weighted probabilities
- Applies affine transformation to current point
- Optionally applies non-linear variations
- Accumulates points on density grid with color cycling
- Applies gamma correction and normalization

### Variation Functions Implemented
- Linear: Identity mapping
- Sinusoidal: sin(x), sin(y)
- Spherical: Inversion through origin
- Swirl: Circular distortion
- Horseshoe: Cylindrical folding
- Polar: Rectangular to polar conversion
- Handkerchief: Angular-radial mixing
- Heart: Polar-angular transformation
- Disc: Angular compression
- Spiral: Radial-angular mixing
- Hyperbolic: Reciprocal-angular mapping

### Token Management System
The system implements a sliding window approach:
- Calculates token count based on word frequency (approx. 1.3 tokens per English word)
- Limits history to 1000 tokens maximum
- Preserves initial context (Narrator entries) when possible
- Maintains chronological order of recent entries
- Prevents overflow and repetitive loops

### Error Handling and Resilience
- API rate limit management with exponential backoff
- Connection timeout handling with retries
- Parameter validation for mathematical stability
- Backup responses for API failures
- Progress checkpoints to resume interrupted processes
- Graceful degradation for invalid inputs

## Command-Line Interface

### Generator Parameters
- `--output`: Output directory for fractals and parameters
- `--resolution`: Image resolution (width and height)
- `--renders`: Number of fractals to generate

### Classifier Parameters
- `--input`: Input directory with fractal images
- `--output`: Output CSV file for classifications  
- `--classifier-work-dir`: Temporary directory for batch processing
- `--api-endpoint`: Vision model API endpoint
- `--model`: Model identifier
- `--api-key`: Authentication key

### Analyzer Parameters
- `--input-csv`: Classification results CSV file
- `--input-base-dir`: Directory where original fractals are stored
- `--output-refined-params-dir`: Directory for refined parameter files
- `--output-report`: Analysis report file path
- `--refine-count`: Number of refined variations per interesting fractal

## Visual Motif Categories

### Jewelry/Pendant-Like Properties
- Crystalline or gem-like structures
- Metallic highlights or reflections
- Symmetrical ornamental patterns
- Intricate decorative elements
- Highly detailed formations

### Landscape/Organic Properties
- Flowing or curving structures
- Biological or plant-like forms
- Natural color palettes
- Growth-like patterns
- Ecosystem-like compositions

### Cosmic/Nebula Properties
- Stellar or galactic formations
- Gas cloud-like distributions
- Deep space color schemes
- Star cluster arrangements
- Nebular color blending

### Symmetrical/Geometric Properties
- Radial or bilateral symmetry
- Geometric pattern formation
- Mathematical precision
- Repeating geometric elements
- Crystallographic properties

### Chaotic/Complex Properties
- Dense intricate patterns
- Dynamic swirling formations
- Non-repeating structures
- High information density
- Organic complexity

## Reproducibility Features

Each fractal includes:
- Deterministic seed for parameter generation
- Complete parameter file with all coefficients
- Unambiguous reproduction instructions
- Validation checksums for parameters

## Performance Considerations

- Rendering time: 5-10 seconds per fractal at 256x256 resolution
- Batch processing: 49 images per API call during classification
- API rate limiting: Built-in delays to respect limits
- Memory management: Streaming processing to minimize RAM usage
- Storage efficiency: Separate image and parameter files for easy access

## Troubleshooting

### Common Issues
- API Rate Limits: System includes exponential backoff and retry logic
- Resolution Scaling: Adjust resolution based on computational resources
- Batch Size Adjustment: Modify batch size if API has different limits
- Network Failures: Built-in retry mechanisms for common failures
- Parameter Validation: Automatic adjustment of unstable parameters

### Performance Tuning
- Lower resolution for faster generation during testing
- Reduce number of points for quicker rendering
- Adjust batch size for optimal API usage
- Increase similarity thresholds to reduce repetitive patterns
- Use local models if available to avoid API limits

This system enables automated discovery and enhancement of aesthetically pleasing flame fractal configurations through AI-assisted analysis of visual motifs.