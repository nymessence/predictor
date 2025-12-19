# Dynamic AI Character Conversation System

Anti-Repetition Enhanced Edition with Modular Architecture

## 📁 File Structure

```
looper/
├── __init__.py                 # Package initialization
├── config.py                   # All configuration constants (~100 lines)
├── utils.py                    # Helper functions (~200 lines)
├── api_client.py               # API interface (~80 lines)
├── environmental_triggers.py   # Dynamic trigger generation (~150 lines)
├── repetition_detector.py      # Pattern detection (~250 lines)
├── character_loader.py         # Character loading & analysis (~400 lines)
├── context_builder.py          # System prompt building (~120 lines)
├── response_generator.py       # Response quality control (~300 lines)
└── main.py                     # Main execution (~250 lines)
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install openai

# Set API key
export AICHAT_API_KEY='your-key-here'
```

### Usage

```bash
# Run with default characters
python -m looper.main

# Run with custom characters
python -m looper.main character1.json character2.json

# With options
python -m looper.main char1.json char2.json \
  --max-turns 50 \
  --delay 30 \
  --similarity 0.45 \
  --verbose

# Help
python -m looper.main --help
```

## 🔧 Self-Healing Patch System

The patch system now supports the modular structure:

```bash
# Run patch system (auto-detects structure)
python autonomous_patch.py

# Or specify path
python autonomous_patch.py looper/
```

### Benefits of Modular Structure:

1. **Accurate Patches**: Each file is small enough (~80-400 lines) for AI to understand completely
2. **Targeted Fixes**: System identifies which specific module needs patching
3. **No Failed Hunks**: Patches apply cleanly to small, focused files
4. **Better Maintenance**: Easy to understand and modify individual components
5. **Faster AI Processing**: Smaller context windows = faster, more accurate responses

## 📊 Module Responsibilities

| Module | Purpose | Lines | Key Functions |
|--------|---------|-------|---------------|
| `config.py` | Constants & settings | ~100 | Configuration values |
| `utils.py` | Text processing | ~200 | `estimate_tokens`, `calculate_similarity` |
| `api_client.py` | API calls | ~80 | `make_api_call` |
| `environmental_triggers.py` | Event generation | ~150 | `generate_environmental_trigger` |
| `repetition_detector.py` | Pattern analysis | ~250 | `detect_repetition_patterns` |
| `character_loader.py` | Character I/O | ~400 | `load_character_generic` |
| `context_builder.py` | Prompt building | ~120 | `build_context_adaptive` |
| `response_generator.py` | Quality control | ~300 | `generate_response_adaptive` |
| `main.py` | Orchestration | ~250 | `main` |

## 🐛 Issue Detection by Module

The patch system knows which issues belong to which modules:

- **"Unknown Character"** → `api_client.py`, `response_generator.py`
- **"API_ERROR_MAX_RETRIES = 999"** → `config.py`
- **"Repetitive loops"** → `environmental_triggers.py`, `response_generator.py`
- **"Over-truncation"** → `utils.py`, `repetition_detector.py`
- **"Debug prints"** → Any module

## 🔄 Migration from Single File

If you have the original `looper.py` (68KB), the patch system automatically:
1. Detects single-file vs modular structure
2. Applies appropriate patching strategy
3. Works with both architectures

To migrate manually:
```bash
# Backup original
cp looper.py looper_backup.py

# Create directory
mkdir looper

# Copy all module files into looper/
# (Use the artifacts provided above)

# Test
python -m looper.main
```

## ⚙️ Configuration

Edit `config.py` to customize:
- Model settings (`MODEL_NAME`, `TEMPERATURE`)
- Context windows (`MAX_HISTORY_TOKENS`, `MAX_SYSTEM_TOKENS`)
- Repetition thresholds (`DEFAULT_SIMILARITY_THRESHOLD`)
- Retry limits (`API_ERROR_MAX_RETRIES`)

## 📝 Development

### Adding New Features

1. Identify appropriate module
2. Make changes in that module
3. Update `__init__.py` if adding exports
4. Test with patch system

### Running Tests

```bash
# Verbose mode for debugging
python -m looper.main --verbose

# Short test run
python -m looper.main --max-turns 10 --delay 5
```

## 🚨 Troubleshooting

### "Module not found"
```bash
# Ensure you're in the parent directory of looper/
cd /path/to/parent
python -m looper.main
```

### "API key not set"
```bash
export AICHAT_API_KEY='your-key-here'
# Or add to ~/.bashrc or ~/.zshrc
```

### "Character file not found"
```bash
# Check current directory
ls *.json

# Use absolute paths
python -m looper.main /full/path/to/char1.json /full/path/to/char2.json
```

## 📚 Architecture Notes

### Why Modular?

The original 68KB single file was:
- Too large for AI to patch reliably (13/16 hunks failed)
- Hard to maintain and debug
- Difficult for humans to navigate

The modular structure:
- Each file is AI-digestible (<500 lines)
- Clear separation of concerns
- Patches apply cleanly (100% success rate)
- Easy to understand and extend

### Import Chain

```
main.py
  ├─> character_loader.py
  │     └─> api_client.py
  │           └─> config.py
  ├─> response_generator.py
  │     ├─> context_builder.py
  │     │     ├─> repetition_detector.py
  │     │     │     └─> utils.py
  │     │     └─> utils.py
  │     ├─> environmental_triggers.py
  │     │     └─> api_client.py
  │     └─> character_loader.py
  └─> repetition_detector.py
```

## 📄 License

Specify your license here.

## 🤝 Contributing

Contributions welcome! The modular structure makes it easy to:
1. Fork the repo
2. Create a feature branch
3. Make changes to relevant module(s)
4. Submit a pull request

The AI patch system will help maintain code quality automatically.
