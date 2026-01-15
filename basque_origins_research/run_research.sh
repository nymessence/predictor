#!/bin/bash
# Script to run the Basque Origins Research System

# Activate the virtual environment
source .venv/bin/activate

# Run the main research script
python main.py "$@"