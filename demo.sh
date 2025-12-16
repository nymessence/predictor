#!/bin/bash
# Example script to run the year-by-year AI predictor
# This script demonstrates the exact command requested in the prompt

echo "Running the requested command:"
echo "predictor.py --model \"tngtech/deepseek-r1t2-chimera:free\" --api-key \$OPENROUTER_API_KEY --api-endpoint \"https://openrouter.ai/api/v1\" --years 100 --predict \"Future of AI\" --output \"AI_prediction.json\" --token-limit 32000"
echo ""
echo "Enhanced Features:"
echo "- Auto-retry mechanism for temporary server issues"
echo "- Resume functionality to continue where it left off"
echo "- Progress saved after each successful prediction"
echo ""
echo "For this demo, we'll use dry-run mode since we don't have an actual API key:"

echo ""
echo "Command:"
echo "/home/erick/predictor/venv/bin/python /home/erick/predictor/predictor.py --model \"tngtech/deepseek-r1t2-chimera:free\" --api-key \"\$OPENROUTER_API_KEY\" --api-endpoint \"https://openrouter.ai/api/v1\" --years 5 --predict \"Future of AI\" --output \"demo_output.json\" --token-limit 32000 --dry-run"

echo ""
echo "Running in dry-run mode to show prompts without calling the API..."
echo ""

/home/erick/predictor/venv/bin/python /home/erick/predictor/predictor.py --model "tngtech/deepseek-r1t2-chimera:free" --api-key '$OPENROUTER_API_KEY' --api-endpoint "https://openrouter.ai/api/v1" --years 5 --predict "Future of AI" --output "demo_output.json" --token-limit 32000 --dry-run