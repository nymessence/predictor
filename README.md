# Year-by-Year AI Predictor

This tool generates year-by-year predictions for a given topic using AI models through an API service like OpenRouter.

## Features

- Cycles through years making predictions for each year
- Uses context from previous years to inform later predictions
- Configurable model, API endpoint, and prediction parameters

## Installation

Make sure you have Python 3.7+ installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python predictor.py --model "MODEL_IDENTIFIER" \
                    --api-key "YOUR_API_KEY" \
                    --api-endpoint "API_ENDPOINT_URL" \
                    --years NUMBER_OF_YEARS \
                    --predict "TOPIC_TO_PREDICT"
```

### Arguments

- `--model`: Model identifier (e.g., "tngtech/deepseek-r1t2-chimera:free")
- `--api-key`: API key for the model service
- `--api-endpoint`: API endpoint URL (e.g., "https://openrouter.ai/api/v1")
- `--years`: Number of years to predict
- `--predict`: Topic to predict (e.g., "Future of AI")
- `--start-year`: Starting year for predictions (default: next year)
- `--token-limit`: Token limit for context (default: 32000)
- `--output`: Output file to save predictions in JSON format
- `--chinese-penalty`: Apply logit bias to penalize Chinese characters in responses

### Example

```bash
python predictor.py --model "tngtech/deepseek-r1t2-chimera:free" \
                    --api-key $OPENROUTER_API_KEY \
                    --api-endpoint "https://openrouter.ai/api/v1" \
                    --years 20 \
                    --predict "Future of AI"
```

### Full Example with all options

```bash
python predictor.py --model "tngtech/deepseek-r1t2-chimera:free" \
                    --api-key $OPENROUTER_API_KEY \
                    --api-endpoint "https://openrouter.ai/api/v1" \
                    --years 100 \
                    --predict "Future of AI" \
                    --output "AI_prediction.json" \
                    --token-limit 32000
```

## Notes

- The API key should be kept secure and never committed to version control
- Consider rate limits when making many requests to the API
- Previous predictions are included in context for later predictions
- The tool may take some time to run depending on the number of years and API response times