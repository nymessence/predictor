#!/bin/bash
# Autonomous Chess Debugging Loop

set -e  # Exit on error

echo "🤖 Starting Autonomous Chess Game Debugger..."
echo "📅 $(date)"

FAILED_RUNS=0
TOTAL_RUNS=0
OUTPUT_FILE="character_interactions/json/Nya_&_Azalea_Chess_debug.json"

# Ensure the failure log exists
mkdir -p .qwen
touch .qwen/FAILURES.md

while true; do
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    RUN_ID="chess-debug-$(printf '%03d' $TOTAL_RUNS)"
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo ""
    echo "══════════════════════════════════════════════════════════════════════════════"
    echo "🔄 RUN #$TOTAL_RUNS: $TIMESTAMP"
    echo "══════════════════════════════════════════════════════════════════════════════"
    
    # Execute the canonical command with timeout
    timeout 900s uv run --active character_interactions/main.py \
        "character_interactions/json/nya_elyria.json" \
        "character_interactions/json/empress_azalea.json" \
        --chess \
        --delay 5 \
        --similarity 0.65 \
        --api-endpoint "https://api.z.ai/api/paas/v4" \
        --model "glm-4.6v-flash" \
        --api-key $Z_AI_API_KEY \
        -o "$OUTPUT_FILE" \
        2>&1 | tee "debug_run_${RUN_ID}.log"

    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 124 ]; then
        # Timeout occurred
        echo "⏰ Timeout reached (15 min limit)"
        echo "" >> .qwen/FAILURES.md
        echo "## Entry $(grep -c '^## Entry' .qwen/FAILURES.md | awk '{print $2+1}'):" >> .qwen/FAILURES.md
        echo "Timestamp: $TIMESTAMP" >> .qwen/FAILURES.md
        echo "Run ID: $RUN_ID" >> .qwen/FAILURES.md
        echo "Symptom: Process timed out after 15 minutes" >> .qwen/FAILURES.md
        echo "Root Cause: Game likely stuck in infinite loop or extremely slow" >> .qwen/FAILURES.md
        echo "Fix Applied: None yet" >> .qwen/FAILURES.md
        echo "Verification: Still experiencing issues" >> .qwen/FAILURES.md
        echo "Status: Active" >> .qwen/FAILURES.md
        
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "❌ Run failed due to timeout (failed runs: $FAILED_RUNS)"
    elif [ $EXIT_CODE -ne 0 ]; then
        # Other error occurred
        echo "💥 Run #$TOTAL_RUNS failed with exit code: $EXIT_CODE"
        echo "" >> .qwen/FAILURES.md
        echo "## Entry $(grep -c '^## Entry' .qwen/FAILURES.md | awk '{print $2+1}'):" >> .qwen/FAILURES.md
        echo "Timestamp: $TIMESTAMP" >> .qwen/FAILURES.md
        echo "Run ID: $RUN_ID" >> .qwen/FAILURES.md
        echo "Symptom: Process failed with exit code $EXIT_CODE" >> .qwen/FAILURES.md
        echo "Root Cause: Error in execution" >> .qwen/FAILURES.md
        echo "Fix Applied: None yet" >> .qwen/FAILURES.md
        echo "Verification: Still experiencing issues" >> .qwen/FAILURES.md
        echo "Status: Active" >> .qwen/FAILURES.md
        
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "❌ Run failed due to error (failed runs: $FAILED_RUNS)"
    else
        # Success - check if game progressed properly
        if [ -f "$OUTPUT_FILE" ]; then
            LINES_COUNT=$(wc -l < "$OUTPUT_FILE")
            TURNS_COUNT=$(grep -c "turn" "$OUTPUT_FILE" || echo "0")
            
            echo "📊 Output file stats: $LINES_COUNT lines, ~$TURNS_COUNT turns"
            
            if [ $TURNS_COUNT -gt 10 ]; then
                echo "✅ Game progressed normally ($TURNS_COUNT turns)"
                # If we have a successful run with good progress, we can analyze if it's truly fixed
                # For now, just continue the loop
            else
                echo "🤔 Game may not have progressed sufficiently ($TURNS_COUNT turns)"
                echo "" >> .qwen/FAILURES.md
                echo "## Entry $(grep -c '^## Entry' .qwen/FAILURES.md | awk '{print $2+1}'):" >> .qwen/FAILURES.md
                echo "Timestamp: $TIMESTAMP" >> .qwen/FAILURES.md
                echo "Run ID: $RUN_ID" >> .qwen/FAILURES.md
                echo "Symptom: Game completed but made insufficient progress ($TURNS_COUNT turns)" >> .qwen/FAILURES.md
                echo "Root Cause: May still have underlying issues" >> .qwen/FAILURES.md
                echo "Fix Applied: None yet" >> .qwen/FAILURES.md
                echo "Verification: Needs more turns to be considered fixed" >> .qwen/FAILURES.md
                echo "Status: Active" >> .qwen/FAILURES.md
                
                FAILED_RUNS=$((FAILED_RUNS + 1))
            fi
        else
            echo "❌ Output file was not created"
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
    fi
    
    echo "📈 Run Stats - Total: $TOTAL_RUNS, Failed: $FAILED_RUNS, Success Rate: $(( (TOTAL_RUNS-FAILED_RUNS)*100/TOTAL_RUNS ))%"
    
    # Small delay before next run
    sleep 5
    
    if [ $FAILED_RUNS -gt 10 ]; then
        echo "🚨 Too many failures ($FAILED_RUNS). Stopping for manual inspection."
        break
    fi
done