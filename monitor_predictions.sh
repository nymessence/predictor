#!/bin/bash
# Monitoring script for the autonomous research prediction system

LOG_FILE="/home/erick/predictor/autonomous_research/corrected_prediction_run.log"
PID_FILE="/home/erick/predictor/autonomous_research/prediction.pid"

# Save the process ID
echo 533701 > $PID_FILE

echo "Monitoring autonomous research prediction system..."
echo "PID: $(cat $PID_FILE)"
echo "Log file: $LOG_FILE"
echo "Started: $(date)"
echo "----------------------------------------"

while true; do
    # Check if process is still running
    if ps -p $(cat $PID_FILE) > /dev/null 2>&1; then
        echo "[$(date)] Process is running"
        echo "Latest log entries:"
        tail -5 $LOG_FILE
        echo "----------------------------------------"
    else
        echo "[$(date)] Process has stopped!"
        echo "Last 10 log entries:"
        tail -10 $LOG_FILE
        echo "----------------------------------------"
        exit 0
    fi

    # Wait 5 minutes before next check
    sleep 300
done