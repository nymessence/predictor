# Failure Ledger
Persistent record of system failures and fixes

## Entry 1
Timestamp: 2025-12-23 04:49:00
Run ID: chess-debug-001
Symptom: Characters not providing moves in JSON format, causing game to stall
Root Cause: AI character responses don't follow required JSON schema with "move" field
Fix Applied: Enhanced JSON parsing with fallback regex patterns to detect chess notation
Verification: Partial - moves detected but still some stalling issues
Status: Active

## Entry 2
Timestamp: 2025-12-23 04:51:00
Run ID: chess-debug-002
Symptom: Consecutive failure tracking not properly resetting after valid moves
Root Cause: consecutive_failed_moves counter not reset when moves are successful
Fix Applied: Added counter reset for successful moves in all game modes
Verification: Improved, but still seeing occasional stalling
Status: Active