#!/usr/bin/env python3
"""
Chess Game Diagnostic Monitor
Watches game state for signs of stalling or corruption
"""

import json
import time
import os
import signal
import sys
from pathlib import Path
import hashlib

class ChessGameMonitor:
    def __init__(self, output_file):
        self.output_file = output_file
        self.previous_state_hash = None
        self.stall_count = 0
        self.max_stalls_before_kill = 5
        self.turn_count = 0
        self.previous_turn_count = 0
        
    def calculate_state_hash(self, history):
        """Calculate hash of recent history to detect repeated states"""
        if not history:
            return None
            
        # Consider last 5 moves for state comparison
        recent_moves = history[-5:] if len(history) >= 5 else history
        state_str = str(recent_moves)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def validate_output_json(self):
        """Check if output JSON is valid and progressing"""
        if not os.path.exists(self.output_file):
            return True, "File doesn't exist yet"
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if JSON is valid format
            if not isinstance(data, list):
                return False, f"JSON is not a list: {type(data)}"
            
            for item in data:
                if not isinstance(item, dict):
                    return False, f"Entry {data.index(item)} is not a dict: {type(item)}"
                if 'name' not in item or 'content' not in item:
                    return False, f"Entry {data.index(item)} missing required fields"
            
            # Count valid turns
            self.turn_count = len([item for item in data if isinstance(item, dict) and item.get('name')])
            
            # Check for state repetition
            current_hash = self.calculate_state_hash(data)
            if current_hash and self.previous_state_hash and current_hash == self.previous_state_hash:
                self.stall_count += 1
                if self.stall_count >= self.max_stalls_before_kill:
                    return False, f"State appears to be repeating for {self.stall_count} consecutive checks"
            else:
                self.stall_count = 0  # Reset if state changed
                
            self.previous_state_hash = current_hash
            return True, f"Valid JSON with {self.turn_count} turns"
            
        except json.JSONDecodeError as e:
            return False, f"JSON decode error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def check_for_problems(self):
        """Check for various problems in the output"""
        is_valid, message = self.validate_output_json()
        
        if not is_valid:
            print(f"🔴 MONITOR ALERT: {message}")
            return False, message
        else:
            print(f"🟢 Monitor OK: {message}")
            return True, message

def run_canonical_chess_simulation():
    """Run the canonical chess simulation with timeout monitoring"""
    import subprocess
    import threading
    
    # Canonical command from the meta prompt
    cmd = [
        "uv", "run", "--active", "character_interactions/main.py",
        "character_interactions/json/nya_elyria.json",
        "character_interactions/json/empress_azalea.json",
        "--chess",
        "--delay", "5",
        "--similarity", "0.65",
        "--api-endpoint", "https://api.z.ai/api/paas/v4",
        "--model", "glm-4.6v-flash",
        "--api-key", os.environ.get("Z_AI_API_KEY", ""),
        "-o", "character_interactions/json/Nya_&_Azalea_Chess_debug.json"
    ]
    
    # Remove empty API key if not provided
    if not os.environ.get("Z_AI_API_KEY"):
        print("❌ Z_AI_API_KEY environment variable not set")
        return False
    
    print("🚀 Starting canonical chess simulation...")
    print(f"Command: {' '.join(cmd[:-1])} [API_KEY_OMITTED]")
    
    # Start the process with timeout
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Start monitoring thread
        monitor = ChessGameMonitor("character_interactions/json/Nya_&_Azalea_Chess_debug.json")
        
        start_time = time.time()
        timeout = 15 * 60  # 15 minutes timeout
        
        while process.poll() is None:
            # Check if timeout exceeded
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"⏰ TIMEOUT: Killing process after {elapsed:.2f}s")
                process.kill()
                return False, f"Timeout after {elapsed:.2f}s"
            
            # Monitor output file periodically
            time.sleep(10)  # Check every 10 seconds
            is_ok, message = monitor.check_for_problems()
            
            if not is_ok:
                print(f"🛑 Monitoring detected problem: {message}")
                process.kill()
                return False, message
                
        # Process completed normally
        return_code = process.returncode
        print(f"🏁 Process completed with return code: {return_code}")
        return return_code == 0, f"Completed with return code: {return_code}"
        
    except Exception as e:
        print(f"💥 Process exception: {e}")
        return False, str(e)

if __name__ == "__main__":
    success, message = run_canonical_chess_simulation()
    if success:
        print(f"✅ Simulation completed successfully: {message}")
    else:
        print(f"❌ Simulation failed: {message}")
        sys.exit(1)