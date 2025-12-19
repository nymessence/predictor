#!/usr/bin/env python3
"""
Fully Autonomous Self-Healing Patch System with Dynamic Issue Prioritization
- Primary AI agent: high-severity code & JSON patches (returns diff patch)
- Secondary AI watchdog: 10-turn batch micro-patching for medium-severity issues
- Low-severity issues logged, upgraded if recurring
- Incremental simulation validation: 10 → 30 → 100 turns
- Automatic git commit and rollback
- Persistent API retry (Handles 5xx errors)
- Resumes progress after interruption (using RUN_STATE_FILE)
"""

import os
import json
import time
import random
import subprocess
from difflib import SequenceMatcher, unified_diff
import requests

# -----------------------
# CONFIG
# -----------------------
BASE_URL = "https://api.airforce"
API_KEY = os.environ.get("AIRFORCE_API_KEY")
COMPLETIONS_ENDPOINT = f"{BASE_URL}/v1/chat/completions"
MODEL_NAME = "deepseek-v3"
TEMPERATURE = 0.8
SIMILARITY_THRESHOLD = 0.65
MAX_HISTORY = 5
WATCHDOG_BATCH = 10 # Check every 10 turns
SIMULATION_TURNS = [10,30,100]
ISSUE_HISTORY_FILE = "issue_history.json"
RUN_STATE_FILE = "run_state.json" # File for checkpointing progress
MAX_OCCURRENCE_FOR_UPGRADE = 3
MAX_INACTIVE_FOR_DOWNGRADE = 5
MAX_PATCH_ATTEMPTS = 10

# -----------------------
# UTILS
# -----------------------
def sanitize_url(url):
    return url.strip().replace('[','').replace(']','').replace('(','').replace(')','')

COMPLETIONS_ENDPOINT = sanitize_url(COMPLETIONS_ENDPOINT)

def persistent_request(payload):
    """
    Makes a POST request with persistent retries on 429 (Rate Limit), 
    all 5xx Server Errors (like 502), and general connection issues.
    """
    while True:
        try:
            r = requests.post(COMPLETIONS_ENDPOINT,
                              headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                              json=payload,
                              timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            # Check for 429 (Rate Limit) OR 5xx (Server Errors: 500, 502, 503, 504, etc.)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                print(f"⚠️ Server error {r.status_code} ({r.reason}): Retrying in 5 seconds...")
                time.sleep(5)
            else:
                # Raise for client errors (4xx like 400, 401, 404)
                raise e
        except requests.RequestException as e:
            # Handles connection timeouts, DNS errors, etc.
            print(f"🌐 Connection error ({type(e).__name__}): Retrying in 2 seconds...")
            time.sleep(2)

def commit_changes(msg):
    subprocess.run(["git","add","."])
    subprocess.run(["git","commit","-m",msg])

def revert_last_commit():
    subprocess.run(["git","reset","--hard","HEAD~1"])

def validate_character_json(char_json):
    defaults = {"greeting": "Hello", "agenda": [], "emotional_state": "neutral"}
    for k,v in defaults.items():
        if k not in char_json:
            char_json[k] = v
    return char_json

def semantic_similarity(a,b):
    return SequenceMatcher(None,a,b).ratio()

def detect_loop(history,new_text):
    for prev in history[-MAX_HISTORY:]:
        if semantic_similarity(prev,new_text) >= SIMILARITY_THRESHOLD:
            return True
    return False

def inject_divergence():
    triggers = [
        "*A sudden memory flashes*", "*A loud noise startles them*",
        "*They glance at the ceiling*", "*An unexpected emotion crosses their face*"
    ]
    return random.choice(triggers)

def partial_loop_fix(history,new_text):
    if not history: return new_text
    last = history[-1]
    sm = SequenceMatcher(None,last,new_text)
    for block in reversed(sm.get_matching_blocks()):
        if block.size > 5:
            new_text = new_text[:block.a] + "*a sudden shift occurs*" + new_text[block.b:]
            break
    return new_text

def apply_patch_string(script_path, patch_content):
    """Applies a patch (diff) string to the specified file using the 'patch' command."""
    if not patch_content or "--- " not in patch_content:
        print("🚨 WARNING: Patch content is empty or invalid (missing '--- '). Skipping application.")
        return False
    
    try:
        # Use subprocess to pipe the patch content directly to the 'patch' command
        process = subprocess.run(
            ["patch", "-p0", script_path],
            input=patch_content.encode('utf-8'),
            capture_output=True,
            check=True # Will raise CalledProcessError if patch fails
        )
        print("✅ Patch applied successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Patch failed to apply (Error Code {e.returncode}). Reverting...")
        print(f"STDOUT: {e.stdout.decode().strip()}")
        print(f"STDERR: {e.stderr.decode().strip()}")
        return False

# -----------------------
# CHECKPOINTING
# -----------------------

def load_run_state():
    """Loads the last recorded run_number and the index of the last completed simulation turn."""
    if os.path.exists(RUN_STATE_FILE):
        try:
            with open(RUN_STATE_FILE) as f:
                state = json.load(f)
                return state.get("run_number", 1), state.get("turn_index", 0)
        except json.JSONDecodeError:
            print("Corrupted run state file. Starting from run 1.")
            return 1, 0
    return 1, 0

def save_run_state(run_number, turn_index):
    """Saves the current state of the patch loop."""
    state = {"run_number": run_number, "turn_index": turn_index}
    with open(RUN_STATE_FILE,"w") as f:
        json.dump(state,f,indent=2)


# -----------------------
# AI PATCH FUNCTIONS
# -----------------------
def find_relevant_code_sections(code_str, issues_log):
    """
    Extract relevant code sections based on issues, with surrounding context.
    Returns a dict mapping line ranges to code sections.
    """
    lines = code_str.splitlines()
    relevant_sections = {}
    
    # Search patterns based on common issues
    search_patterns = {
        "Unknown Character": ["Unknown Character", "fallback", "max_retries", "API_ERROR"],
        "repetitive physical loop": [".lean back", ".lean forward", "*leans", "*shifts"],
        "Anti-echo may over-truncate": ["truncate", "anti_echo", "anti-repetition", "similarity"],
        "Debug print": ["print(", "DEBUG:", "print f\""]
    }
    
    # Find all relevant line numbers
    relevant_lines = set()
    for issue_key, patterns in search_patterns.items():
        if any(issue_key.lower() in issues_log.lower() for issue_key in [issue_key]):
            for i, line in enumerate(lines):
                if any(pattern in line for pattern in patterns):
                    # Add this line plus context (10 lines before and after)
                    for j in range(max(0, i-10), min(len(lines), i+11)):
                        relevant_lines.add(j)
    
    # If we found too few lines, just include the whole file
    if len(relevant_lines) < 20:
        return {(0, len(lines)-1): code_str}
    
    # Group consecutive lines into sections
    sorted_lines = sorted(relevant_lines)
    sections = []
    current_section = [sorted_lines[0]]
    
    for line_num in sorted_lines[1:]:
        if line_num - current_section[-1] <= 5:  # Merge if within 5 lines
            current_section.append(line_num)
        else:
            sections.append(current_section)
            current_section = [line_num]
    sections.append(current_section)
    
    # Build section dict with actual code
    result = {}
    for section in sections:
        start = section[0]
        end = section[-1]
        section_code = '\n'.join(lines[start:end+1])
        result[(start, end)] = section_code
    
    return result


def ai_generate_patch(code_str, issues_log, script_filename):
    """
    Enhanced patch generation with smart context extraction.
    """
    # Try to find relevant sections first
    sections = find_relevant_code_sections(code_str, issues_log)
    
    # If code is small enough or we only have one section covering everything, send it all
    if len(code_str) < 20000 or (len(sections) == 1 and list(sections.keys())[0] == (0, len(code_str.splitlines())-1)):
        code_context = f"Full code:\n```python\n{code_str}\n```"
        context_info = f"Full file ({len(code_str)} chars, {len(code_str.splitlines())} lines)"
    else:
        # Build targeted context with line numbers
        context_parts = []
        for (start_line, end_line), section_code in sorted(sections.items()):
            context_parts.append(f"Lines {start_line+1}-{end_line+1}:\n```python\n{section_code}\n```")
        code_context = "\n\n".join(context_parts)
        context_info = f"Relevant sections ({len(sections)} sections, {sum(len(s) for s in sections.values())} chars)"
    
    prompt = f"""Fix these issues in the file '{script_filename}':

Issues:
{issues_log}

Code from {script_filename}:
{code_context}

Generate a complete unified diff patch for {script_filename}. Requirements:
1. Start with exactly: --- {script_filename}
2. Use EXACT line numbers from the actual file (line numbers are shown above)
3. Include ALL changes needed to fix ALL instances of the issues
4. Output the complete patch with all hunks for every location that needs fixing

Example format:
--- {script_filename}
+++ {script_filename}
@@ -10,7 +10,7 @@
 context line
-old line
+new line
 context line
@@ -25,3 +25,3 @@
-another old line
+another new line
"""

    payload = {
        "model": MODEL_NAME,
        "temperature": 0.7,
        "max_tokens": 8000,
        "messages":[
            {"role":"system","content":f"You generate complete git diff patches with accurate line numbers. Always use filename '{script_filename}' in patch headers."},
            {"role":"user","content":prompt}
        ]
    }
    
    # Debug: Print payload size
    print(f"DEBUG: Context: {context_info}")
    print(f"DEBUG: Prompt length: {len(prompt)} chars")

    initial_temp = payload['temperature']

    # Test: Try a simple request first to verify API is working (only on first call)
    if not hasattr(ai_generate_patch, '_tested'):
        test_payload = {
            "model": MODEL_NAME,
            "temperature": 0.7,
            "max_tokens": 100,
            "messages":[
                {"role":"user","content":"Say 'API test successful'"}
            ]
        }
        print("\n🧪 Testing API with simple request...")
        test_resp = persistent_request(test_payload)
        test_content = test_resp.get("choices",[{}])[0].get("message",{}).get("content","")
        print(f"TEST RESPONSE: {repr(test_content[:100])}")
        if not test_content:
            print("❌ CRITICAL: API returns empty even for simple requests!")
        else:
            print("✅ API is responding to simple requests")
        ai_generate_patch._tested = True
    
    for attempt in range(MAX_PATCH_ATTEMPTS):
        print(f"\n🔄 Patch generation attempt {attempt + 1}/{MAX_PATCH_ATTEMPTS} (temp={payload['temperature']:.2f})")
        
        resp = persistent_request(payload)
        
        # Minimal debug for subsequent attempts
        choice = resp.get("choices",[{}])[0]
        message = choice.get("message",{})
        content = message.get("content","")
        
        # Only show detailed debug on first attempt or failures
        if attempt == 0 or not content:
            print(f"DEBUG: Finish reason: {choice.get('finish_reason')}")
            if 'refusal' in message:
                print(f"🚫 REFUSAL: {message.get('refusal')}")
        
        if content is None:
            content = ""
        
        content = content.strip() if content else ""
        
        # Check if response is completely empty
        if not content:
            print(f"⚠️ AI returned EMPTY response (attempt {attempt + 1}/{MAX_PATCH_ATTEMPTS})")
            payload['temperature'] = min(1.0, initial_temp + (attempt + 1) * 0.15)
            time.sleep(2)
            continue
        
        print("\n--- AI PATCH OUTPUT (Start) ---")
        print(content[:500] + ("..." if len(content) > 500 else ""))  # Show first 500 chars
        print("--- AI PATCH OUTPUT (End) ---\n")

        # Try to extract patch if it starts correctly
        if content.startswith("--- "):
            return content
        
        # Try to find patch embedded in response
        if "--- " in content:
            patch_start = content.find("--- ")
            extracted = content[patch_start:]
            if extracted.startswith("--- "):
                print("✓ Extracted patch from wrapped output")
                return extracted
        
        # Check if it's wrapped in code blocks
        if "```" in content:
            # Try to extract from code block
            lines = content.split('\n')
            in_block = False
            patch_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or line.startswith("--- ") or line.startswith("+++ ") or line.startswith("@@"):
                    patch_lines.append(line)
            
            if patch_lines and patch_lines[0].startswith("--- "):
                extracted = '\n'.join(patch_lines)
                print("✓ Extracted patch from code block")
                return extracted
        
        print(f"⚠️ AI output invalid (missing proper patch format). Retrying...")
        payload['temperature'] = min(1.0, initial_temp + (attempt + 1) * 0.15)
        time.sleep(1)

    # Fallback: Generate corrected code and create patch manually
    print("⚠️ All patch generation attempts failed. Trying fallback: generate fixed code...")
    return ai_generate_code_and_diff(code_str, issues_log, script_filename)


def ai_generate_code_and_diff(code_str, issues_log, script_filename):
    """
    Fallback method: Ask AI to generate fixed code, then create diff ourselves.
    """
    # Give full context for accurate fixes
    prompt = f"""Fix these issues in the code:

Issues:
{issues_log}

Full original code:
```python
{code_str}
```

Provide the complete corrected Python code with all issues fixed. Output only the fixed code, no explanations."""

    payload = {
        "model": MODEL_NAME,
        "temperature": 0.7,
        "max_tokens": 10000,  # Increased to handle full code output
        "messages":[
            {"role":"system","content":"You fix Python code. Output only corrected code."},
            {"role":"user","content":prompt}
        ]
    }
    
    resp = persistent_request(payload)
    fixed_code = resp.get("choices",[{}])[0].get("message",{}).get("content","").strip()
    
    if not fixed_code:
        print("❌ Fallback code generation also failed.")
        return ""
    
    # Clean spam footer
    if "Want best roleplay experience?" in fixed_code:
        fixed_code = fixed_code.split("Want best roleplay experience?")[0].strip()
    
    # Clean up if wrapped in code blocks
    if fixed_code.startswith("```python"):
        fixed_code = fixed_code.split("```python", 1)[1]
    if fixed_code.startswith("```"):
        fixed_code = fixed_code.split("```", 1)[1]
    if fixed_code.endswith("```"):
        fixed_code = fixed_code.rsplit("```", 1)[0]
    fixed_code = fixed_code.strip()
    
    # Generate unified diff
    original_lines = code_str.splitlines(keepends=True)
    fixed_lines = fixed_code.splitlines(keepends=True)
    
    diff = unified_diff(original_lines, fixed_lines, 
                       fromfile=script_filename, 
                       tofile=script_filename, 
                       lineterm='')
    diff_str = '\n'.join(diff)
    
    if diff_str and diff_str.startswith("--- "):
        print("✅ Generated patch from AI-corrected code")
        return diff_str
    
    print("❌ Could not generate valid diff from corrected code")
    return ""


def ai_micro_patch(batch_turns, history):
    prompt = f"""You are an AI micro-patcher for character dialogue.
Batch output may have loops, truncation, or missing emotional cues:

{batch_turns}

Previous dialogue:
{history[-MAX_HISTORY:] if history else []}

Return corrected version preserving realism, breaking loops, restoring truncated ideas, and injecting context events."""

    payload = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": 2000,
        "messages":[
            {"role":"system","content":"You are an AI micro-patcher."},
            {"role":"user","content":prompt}
        ]
    }
    resp = persistent_request(payload)
    patched_batch = resp.get("choices",[{}])[0].get("message",{}).get("content","")
    return patched_batch.splitlines() if patched_batch else batch_turns

# -----------------------
# PRIORITIZED ISSUE ANALYSIS
# -----------------------
def analyze_looper_priority(code_path, char_files):
    issues = []

    for f in char_files:
        try:
            with open(f) as jf:
                validate_character_json(json.load(jf))
        except:
            issues.append({'issue': f"Malformed JSON: {f}", 'severity': 'high'})

    if not os.path.exists(code_path):
        issues.append({'issue': f"Critical: Script file missing or inaccessible: {code_path}", 'severity': 'high'})
        return issues

    with open(code_path) as cf:
        code = cf.read()
        if "Unknown Character" in code:
            issues.append({'issue': "Fallback 'Unknown Character' present", 'severity': 'high'})
        if ".lean back" in code or ".lean forward" in code:
            issues.append({'issue': "Possible repetitive physical loop", 'severity': 'high'})
        if "truncate" in code or "anti_echo" in code:
            issues.append({'issue': "Anti-echo may over-truncate", 'severity': 'medium'})
        if "print(" in code:
            issues.append({'issue': "Debug print statements present", 'severity': 'low'})
    return issues

# -----------------------
# DYNAMIC ISSUE TRACKER
# -----------------------
def load_issue_history():
    if os.path.exists(ISSUE_HISTORY_FILE):
        with open(ISSUE_HISTORY_FILE) as f:
            return json.load(f)
    return {}

def save_issue_history(history):
    with open(ISSUE_HISTORY_FILE,"w") as f:
        json.dump(history,f,indent=2)

def update_issue_history(detected_issues, current_run):
    history = load_issue_history()
    for issue in detected_issues:
        desc = issue['issue']
        sev = issue['severity']
        if desc not in history:
            history[desc] = {"severity": sev, "count": 1, "last_seen": current_run}
        else:
            history[desc]["count"] += 1
            history[desc]["last_seen"] = current_run
            # Upgrade severity if it recurs often
            if history[desc]["count"] >= MAX_OCCURRENCE_FOR_UPGRADE:
                if history[desc]["severity"] == "medium":
                    history[desc]["severity"] = "high"
                elif history[desc]["severity"] == "low":
                    history[desc]["severity"] = "medium"
    # Optional: downgrade old issues
    for desc,data in history.items():
        if current_run - data["last_seen"] >= MAX_INACTIVE_FOR_DOWNGRADE:
            if data["severity"] == "high":
                data["severity"] = "medium"
            elif data["severity"] == "medium":
                data["severity"] = "low"
    save_issue_history(history)
    return history

def apply_high_priority_patch_with_history(script_path, char_files, run_number):
    """Apply patches to modular structure or single file"""
    detected_issues = analyze_looper_priority(script_path, char_files)
    issue_history = update_issue_history(detected_issues, run_number)
    
    # Group issues by file
    issues_by_file = {}
    for issue in detected_issues:
        file_name = issue.get('file', 'unknown')
        severity = issue_history.get(issue['issue'], {}).get('severity', issue['severity'])
        
        if severity == 'high':
            if file_name not in issues_by_file:
                issues_by_file[file_name] = []
            issues_by_file[file_name].append(issue)
    
    if not issues_by_file:
        print(f"✓ No high-severity issues detected for Run {run_number}")
        return issue_history
    
    print(f"\n{'='*60}")
    print(f"🔧 Attempting high-severity patches for Run {run_number}")
    print(f"{'='*60}")
    
    # Process each file with issues
    for file_name, file_issues in issues_by_file.items():
        print(f"\n📄 Processing file: {file_name}")
        
        # Determine actual file path
        if os.path.isdir(script_path):
            actual_file_path = os.path.join(script_path, file_name)
        else:
            actual_file_path = script_path
        
        if not os.path.exists(actual_file_path):
            print(f"⚠️  File not found: {actual_file_path}, skipping...")
            continue
        
        # Read the file
        with open(actual_file_path) as f:
            code_str = f.read()
        
        # Build issues log for this file
        issues_log = "\n".join([f"- {i['issue']}" for i in file_issues])
        print(f"Issues to fix:\n{issues_log}\n")
        
        # Generate patch for this specific file
        patch_content = ai_generate_patch(code_str, issues_log, file_name)
        
        if patch_content and apply_patch_string(actual_file_path, patch_content):
            print(f"✅ Patch applied to {file_name}")
        else:
            print(f"⚠️ Patch failed for {file_name}")
    
    # Commit all changes together
    if any(issues_by_file.values()):
        commit_changes(f"Run {run_number}: High-severity AI patches applied to {len(issues_by_file)} files")
        print(f"\n✅ All patches committed for Run {run_number}")
            
    return issue_history

# -----------------------
# SIMULATION WITH WATCHDOG
# -----------------------
def run_simulation_with_watchdog(script_path, char_files, turns, simulation_log):
    history = []
    for t in range(turns):
        payload = {
            "model": MODEL_NAME,
            "temperature": TEMPERATURE,
            "max_tokens": 500,
            "messages":[
                {"role":"system","content":"Simulate character dialogue realistically."},
                {"role":"user","content":""}
            ]
        }
        
        char_data = []
        for f in char_files:
            with open(f) as jf:
                char_data.append(validate_character_json(json.load(jf)))
        payload["messages"][1]["content"] = json.dumps(char_data)

        resp = persistent_request(payload)
        text = resp.get("choices",[{}])[0].get("message",{}).get("content","")
        
        if "Context length exceeded" in text:
            raise RuntimeError(f"Context limit exceeded on turn {t+1}. History needs manual review/pruning.")
        
        if detect_loop(history,text) or len(text.split()) < 3:
            text = partial_loop_fix(history,text) + " " + inject_divergence()
            
        history.append(text)
        
        if len(history) % WATCHDOG_BATCH == 0:
            batch = history[-WATCHDOG_BATCH:]
            patched_batch = ai_micro_patch(batch, history)
            history[-WATCHDOG_BATCH:] = patched_batch
        
        with open(simulation_log,"w") as f:
            json.dump(history,f,indent=2)
            
    return history

# -----------------------
# AUTONOMOUS PATCH LOOP
# -----------------------
def autonomous_patch(script_path, char_files):
    run_number, start_turn_index = load_run_state()
    print(f"\n🚀 Starting autonomous patch system")
    print(f"📍 Resuming from Run {run_number}, turn index {start_turn_index}")
    
    simulation_log = "Nya_&_Azalea_current.json"
    
    for turn_index, turns in enumerate(SIMULATION_TURNS):
        if turn_index < start_turn_index:
            print(f"⏭️  Skipping completed {turns}-turn simulation.")
            continue
        
        while True:
            save_run_state(run_number, turn_index)
            
            try:
                issue_history = apply_high_priority_patch_with_history(script_path,char_files,run_number)
            except FileNotFoundError as e:
                print(f"🚨 CRITICAL FILE ERROR: {e}")
                return
            
            try:
                print(f"\n🧪 Starting validation Run {run_number} with {turns} turns...")
                history = run_simulation_with_watchdog(script_path,char_files,turns,simulation_log)
            
            except Exception as e:
                print(f"💥 CRITICAL ERROR during simulation: {e}")
                print("⏮️  Reverting last patch and retrying...")
                revert_last_commit()
                run_number += 1
                continue
                
            if len(history[-1].split()) >= 3:
                print(f"\n✅ Run {run_number} successful for {turns}-turn simulation!")
                print(f"   Moving to next turn size.\n")
                run_number += 1
                break 
            else:
                print(f"\n❌ Run {run_number} FAILED {turns}-turn simulation (output too short)")
                print("⏮️  Reverting patch and retrying...\n")
                revert_last_commit()
                run_number += 1
    
    print("\n🎉 All simulation phases complete!")
    print("🔄 Resetting run state for next execution.")
    save_run_state(1, 0)


# -----------------------
# MAIN
# -----------------------
if __name__=="__main__":
    # Can work with either single file or directory structure
    # For single file: autonomous_patch("looper.py", ["nya_elyria.json","empress_azalea.json"])
    # For modular: autonomous_patch("looper/", ["nya_elyria.json","empress_azalea.json"])
    
    import sys
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
    else:
        # Auto-detect structure
        if os.path.isdir("looper"):
            script_path = "looper/"
            print("📁 Detected modular structure: looper/")
        elif os.path.exists("looper.py"):
            script_path = "looper.py"
            print("📄 Detected single file: looper.py")
        else:
            print("❌ Could not find looper.py or looper/ directory")
            sys.exit(1)
    
    autonomous_patch(script_path, ["nya_elyria.json","empress_azalea.json"])
