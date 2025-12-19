"""
Advanced repetition detection and pattern blocking
"""

import re
from typing import Dict, List, Any
from collections import Counter
from utils import calculate_similarity
from config import MAX_ACTION_DESCRIPTIONS


def detect_repetition_patterns(history: List[Dict], similarity_threshold: float) -> Dict[str, Any]:
    """Dramatically enhanced repetition detection with aggressive pattern blocking"""
    if len(history) < 4:
        return {'repetition_score': 0.0, 'issues': [], 'blocked_patterns': []}
    
    issues = []
    blocked_patterns = []
    repetition_score = 0.0
    
    # Get last 8 messages for comprehensive analysis
    recent_messages = history[-8:] if len(history) >= 8 else history
    recent_contents = [str(msg.get('content', '')).lower() for msg in recent_messages if isinstance(msg, dict)]
    
    if len(recent_contents) < 3:
        return {'repetition_score': 0.0, 'issues': [], 'blocked_patterns': []}
    
    # 1. AGGRESSIVE opening phrase detection
    opening_phrases = []
    for i, content in enumerate(recent_contents):
        words = content.split()
        if len(words) >= 7:
            opening = " ".join(words[:7])
        else:
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            opening = sentences[0] if sentences else content[:30]
        opening_phrases.append((i, opening.strip()))
    
    # Block repetitive openings
    opening_counter = Counter([opening for _, opening in opening_phrases])
    for opening, count in opening_counter.items():
        if count >= 2 and len(recent_messages) >= 5:
            issues.append(f"CRITICAL: repetitive opening pattern: '{opening[:25]}...'")
            blocked_patterns.append(f"*{opening}*")
            repetition_score += min(count * 0.25, 0.6)
    
    # 2. BLOCK copied content between consecutive messages
    for i in range(1, len(recent_contents)):
        similarity = calculate_similarity(recent_contents[i-1], recent_contents[i])
        if similarity > 0.7:
            issues.append(f"CRITICAL: content copying between messages {i} and {i+1}: {similarity:.2f}")
            common_words = set(recent_contents[i-1].split()).intersection(set(recent_contents[i].split()))
            if common_words:
                blocked_patterns.append(" ".join(list(common_words)[:10]))
            repetition_score += min(similarity * 0.9, 0.8)
    
    # 3. DETECT character voice contamination
    if len(set([h['name'] for h in recent_messages])) > 1:
        char_contents = {}
        for msg in recent_messages:
            if isinstance(msg, dict) and msg.get('name') and msg.get('content'):
                char_name = msg['name']
                char_contents.setdefault(char_name, []).append(str(msg['content']).lower())
        
        for char1, contents1 in char_contents.items():
            for char2, contents2 in char_contents.items():
                if char1 == char2:
                    continue
                for c1 in contents1:
                    for c2 in contents2:
                        similarity = calculate_similarity(c1, c2)
                        if similarity > 0.65:
                            issues.append(f"CRITICAL: characters {char1} and {char2} copying speech patterns")
                            blocked_patterns.append(f"*{c1[:30]}*")
                            repetition_score += 0.3
    
    # 4. ACTION DESCRIPTION OVERUSE DETECTION
    action_count = 0
    total_messages = 0
    action_pattern = re.compile(r'\*[^*]+\*')
    
    for content in recent_contents:
        if action_pattern.search(content):
            action_count += 1
        total_messages += 1
    
    if total_messages > 0:
        action_ratio = action_count / total_messages
        if action_ratio > MAX_ACTION_DESCRIPTIONS / 2:
            issues.append(f"CRITICAL: overused action descriptions ({action_ratio:.2f})")
            blocked_patterns.append("*action description*")
            repetition_score += min(action_ratio * 0.7, 0.5)
    
    # 5. ENVIRONMENTAL TRIGGER IGNORE DETECTION
    ignored_triggers = 0
    total_triggers = 0
    
    trigger_words = [
        "wind", "lights flicker", "chime", "enters", "shatters", 
        "temperature drops", "aroma", "message arrives", "ground trembles",
        "creature's call", "music changes", "clock strikes", "explosion",
        "alarm", "stranger", "floor gives way", "holographic", "time slows",
        "walls shift", "portal", "technology powers down", "mysterious figure"
    ]
    
    response_words = [
        "responds to", "reacts to", "notices", "acknowledges", 
        "addresses", "deals with", "handles", "confronts", "reacting to",
        "looks at", "turns toward", "focuses on", "acknowledges the", "observes"
    ]
    
    for i, content in enumerate(recent_contents):
        if i == 0:
            continue
        
        prev_content = recent_contents[i-1]
        if any(trigger in prev_content for trigger in trigger_words):
            total_triggers += 1
            if not any(response in content for response in response_words):
                ignored_triggers += 1
    
    if total_triggers > 0 and ignored_triggers / total_triggers > 0.4:
        issues.append(f"CRITICAL: ignoring environmental triggers ({ignored_triggers}/{total_triggers})")
        repetition_score += 0.4
    
    # 6. INCOMPLETE SENTENCE DETECTION
    incomplete_count = 0
    for content in recent_contents:
        if content.strip().endswith((',', '-', ':', ';', '...')) or len(content.split()) < 3:
            incomplete_count += 1
    
    if incomplete_count / len(recent_contents) > 0.3:
        issues.append(f"CRITICAL: incomplete sentences ({incomplete_count}/{len(recent_contents)})")
        blocked_patterns.append("incomplete sentence")
        repetition_score += 0.3
    
    # CAP and normalize score
    repetition_score = min(repetition_score, 1.0)
    
    return {
        'repetition_score': repetition_score,
        'issues': issues[:5],
        'blocked_patterns': list(set(blocked_patterns))
    }


def generate_anti_repetition_guidance(repetition_data: Dict, character_name: str, 
                                     other_character_name: str) -> str:
    """Generate extremely aggressive anti-repetition guidance with content blocking"""
    if repetition_data.get('repetition_score', 0) < 0.3:
        return ""
    
    guidance = "\n\n### 🔥 EMERGENCY ANTI-REPETITION PROTOCOLS ACTIVATED ###\n"
    
    # BLOCK specific patterns
    blocked_patterns = repetition_data.get('blocked_patterns', [])
    if blocked_patterns:
        guidance += "🚨 ABSOLUTE CONTENT BLOCKING:\n"
        for pattern in blocked_patterns[:3]:
            guidance += f"- NEVER use: \"{pattern}\"\n"
        guidance += "- Replace blocked patterns with completely original content\n\n"
    
    # Address critical issues
    for issue in repetition_data.get('issues', []):
        if 'opening pattern' in issue.lower():
            guidance += "🔄 OPENING PHRASE PROHIBITION:\n"
            guidance += f"- ABSOLUTELY NO repetitive openings like 'I lean back', 'I chuckle softly'\n"
            guidance += "- Start with: [Question], [Unexpected statement], [Environmental reaction], or [Memory flashback]\n\n"
        
        elif 'content copying' in issue.lower():
            guidance += "❌ CONTENT COPY PROHIBITION:\n"
            guidance += f"- NEVER copy or closely paraphrase {other_character_name}'s words\n"
            guidance += "- If you find yourself repeating, STOP and introduce a completely new topic\n\n"
        
        elif 'speech patterns' in issue.lower():
            guidance += "🎭 CHARACTER VOICE CONTAMINATION:\n"
            guidance += f"- You are {character_name}, NOT {other_character_name}\n"
            guidance += "- Your speech patterns must be distinct and consistent\n\n"
        
        elif 'action descriptions' in issue.lower():
            guidance += "⚔️ ACTION DESCRIPTION LIMIT:\n"
            guidance += f"- MAXIMUM {MAX_ACTION_DESCRIPTIONS} action/emotional description per response\n\n"
        
        elif 'environmental triggers' in issue.lower():
            guidance += "🌪️ ENVIRONMENTAL TRIGGER RESPONSE MANDATORY:\n"
            guidance += f"- You MUST react to environmental changes within 2 responses\n\n"
        
        elif 'incomplete sentences' in issue.lower():
            guidance += "✍️ COMPLETE SENTENCE REQUIREMENT:\n"
            guidance += "- EVERY response must contain complete, grammatically correct sentences\n\n"
    
    guidance += "### 🚨 EMERGENCY RESPONSE PROTOCOLS ###\n"
    guidance += "If repetition detected:\n"
    guidance += "- IMMEDIATELY introduce a memory from your past\n"
    guidance += "- Ask a direct question about the other character's motivations\n"
    guidance += "- Reference a specific detail from earlier in the conversation\n"
    guidance += "- Change the subject to something completely unrelated\n\n"
    
    return guidance
