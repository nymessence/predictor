"""
Scenario progression system for advancing conversations through different stages
"""

import re
from typing import Dict, List, Optional
from api_client import make_api_call


class ScenarioProgressor:
    """Manages scenario progression through different stages"""

    def __init__(self, initial_scenario: str):
        self.initial_scenario = initial_scenario
        self.progression_completed = False
        self.progression_history = []
        self.last_progression_turn = 0  # Track when last progression occurred
        self.generated_stage_sequence = []  # Store procedurally generated stages
        self.current_stage_idx = 0  # Track current position in generated sequence
        self.current_stage = ""
        self.stage_descriptions = {}
        self.stage_transitions = {}
        self.stage_guidances = {}
        self.dynamic_scenario_context = initial_scenario  # Updated based on story flow

        # Generate the progression sequence based on the initial scenario
        self._generate_progression_sequence()

        # Set initial stage if sequence was generated
        if self.generated_stage_sequence:
            self.current_stage = self.generated_stage_sequence[0] if self.generated_stage_sequence else "initial"
        else:
            self.current_stage = "initial"  # fallback

    def _generate_progression_sequence(self):
        """Generate a custom progression sequence based on the initial scenario using AI"""
        try:
            prompt = f"""Based on this scenario, generate a logical progression sequence of 3-7 stages that the story might follow. Each stage should be a natural evolution from the previous one.

SCENARIO: {self.initial_scenario}

Generate a progression sequence as a numbered list. Each stage should:
1. Be a natural progression from the previous stage
2. Be specific to the scenario context
3. Be something characters can have meaningful conversations about
4. Have a clear connection to the story flow

Format:
1. [Stage name]: [Brief description]
2. [Stage name]: [Brief description]
...

Example for space travel scenario:
1. Pre-launch preparations: Characters prepare for space travel
2. Initial space travel: Early phase of journey through space
3. Mid-journey challenges: Obstacles or discoveries during travel
4. Approach destination: Getting close to target location
5. Atmospheric entry: Entering planet's atmosphere if applicable
6. Landing procedures: Safe landing sequence
7. Surface exploration: Exploring the new environment

Now generate for the given scenario:"""

            response = make_api_call(prompt, max_tokens=300, temperature=0.7)

            # Parse the response to extract stage names
            stages = []
            descriptions = {}
            transitions = {}
            guidances = {}

            # First, collect all valid stage lines
            valid_stage_lines = []
            for line in response.split('\n'):
                line = line.strip()
                # Look for lines that start with an incremental number followed by a period (e.g., "1.", "2.", "3.", etc.)
                # This ensures we get them in order and only process valid numbered entries
                for expected_num in range(1, 20):  # Reasonable range for stages
                    if line.startswith(f"{expected_num}. ") or line.startswith(f"{expected_num}."):
                        # Extract stage name and description
                        parts = line.split(':', 1)
                        if len(parts) >= 2:
                            stage_part = parts[0].split('.', 1)[-1].strip()  # Remove the number
                            desc_part = parts[1].strip()
                            valid_stage_lines.append((expected_num, stage_part.strip(), desc_part.strip()))
                        break  # Stop checking other numbers once we find a match

            # Process the valid lines in order to maintain sequence
            for num, stage_part, desc_part in sorted(valid_stage_lines, key=lambda x: x[0]):
                # Create a clean stage identifier
                stage_id = self._clean_stage_name(stage_part)
                if stage_id not in stages:  # Avoid duplicates
                    stages.append(stage_id)
                    descriptions[stage_id] = desc_part
                    guidances[stage_id] = f"During this stage: {desc_part}. Characters should react to this development."

            # If parsing failed, create a generic sequence based on keywords
            if not stages:
                scenario_lower = self.initial_scenario.lower()
                if any(kw in scenario_lower for kw in ['warp', 'space', 'spacecraft', 'travel']):
                    stages = ['departure', 'space_travel', 'mid_journey', 'approach', 'entry', 'landing', 'surface']
                    descriptions = {
                        'departure': 'Departing from the starting location',
                        'space_travel': 'Journeying through space',
                        'mid_journey': 'Mid-point developments during travel',
                        'approach': 'Approaching the destination',
                        'entry': 'Entering the destination environment',
                        'landing': 'Landing or docking procedures',
                        'surface': 'Surface activities at destination'
                    }
                    guidances = {k: f"Focus on this phase: {v}" for k, v in descriptions.items()}

            # Generate default transitions if not created
            if not transitions and stages:
                for i, stage in enumerate(stages[1:], 1):  # Skip first stage as it's initial
                    prev_stage = stages[i-1]
                    transitions[stage] = f"Moving from {prev_stage.replace('_', ' ')} to {stage.replace('_', ' ')}"

            self.generated_stage_sequence = stages
            self.stage_descriptions = descriptions
            self.stage_guidances = guidances

            # Generate transition messages between stages
            for i in range(1, len(stages)):
                from_stage = stages[i-1]
                to_stage = stages[i]
                transition_prompt = f"""Generate a narrative transition message for moving from stage '{from_stage}' to stage '{to_stage}' in this scenario: {self.initial_scenario}

The transition should be engaging and contextually appropriate. Keep it to 1-2 sentences.

Transition message:"""

                try:
                    transition_msg = make_api_call(transition_prompt, max_tokens=100, temperature=0.8)
                    # Clean up the response
                    transition_msg = transition_msg.strip().split('\n')[0] if transition_msg else f"Moving to next phase: {to_stage}"
                    self.stage_transitions[to_stage] = transition_msg
                except:
                    self.stage_transitions[to_stage] = f"Moving from {from_stage.replace('_', ' ')} to {to_stage.replace('_', ' ')}"

        except Exception as e:
            print(f"⚠️  AI progression generation failed, using fallback: {e}")
            # Fallback to basic progression
            self._generate_fallback_progression()

    def _clean_stage_name(self, stage_text: str) -> str:
        """Convert stage text to a clean identifier"""
        # Remove special characters, replace spaces with underscores
        cleaned = re.sub(r'[^\w\s-]', '', stage_text.lower())
        cleaned = re.sub(r'\s+', '_', cleaned.strip())
        return cleaned or "stage"

    def _generate_fallback_progression(self):
        """Generate a fallback progression sequence"""
        scenario_lower = self.initial_scenario.lower()

        if any(kw in scenario_lower for kw in ['warp', 'space', 'spacecraft', 'travel', 'departed earth']):
            stages = ['warp_travel', 'mid_journey', 'approach', 'descent', 'landing', 'surface_arrival']
        elif any(kw in scenario_lower for kw in ['battle', 'combat', 'war', 'fight']):
            stages = ['preparation', 'encounter', 'battle', 'lull', 'aftermath', 'recovery']
        elif any(kw in scenario_lower for kw in ['negotiation', 'talk', 'meeting', 'discussion']):
            stages = ['introductions', 'topic_introduction', 'tension', 'compromise', 'agreement', 'follow_up']
        else:
            stages = ['beginning', 'development', 'conflict', 'resolution', 'conclusion']

        self.generated_stage_sequence = stages
        self.stage_descriptions = {stage: f"Stage: {stage.replace('_', ' ')}" for stage in stages}
        self.stage_guidances = {stage: f"Focus on this stage: {stage.replace('_', ' ')}" for stage in stages}
        # Generate basic transitions
        for i in range(1, len(stages)):
            self.stage_transitions[stages[i]] = f"Transitioning from {stages[i-1]} to {stages[i]}"
    
    def get_current_stage_description(self) -> str:
        """Get the description for the current stage"""
        return self.stage_descriptions.get(self.current_stage, f"In stage: {self.current_stage}")
    
    def should_advance_stage(self, history: List[Dict], turn: int) -> bool:
        """Determine if we should advance to the next stage"""
        if self.progression_completed:
            return False

        # Don't advance if we already advanced recently (avoid rapid advancement)
        if turn - self.last_progression_turn < 20:  # Wait at least 20 turns between progressions
            return False

        # Check if we're already at the last stage
        next_idx = self.current_stage_idx + 1
        if next_idx >= len(self.generated_stage_sequence):
            # We're at the final stage, mark progression as completed and return False
            self.progression_completed = True
            return False

        # Advance every 40 turns, but don't advance too frequently
        if turn % 40 == 0 and turn > 0:
            return True

        # Check if we've been in the same stage for too long
        # Use the length of history directly (should be efficient)
        if len(history) > 50:
            return True

        return False
    
    def advance_stage(self, history: List[Dict], current_turn: int = 0) -> str:
        """Advance to the next stage in the progression"""
        next_idx = self.current_stage_idx + 1

        if next_idx < len(self.generated_stage_sequence):
            # Move to the next stage
            self.progression_history.append(self.current_stage)
            self.current_stage_idx = next_idx
            self.current_stage = self.generated_stage_sequence[next_idx]
            self.last_progression_turn = current_turn  # Track when progression happened

            # Get the transition message for this specific stage
            transition_msg = self.stage_transitions.get(self.current_stage, f"Advancing to stage: {self.current_stage}")
            return transition_msg
        else:
            # No more stages to advance to, mark as completed
            self.progression_completed = True
            return ""
    
    
    def get_scenario_context_for_stage(self) -> str:
        """Get the scenario context for the current stage"""
        base_context = self.dynamic_scenario_context  # Use dynamic context that can be updated by story flow

        stage_description = self.stage_descriptions.get(self.current_stage, f"Currently in the {self.current_stage} phase")

        if self.current_stage in self.stage_descriptions:
            return f"{base_context} {stage_description}"
        else:
            return base_context
    
    def get_stage_guidance(self) -> str:
        """Get guidance for the current stage"""
        return self.stage_guidances.get(self.current_stage, "Continue the conversation naturally.")

    def update_scenario_context_from_story_flow(self, history: List[Dict], turn: int) -> str:
        """
        Analyze the story flow and update the scenario context dynamically
        This allows the scenario to evolve based on character interactions
        """
        try:
            if not history:
                return self.dynamic_scenario_context

            # Get recent history (last few exchanges) to analyze story flow
            recent_history = history[-6:] if len(history) >= 6 else history  # Get last ~3 exchanges

            # Create a summary of recent conversation
            conversation_summary = []
            for entry in recent_history:
                if isinstance(entry, dict) and 'name' in entry and 'content' in entry:
                    conversation_summary.append(f"{entry['name']}: {entry['content'][:200]}")  # Limit length

            summary_text = "\n".join(conversation_summary)

            # Create prompt to analyze story evolution
            prompt = f"""Analyze this recent conversation and update the scenario context to reflect how the story has evolved.

ORIGINAL SCENARIO: {self.initial_scenario}

RECENT CONVERSATION:
{summary_text}

CURRENT STAGE: {self.current_stage}
CURRENT STAGE DESCRIPTION: {self.stage_descriptions.get(self.current_stage, 'No description')}

Based on the conversation flow, provide an updated scenario context that reflects new developments, character discoveries, plot twists, or evolving situation. Keep it concise but meaningful. The updated context should maintain the core scenario while incorporating story developments.

Updated scenario context:"""

            updated_context = make_api_call(prompt, max_tokens=200, temperature=0.7)

            if updated_context and updated_context.strip():
                self.dynamic_scenario_context = updated_context.strip()
                return self.dynamic_scenario_context
        except Exception as e:
            print(f"⚠️  Failed to update scenario context from story flow: {e}")

        # Return current context if update failed
        return self.dynamic_scenario_context


def check_scenario_progression(scenario_progressor: ScenarioProgressor, history: List[Dict], turn: int) -> Optional[str]:
    """Check if scenario progression is needed and advance if appropriate"""
    # First, update the scenario context based on story flow
    scenario_progressor.update_scenario_context_from_story_flow(history, turn)

    if scenario_progressor.should_advance_stage(history, turn):
        transition_message = scenario_progressor.advance_stage(history, turn)
        return transition_message
    return None