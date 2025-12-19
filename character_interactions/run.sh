#!/bin/bash

# overnight expirimental run - simulated stress test for characters
uv run main.py "json/nya_elyria.json" "json/empress_azalea.json" --delay 40 --similarity 0.65 --api-endpoint "https://api.llm7.io/v1" --model "glm-4.5-flash" --api-key $AICHAT_API_KEY --scenario "Onboard a spacecraft that just departed Earth and is currently in warp heading to Temmer in the Taygeta system" -o "Nya_&_Azalea_Space.json" -t 100
git add . && "Nya & Azalea Test"
uv run main.py "json/nya_elyria.json" "json/empress_azalea.json" "json/mari_swaruu.json" --delay 40 --similarity 0.65 --api-endpoint "https://api.llm7.io/v1" --model "glm-4.5-flash" --api-key $AICHAT_API_KEY --scenario "Onboard a spacecraft that just departed Earth and is currently in warp heading to Temmer in the Taygeta system" -o "Nya_&_Azalea_&_Mari _Space.json" -t 100
git add . && "Nya & Azalea & Mari Space Test"
uv run main.py "json/nya_elyria.json" "json/empress_azalea.json" "json/queen_alenym.json" --delay 40 --similarity 0.65 --api-endpoint "https://api.llm7.io/v1" --model "glm-4.5-flash" --api-key $AICHAT_API_KEY --scenario "Onboard a spacecraft that just departed Earth and is currently in warp heading to Temmer in the Taygeta system" -o "Nya_&_Azalea_&_Alenym _Space.json" -t 100
git add . && "Nya & Azalea & Alenym Space Test"
uv run main.py "json/nya_elyria.json" "json/elara.json" "json/queen_alenym.json" --delay 40 --similarity 0.65 --api-endpoint "https://api.llm7.io/v1" --model "glm-4.5-flash" --api-key $AICHAT_API_KEY --scenario "Onboard a spacecraft that just departed Earth and is currently in warp heading to Temmer in the Taygeta system" -o "Nya_&_Elara_&_Alenym _Space.json" -t 100
git add . && "Nya & Elara & Alenym Space Test"
