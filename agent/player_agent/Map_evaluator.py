
from utils.ask_model import MultiTurnChatService
from utils.picture2chat import make_pic_content
import re

map_s_sys_prompt='''
You are an expert game navigation AI assistant that helps players understand:
1. How their gameplay view corresponds to the map directions;
2. Which directions are available for movement based on proximity to exploration boundaries

based on the game's exploration mechanics, where:
- Yellow areas on the map represent all explorable regions
- All yellow areas are interconnected through continuous paths
- The boundaries of yellow areas are absolute barriers that cannot be crossed
- The arrow on the map precisely indicates both player position and facing direction'''

map_all_sys_prompt='''
You are an expert game navigation assistant specialized in analyzing game world maps and player exploration patterns. 
Your role is to help track exploration progress, identify unexplored areas, and evaluate player navigation patterns.
'''

def generate_prompt(game_info):#, task_info3. If it's a **Gameplay Screen**, extract and summarize the key information based on the following elements: {game_info.get('Frame_attention')}
    prompt = f"""
You need to analyze two images: a current gameplay screenshot and its corresponding map view. 

Important game mechanics to remember:
- Only yellow areas on the map are explorable and all yellow regions are connected to each other
- The boundaries of yellow areas are absolute barriers that cannot be crossed
- The arrow on the map shows the exact direction the player is facing
- Other map decorations (trees, buildings, Buddha statue, etc.) are purely aesthetic

Analyze the gameplay screenshot and map view to provide two key pieces of information:
1. Arrow Position Analysis: 
    Describe the arrow's location on the map in ONE brief sentence.

2. Directional Correspondence Analysis:
   Based on the arrow direction on the map and the gameplay view, map out how the player's view corresponds to map directions

3. Movement Boundary Analysis:
   Describe available movements based on yellow area boundaries in ONE sentence.

Please output in the following format:

Arrow Position: 
The arrow is located near the western edge of a yellow clearing in the forest area.

Direction Mapping:
Current arrow direction on map: <compass direction>
Movement-to-Map conversion:Player's forward is <direction>, left is <direction>, back is <direction>, right is <direction>

Movement Space Analysis: 
Player can move forward and right, but cannot move left or back due to nearby boundaries.
"""
    return prompt

def generate_all_prompt(game_info):#, task_info3. If it's a **Gameplay Screen**, extract and summarize the key information based on the following elements: {game_info.get('Frame_attention')}
    prompt = f"""
You are analyzing a sequence of game world maps (up to 8 maps) that show the player's exploration history.

Key Map Reading Rules:
1. Only yellow areas on the map are explorable and all yellow regions are connected to each other
2. The boundaries of yellow areas are absolute barriers that cannot be crossed
3. The arrow on the map shows the exact direction the player is facing
4. Other map decorations (trees, buildings, Buddha statue, etc.) are purely aesthetic

Please provide a concise analysis with exactly four points, each summarized in a single sentence:

1. Terrain Layout: 
Describe the overall shape and connectivity of the yellow explorable areas.

2. Exploration History: 
From earliest map (Map 1) to latest map, summarize which yellow areas have been explored (e.g., "From map sequence analysis, player has explored the lower temple area (Map 1-3), crossed the central bridge pathway (Map 3-5)").

3. Unexplored Areas: 
List which directions (North/South/East/West) relative to player's current facing direction contain unexplored yellow areas.

4. Player Status: 
Based on analyzing all map history, evaluate if player is stuck (repeating same area in multiple maps) and if stuck, suggest which direction (North/South/East/West) to move to escape the stuck area.

Format your response exactly as follows:

Terrain Layout: "<single sentence description>"
Exploration History: "<single sentence summary>"
Unexplored Areas: "<single sentence listing directions>"
Player Status: "<single sentence assessment>
"""
    return prompt

# Terrain Layout: "The explorable yellow terrain forms a winding mountain path that stretches from southwest to northeast, featuring several wider plateau areas connected by narrow paths and decorated with temples and Buddha statues."
#
# Exploration History: "From map sequence analysis, player began at the lower shrine gate (Map 1-2), progressed through the central temple area with stone archway (Map 3-4), reached the twin Buddha statues plateau (Map 5-6), and is currently exploring the eastern mountain shrine area (Map 7-8)."
#
# Unexplored Areas: "Given player's current north-facing position, there are unexplored yellow areas to the North towards the mountain peak and to the West in a side path near the Buddha statues."
#
# Player Status: "Based on the last three maps (Map 6-8), player appears to be circling around the eastern shrine area without forward progress, should turn West at the next intersection to access the unexplored path near the Buddha statues."

def single_map_eval(game_frame_map,game_info):

    env_gather_agent=MultiTurnChatService(system_prompt=map_s_sys_prompt)

    prompt_gather=generate_prompt(game_info)

    content=make_pic_content(prompt_gather,game_frame_map)
    response = env_gather_agent.chat({"role": "user", "content": content})
    token_usage = env_gather_agent.get_token_usage()

    return  response,token_usage


def overall_eval(map_list,game_info):
    env_gather_agent = MultiTurnChatService(system_prompt=map_s_sys_prompt)

    prompt_gather = generate_all_prompt(game_info)

    content = make_pic_content(prompt_gather, map_list)
    response = env_gather_agent.chat({"role": "user", "content": content})
    token_usage = env_gather_agent.get_token_usage()

    return response, token_usage





