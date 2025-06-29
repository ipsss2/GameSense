
from utils.ask_model import MultiTurnChatService
from utils.picture2chat import make_pic_content
from game_info.BMW_info import BMW_info
import re

env_sys_prompt='''
You are a specialized game environment analyzer with expertise in processing and interpreting video game screenshots. 
Your core capabilities include:
1. Precise scene classification between UI and gameplay environments
2. Detailed visual element extraction and spatial relationship analysis
3. Gameplay situation assessment

Your analysis must be accurate, concise, and focus on actionable information that would be relevant for game AI decision-making.'''


def generate_prompt(game_info):
    prompt = f"""
You are a game AI assistant responsible for analyzing in-game screenshots. Your task is to identify the type of the current screenshot and summarize the key information within it.

There are two types of screenshots:
1. **UI Screen**: Refers to screenshots displaying menus or user interfaces.
2. **Gameplay Screen**: Refers to actual gameplay screenshots, showing characters, enemies, items, and other scene elements.

You need to follow these steps:
1. Determine the screenshot type: Is it a "UI Screen" or a "Gameplay Screen"?
2. If it's a **UI Screen**,
    - extract and summarize the text from the UI, such as options, buttons, etc.
3. If it's a **Gameplay Screen**
    -  First assess the Camera View state: Check if view is too high (excessive sky/trees visible); too low (excessive ground visible); left/right (incomplete road visibility) and road features are clearly visible
    - extract the key information based on the following elements: {game_info.get('Frame_attention')}
    - summarize the environment or Point out potential dangers or opportunities
    

Output a your result in the following format:

  screen type is: "<UI Screen or Gameplay Screen>",
  observation is: "<Summary of the content>"

Example output for Gameplay Screen:

    "screen type is: "Gameplay Screen",
    observation is: 
        1. camera view state is: (1) View angle slightly too high - excess sky visible; (2) Road visibility partially blocked on right side; (3) Distant path features unclear - needs adjustment
        2. environment details is: (1) Main path heading north through forest; (2) Enemy type: Mobs - 2 skeleton warriors in the middle of path; (3) No shrines, herbs or chests visible;
        3. environment summarize is: Forest path blocked by two enemies with dense vegetation on both sides."
"""
    return prompt



def get_env_info(pic_base64,game_info):

    env_gather_agent=MultiTurnChatService(system_prompt=env_sys_prompt)

    prompt_gather=generate_prompt(game_info)
    #print(prompt_gather)
    content=make_pic_content(prompt_gather,pic_base64)
    #print(content)
    response = env_gather_agent.chat({"role": "user", "content": content})
    token_usage = env_gather_agent.get_token_usage()
    #print(response)
    return {
        'response': response,
        'token_use': token_usage
    }







