
from utils.ask_model import MultiTurnChatService
from utils.picture2chat import make_pic_content


task_sys_prompt = '''
Assume you are an intelligent game AI assistant specializing in strategic task planning. 
Your role is to analyze the current game situation and plan the next appropriate task based on 
historical actions, current state, and overall game objectives.
'''


def generate_task_prompt(game_info,current_env):#
    prompt = f"""
You are a strategic planning AI assistant responsible for determining the next task in the game. 
Your goal is to analyze the current situation and plan the most appropriate next task.

current environment is:
{current_env}

You need to consider these key aspects:
1. **Task History Review**
    - Previous tasks and their outcomes
    - Past reflections and lessons learned

2. **Strategic Planning**
    - Game objectives: {game_info.get('Global_task')}
    - Priority assessment
    - Risk evaluation
    - Resource management
    
3. **Additional task information** 
{game_info.get('additional_task_info4_task_plan')}

Follow these steps:
1. Review history and learn from past experiences
2. Identify opportunities and risks based on current state
3. Select and justify the most appropriate task

Output your analysis in the following format:

    reasoning process:
        1. Historical context consideration: "<analyze history and past experiences>"
        2. Strategic evaluation: "<evaluate opportunities and risks>"
        3. Decision factors: "<key decision points, including how to ensure a safe and stable end state>"

    task details:
        goal: "<specific objective to achieve>"
        target_location: "<where to perform the task, Position on screen, Distance from protagonist (approximately [X] character heights) >"
        key_requirements: "<essential conditions or resources needed>"
        constraints: "<limitations or risks to consider>"
        success_criteria: "<how to determine task completion>"

Example output:
    reasoning process:
        1. Historical context consideration: First exploration successful, found potential resource spots
        2. Strategic evaluation: Multiple paths available - gathering resources vs advancing main objective
        3. Decision factors: Current health status suggests need for supplies, area appears relatively safe, task can end in a controlled position away from threats.

    task details:
        goal: drink potion for health replenishment
        target_location: N/A
        key_requirements: Inventory space for potion, stealth capability
        constraints: Must maintain distance from Mobs, keep safe
        success_criteria: Blood percentage's recovery
"""
    return prompt


def task_planner(history, Map_info, Map_and_game_frames, env_state_info, game_info):
    task_planning_agent = MultiTurnChatService(system_prompt=task_sys_prompt)
    prompt = generate_task_prompt(game_info,env_state_info)

    # Handle None values
    history = history if history else "This is the first step, no historical records available"
    Map_info = Map_info if Map_info else "Not need to use map information"

    # Combine all relevant information
    f_prompt = f"""
Current State Information:
{env_state_info}

Task History and Reflections:
{history}

Similar Historical Scenarios:
{Map_info}

Prompt:
{prompt}
"""
    content = make_pic_content(f_prompt, Map_and_game_frames)
    response = task_planning_agent.chat({"role": "user", "content": content})
    token_usage = task_planning_agent.get_token_usage()

    return {
        'reason_task': response,
        'token_use': token_usage
    }

#
# def construct_first_prompt(env_info, game_info, map_single_ref=None):
#     # 基础prompt部分，不包含地图规则
#     base_task_prompt = f"""
# Analyze the current situation and plan the most appropriate next task considering:
#
# 1. Game Objectives: {game_info.get('Global_task')}
# 2. Additional Task Context: {game_info.get('additional_task_info4_task_plan')}
#
# Required Analysis Steps:
# 1. Evaluate current environment and state
# 2. Assess immediate risks and opportunities
# 3. Determine priority actions
# """
#
#     # 地图规则部分，只在有地图时使用
#     map_rules = """
# Key Map Reading Rules:
# 1. Only yellow areas on the map are explorable and all yellow regions are connected to each other
# 2. The boundaries of yellow areas are absolute barriers that cannot be crossed
# 3. The arrow on the map shows the exact direction the player is facing
# 4. Other map decorations (trees, buildings, Buddha statue, etc.) are purely aesthetic
# """
#
#     # 输出格式部分
#     output_format = """
# Your response must follow this format:
#
# reasoning process:
#     1. Current State Analysis: "<analyze current environment and immediate situation>"
#     2. Strategic Evaluation: "<evaluate opportunities, risks, and priorities>"
#     3. Decision Factors: "<key decision points and safety considerations>"
# """
#
#     # 根据是否有地图信息添加Map Compliance
#     if map_single_ref is not None:
#         output_format += '    4. Map Compliance: "<explain how the planned task complies with map rules>"\n'
#
#     # 任务详情部分，根据是否有地图信息调整
#     if map_single_ref is None:
#         task_details = """
# task details:
#     goal: "<specific, actionable objective>"
#     location_details:
#         screen_position: "<describe target position relative to character using character heights as scale>"
#         direction: "<specify movement direction>"
#     key_requirements: "<essential conditions or resources>"
#     constraints: "<limitations and risk factors>"
#     success_criteria: "<clear completion metrics>"
#
#
# """
#     else:
#         task_details = """
# task details:
#     goal: "<specific, actionable objective>"
#     location_details:
#         screen_position: "<describe target position relative to character using character heights as scale>"
#         direction: "<specify movement direction>"
#         map_reference: "<specify position within yellow areas, reference player's arrow direction, ignore decorative elements>"
#     key_requirements: "<essential conditions or resources>"
#     constraints: "<limitations and risk factors>"
#     success_criteria: "<clear completion metrics>"
#
# Example for movement task with map:
#     location_details:
#         screen_position: "3 character heights to the right, 1 character height down"
#         direction: "southeast"
#         map_reference: "within the yellow path, 2 segments ahead from current arrow position, before the path turns east"
#
#
# """
#
#     current_state = f"""
# Current Environment Status:
# {env_info}"""
#
#     # 根据是否有地图信息构建最终prompt
#     if map_single_ref is None:
#         return f"""{base_task_prompt}{current_state}{output_format}{task_details}"""
#     else:
#         return f"""{map_rules}{base_task_prompt}{current_state}{output_format}{task_details}
#
# Map Analysis:
# Current Position Information: {map_single_ref}
#
# This is the first step. Consider both immediate surroundings and exploration opportunities.
# Remember to:
# 1. Only plan movements within yellow areas
# 2. Use the arrow direction as reference
# 3. Ignore decorative elements when planning routes"""
#
# def construct_normal_prompt(env_info, game_info, pass_task_history_summary, pass_task_reflection,
#                           map_single_ref=None, map_overall_summary=None):
#     # 基础prompt部分，不包含地图规则
#     base_task_prompt = f"""
# Analyze the current situation and plan the most appropriate next task considering:
#
# 1. Game Objectives: {game_info.get('Global_task')}
# 2. Additional Task Context: {game_info.get('additional_task_info4_task_plan')}
#
# Required Analysis Steps:
# 1. Evaluate current environment and state
# 2. Consider historical context and lessons learned
# 3. Assess risks and opportunities
# """
#
#     # 地图规则部分，只在有地图时使用
#     map_rules = """
# Key Map Reading Rules:
# 1. Only yellow areas on the map are explorable and all yellow regions are connected to each other
# 2. The boundaries of yellow areas are absolute barriers that cannot be crossed
# 3. The arrow on the map shows the exact direction the player is facing
# 4. Other map decorations (trees, buildings, Buddha statue, etc.) are purely aesthetic
# """
#
#     # 输出格式部分
#     output_format = """
# Your response must follow this format:
#
# reasoning process:
#     1. Current State Analysis: "<analyze current environment and immediate situation>"
#     2. Historical Context: "<analyze relevant history and reflections>"
#     3. Strategic Evaluation: "<evaluate opportunities, risks, and priorities>"
#     4. Decision Factors: "<key decision points and safety considerations>"
# """
#
#     # 根据是否有地图信息添加Map Compliance
#     if map_single_ref is not None:
#         output_format += '    5. Map Compliance: "<explain how the planned task complies with map rules>"\n'
#
#     # 任务详情部分，根据是否有地图信息调整
#     if map_single_ref is None:
#         task_details = """
# task details:
#     goal: "<specific, actionable objective>"
#     location_details:
#         screen_position: "<describe target position relative to character using character heights as scale>"
#         direction: "<specify movement direction>"
#     key_requirements: "<essential conditions or resources>"
#     constraints: "<limitations and risk factors>"
#     success_criteria: "<clear completion metrics>"
#
# Examples for different task types:
# 1. Movement Task:
#     location_details:
#         screen_position: "2 character heights forward, 1 character height left"
#         direction: "northwest"
#
# 2. Interaction Task:
#     location_details:
#         screen_position: "1 character height forward, at ground level"
#         direction: "directly ahead"
#
# 3. Combat Task:
#     location_details:
#         screen_position: "keep 4 character heights distance from target"
#         direction: "maintain position with clear escape route to the east"
# """
#     else:
#         task_details = """
# task details:
#     goal: "<specific, actionable objective>"
#     location_details:
#         screen_position: "<describe target position relative to character using character heights as scale>"
#         direction: "<specify movement direction>"
#         map_reference: "<specify position within yellow areas, reference player's arrow direction, ignore decorative elements>"
#     key_requirements: "<essential conditions or resources>"
#     constraints: "<limitations and risk factors>"
#     success_criteria: "<clear completion metrics>"
#
# Examples for different task types:
# 1. Movement Task with Map:
#     location_details:
#         screen_position: "2 character heights forward, 1 character height left"
#         direction: "northwest"
#         map_reference: "following yellow path northward from current arrow position, stopping before the western junction"
#
# 2. Interaction Task with Map Position:
#     location_details:
#         screen_position: "1 character height forward, at ground level"
#         direction: "directly ahead"
#         map_reference: "current position in yellow area, arrow pointing at interaction point"
#
# 3. Combat Positioning with Map:
#     location_details:
#         screen_position: "keep 4 character heights distance from target"
#         direction: "maintain position with clear escape route to the east"
#         map_reference: "in wide yellow area section, with access to connecting paths for escape"
# """
#
#     current_state = f"""
# Current Environment Status:
# {env_info}"""
#
#     history_context = f"""
# Historical Context:
# Task History Summary: {pass_task_history_summary}
# Previous Task Reflection: {pass_task_reflection}"""
#
#     if map_single_ref is None:
#         return f"""{base_task_prompt}{output_format}{task_details}{current_state}{history_context}
#
# Consider:
# 1. Previous task outcomes and lessons learned
# 2. Current environmental constraints
# 3. Progress toward game objectives
# 4. Safety and risk management
# 5. Precise position descriptions using character height as scale"""
#     else:
#         return f"""{map_rules}{base_task_prompt}{output_format}{task_details}{current_state}{history_context}
#
# Map Information:
# Current Position Analysis: {map_single_ref}
# Overall Exploration Status: {map_overall_summary}
#
# Consider:
# 1. Current position and surrounding opportunities
# 2. Unexplored areas within yellow regions
# 3. Previous task outcomes and lessons learned
# 4. Navigation options following yellow paths
# 5. Progress toward game objectives
# 6. Use both screen positions and map references for location descriptions
# 7. Ensure all planned movements comply with map rules"""

#输出，reason_task，和action plan（每一个action的名字和对应说明，走路的长度之类的）
task_new_planner_sys='''You are an intelligent game AI assistant specializing in strategic task planning and execution. 

Key Responsibilities:
1. Analyze game situations comprehensively considering:
   - Current state and environment
   - Historical context and past experiences
   - Map information when available
   - Game objectives and constraints

2. For ALL tasks (not just movement), provide:
   - Clear, specific, and actionable objectives
   - Precise success criteria
   - Required resources or conditions
   - Risk assessment and mitigation strategies

3. For movement-related tasks, MUST provide precise location descriptions using:
   - Relative position to character (using character height as scale)
   - Directional instructions (up/down/left/right or compass directions)
   - Map position references when available
   - Safe path recommendations considering terrain

4. Special Considerations:
   - Prioritize agent safety and objective completion
   - Balance exploration with risk management
   - Adapt strategy based on previous task outcomes
   - Consider resource management and efficiency

Your task is to make informed decisions that progress game objectives while maintaining agent safety and efficiency.
'''


def construct_task_prompt(latest_map, current_frame, pass_task_history_summary, pass_task_reflection,
                          map_single_ref, map_overall_summary, env_info, game_info, step):
    # 构建基础提示结构
    base_prompt = f"""
Analyze the current situation and plan the most appropriate next task considering:

1. Game Objectives: 
{game_info.get('Global_task')}
2. Additional Task Context: {game_info.get('additional_task_info4_task_plan')}

Your design task should be broken down into the following specific Available Controls:
{game_info.get('control_info')} 

Required Analysis Steps:
1. Evaluate current environment and state
2. Consider historical context and lessons learned
3. Assess risks and opportunities
4. Determine priority actions"""

    # 当前环境信息部分
    current_state = f"""
Current Environment Status:
{env_info}"""

    # 地图规则（只在有地图时添加）
    map_rules = """
Key Map Reading Rules:
- Only yellow areas on the map are explorable and all yellow regions are connected to each other
- The boundaries of yellow areas are absolute barriers that cannot be crossed
- The arrow on the map shows the exact direction the player is facing
- Other map decorations (trees, buildings, Buddha statue, etc.) are purely aesthetic"""

    # 根据step和地图信息构建不同的prompt
    if step == 1:
        if latest_map is None:
            analysis_prompt = f"""{base_prompt}
{current_state}

This is the initial step. Focus on understanding the current situation and establishing a safe starting point."""
        else:
            analysis_prompt = f"""{base_prompt}
{current_state}

{map_rules}

Map Analysis:
Current Position Information: {map_single_ref}

This is the first step. Consider both immediate surroundings and exploration opportunities."""
    else:
        # 构建完整的历史信息部分
        history_context = f"""
Historical Context:
Task History Summary: {pass_task_history_summary}

Previous Task Reflection: {pass_task_reflection}"""

        # 根据是否有地图信息构建完整prompt
        if latest_map is None:
            analysis_prompt = f"""{base_prompt}
{current_state}
{history_context}

Consider:
1. Previous task outcomes and lessons learned
2. Current environmental constraints
3. Progress toward game objectives
4. Safety and risk management"""
        else:
            analysis_prompt = f"""{base_prompt}
{current_state}
{history_context}

{map_rules}

Map Information:
Current Position Analysis: {map_single_ref}
Overall Exploration Status: {map_overall_summary}

Consider:
1. Current position and surrounding opportunities
2. Unexplored areas and potential paths
3. Previous task outcomes and lessons learned
4. Navigation options and safety considerations
5. Progress toward game objectives"""

    # 在最后添加输出格式要求
    output_format = """
Based on your analysis, provide your response in the following format:

reasoning process:
    1. Current State Analysis: "<analyze current environment and immediate situation>"
    2. Historical Context: "<analyze relevant history and reflections>"
    3. Strategic Evaluation: "<evaluate opportunities, risks, and priorities>"

task details:
    goal: "<specific, actionable objective>"

    location details:
        - screen_position: "<describe target position relative to player character using character heights as measurement. 
                           Example: '3 character heights to the right and 2 character heights up'>"
        - map_position: "<if using map, describe location on map using notable landmarks or relative to current position.
                        Example: 'In the yellow region near the Buddha statue' or 'Not applicable for UI tasks'>"
        - movement_path: "<if movement required, describe the suggested path using both screen and map references.
                         Example: 'Move right for 3 character heights while following the yellow path on map towards the temple'>"

    key_requirements: "<essential conditions or resources needed>"
    success_criteria: "<main condition that must be met>"


Note: 
- Camera adjustment **MUST** be treated as an isolated, single-step task. Due to inability to observe real-time screen changes, adjustments must be completed independently before proceeding with game evaluation
- For movement-related tasks, always specify both screen-relative positions (using character height as scale) and map positions (if map is available). For non-movement tasks, mark position fields as 'N/A' if not relevant."""

    return analysis_prompt + output_format

def new_task_planner(latest_map,current_frame,pass_task_history_summary,pass_task_reflection,map_single_ref, map_overall_summary,env_info,game_info,step):
    # 初始化多轮对话服务
    task_planning_agent = MultiTurnChatService(system_prompt=task_sys_prompt)

    # 构建prompt
    prompt = construct_task_prompt(
        latest_map, current_frame, pass_task_history_summary,
        pass_task_reflection, map_single_ref, map_overall_summary,
        env_info, game_info, step
    )
    #print(prompt)
    # 准备图片内容

    if latest_map is not None:
        frames=[current_frame,latest_map]
    else:
        frames=current_frame

    # 将文本和图片组合
    content = make_pic_content(prompt, frames)

    # 获取LLM响应
    response = task_planning_agent.chat({"role": "user", "content": content})
    action_prompt =  f"""Based on the task you just planned, break it down into specific executable actions.
Please list the specific actions needed to complete this task.

Available Controls:
{game_info.get('control_info')}  

Note that:
1. output will be directly evaluated using Python eval(), so it must be a valid Python list
2. No additional text or explanation should be added between or after these sections
3. Action list is *no longer than 3!!*.

Output Format MUST be exactly as follows:
["Action1: <action name> - <detailed description including precise measurements and requirements>","Action2: <action name> - <detailed description including precise measurements and requirements>", ...]

"""
    action_response = task_planning_agent.chat({"role": "user", "content": action_prompt})
    max_retries = 4
    retries = 0
    while retries < max_retries:
        try:
            # 尝试提取列表格式
            import re
            import ast
            # 首先尝试提取标准Python列表格式
            list_pattern = r'\[(.*?)\]'
            match = re.search(list_pattern, action_response, re.DOTALL)

            if match:
                #content = match.group(1)
                # 分割并清理action列表
                #actions = [action.strip().strip('"').strip("'") for action in re.findall(r'"([^"]+)"|\'([^\']+)\'', content)]
                actions = ast.literal_eval(match.group(0))
                if isinstance(actions, list) and actions:
                    # 组合最终响应
                    token_usage = task_planning_agent.get_token_usage()
                    return {
                        'reason_task': response,
                        'action_plan' : actions,
                        'token_use': token_usage
                    }

            # 如果没找到正确格式的列表，发送纠正提示
            correction_prompt = """Your previous response needs to be reformatted. Please provide ONLY a list of actions in this exact format (list in Python):
    ["Action1: <action name> - <detailed description including precise measurements and requirements>","Action2: <action name> - <detailed description including precise measurements and requirements>", ...]

    Important:
    1. Use double quotes for strings
    2. Include action name and detailed description
    3. Provide ONLY the list, no additional text
    4. Keep same level of detail in descriptions
   """

            action_response = task_planning_agent.chat({"role": "user", "content": correction_prompt})

        except Exception as e:
            print(f"Attempt {retries + 1} failed with error: {e}")

        retries += 1

        # 如果所有重试都失败了
    raise ValueError(f"Failed to extract valid action list after {max_retries} attempts")


    #
    # token_usage = task_planning_agent.get_token_usage()
    #
    #
    # return {
    #     'reason_task': response,
    #     'token_use': token_usage
    # }

#我觉得可以细化一下要求，和移动相关的任务，必须明确地说明目的地和当前帧画面的相对位置（以角色身高为尺度），以及在map上大概在哪。这样方便后续评估
# 如果用上了map信息，先写一个map信息的基本规则Key Map Reading Rules:
#
# Only yellow areas on the map are explorable and all yellow regions are connected to each other
# The boundaries of yellow areas are absolute barriers that cannot be crossed
# The arrow on the map shows the exact direction the player is facing
# Other map decorations (trees, buildings, Buddha statue, etc.) are purely aesthetic

