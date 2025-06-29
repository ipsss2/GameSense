from utils.ask_model import MultiTurnChatService
from utils.picture2chat import make_pic_content
import re

action_sys_prompt = '''
You are an expert game AI action planner specializing in converting high-level tasks into precise, executable action sequences.

Key Responsibilities:
1. Convert task descriptions into specific control sequences
2. Ensure accurate timing and duration for each action
3. Maintain action safety and efficiency
4. Generate properly formatted action code that can be directly executed

Important Guidelines:
1. All outputs must be in valid Python dictionary list format
2. Each action must include both description and corresponding control code
3. Control codes must use only valid game controls
4. All durations must be reasonable and safe
'''


# def validate_response(response_text, game_info):
#     """验证输出是否符合要求"""
#     try:
#         # 检查格式完整性
#         if 'action_list:' not in response_text or 'action_code:' not in response_text:
#             return False, "Missing required sections"
#
#         # 提取并验证action_code
#         action_code_str = response_text.split('action_code:')[1].strip()
#         action_code = eval(action_code_str)
#
#         # 验证action_code格式
#         if not isinstance(action_code, list):
#             return False, "action_code must be a list"
#
#         # 验证每个动作元组
#         valid_keys = set(game_info.get('Mapping_info', {}).keys())
#         for action in action_code:
#             if not isinstance(action, tuple) or len(action) != 2:
#                 return False, "Each action must be a tuple of (key, duration)"
#             key, duration = action
#             if not isinstance(duration, (int, float)) or duration <= 0:
#                 return False, "Duration must be a positive number"
#             if key not in valid_keys:
#                 return False, f"Invalid key: {key}"
#
#         return True, "Valid response"
#
#     except Exception as e:
#         return False, f"Validation error: {str(e)}"

def generate_action_prompt(game_info,reason_task,action_plan):
    prompt = f"""
You are an expert game AI action planner specializing in converting high-level action into precise, executable action sequences.

Your current task:
{reason_task}

The action plan for task:
{action_plan}

Available Controls:
{game_info.get('control_info')}  

Additional Action Information:
{game_info.get('additional_action_info')} 

Requirements:
1. Convert each action into specific control sequences
2. Provide both action description and control code
3. Ensure precise timing for each control input
4. Consider safety in all actions

Output Format MUST be exactly as follows:
[
    {{
        "action_name_description": "<original action description>",
        "action_code": [("<key>", <duration>), ...]
    }},
    ...
]

Example Output:
[
    {{
        "action_name_description": "Move Forward - Move 3 character heights forward",
        "action_code": [("W", 2.0)]
    }},
    {{
        "action_name_description": "Jump and Interact - Jump over obstacle and press button",
        "action_code": [("SPACE", 0.1), ("E", 0.1)]
    }}
]

Note:
1. Output will be evaluated using Python ast.literal_eval()
2. Use only valid control keys: {list(game_info.get('Mapping_info', {}).keys())}
3. All durations must be positive numbers
4. Maintain exact format with no additional text
"""
    return prompt

def action_builder(RAG4actions, current_frame, reason_and_task, game_info,action_plan):
    action_planning_agent = MultiTurnChatService(system_prompt=action_sys_prompt)
    prompt = generate_action_prompt(game_info,reason_and_task,action_plan)

    expected_length = len(action_plan)
    # Handle None values
    rag_info = str(RAG4actions) if RAG4actions else "No similar action experiences available"

    f_prompt = f"""
You are an expert game AI action planner specializing in converting high-level action into precise, executable action sequences.

your current task:
{reason_and_task}

Similar Past Actions:
{rag_info}

{prompt}
"""

    content = make_pic_content(f_prompt, current_frame)
    response = action_planning_agent.chat({"role": "user", "content": content})
    #print(response)

    # # 解析响应以提取action_list和action_code
    # response_text = response

    max_attempts = 4  # 最大重试次数
    attempt = 0

    while attempt < max_attempts:
        try:
            import ast
            # 提取列表部分
            # 清理响应文本，只保留JSON格式部分
            # 查找第一个 '[' 和最后一个 ']'
            start_idx = response.find('[')
            end_idx = response.rfind(']')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No valid list format found in response")

            # 提取JSON部分
            json_str = response[start_idx:end_idx + 1]
            print(json_str)

            # 解析响应
            action_list = ast.literal_eval(json_str)

            # 验证列表长度
            if len(action_list) != expected_length:
                raise ValueError(
                    f"Action list length {len(action_list)} does not match expected length {expected_length}")

            # 验证格式
            for action in action_list:
                if not isinstance(action, dict):
                    raise ValueError("Each action must be a dictionary")

                # 验证必需的键存在
                if "action_name_description" not in action or "action_code" not in action:
                    raise ValueError("Missing required fields in action dictionary")

                # 只验证action_code的格式
                if not isinstance(action["action_code"], list):
                    raise ValueError("action_code must be a list")

                for code in action["action_code"]:
                    if not isinstance(code, tuple) or len(code) != 2:
                        raise ValueError("Each code must be a tuple of (key, duration)")
                    if not isinstance(code[1], (int, float)) or code[1] <= 0:
                        raise ValueError("Duration must be a positive number")
                    if code[0] not in game_info.get('Mapping_info', {}):
                        raise ValueError(f"Invalid control key: {code[0]}")

            # 验证通过，返回结果
            token_usage = action_planning_agent.get_token_usage()
            return action_list, token_usage

        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise ValueError(f"Failed to get valid response after {max_attempts} attempts. Last error: {str(e)}")

            correction_prompt = f"""
        Your previous response needs correction. Error: {str(e)}

        Important:
        1. The number of actions MUST be exactly {expected_length}
        2. Each action MUST contain both "action_name_description" and "action_code"
        3. Each action_code MUST be a list of valid control tuples

        Please provide a corrected response following EXACTLY this format:
        [
            {{
                "action_name_description": "<keep original action description>",
                "action_code": [("<key>", <duration>), ...]
            }},
            ...
        ] * {expected_length} actions

        Valid control keys: {list(game_info.get('Mapping_info', {}).keys())}

        Provide ONLY the corrected list.
        """
            response = action_planning_agent.chat({"role": "user", "content": correction_prompt})


#输入，task，action_plan等信息，输出具体的按键与数值
def new_action_build():
    pass