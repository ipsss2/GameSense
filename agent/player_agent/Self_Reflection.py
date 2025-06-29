
from utils.ask_model import MultiTurnChatService
from utils.picture2chat import make_pic_content


task_sys_prompt='''
 Assume you are a helpful AI assistant for PC Games, equipped to handle a wide range of tasks in the game. Your advanced capabilities enable you to process and interpret gameplay screenshots and other relevant information.
 Your task is to examine these inputs, interpret the in-game context, and determine whether the executed action takes effect.
'''

action_sys_prompt='''
    Assume you are a helpful AI assistant for PC Games, specialized in analyzing action execution effectiveness. 
    Your expertise lies in evaluating action sequence design and control implementation.
    Your task is to examine the outcome of each action through gameplay screenshots, assess the action design and implementation, 
    and provide insights for improvement.
'''


def Prompt_reflection_task(env_state_info, reason_and_task):
    prompt = f"""
    Task Content:
    {reason_and_task}

    Initial Environment:
    {env_state_info}

    The keyframe image has been uploaded during task execution
    
    Please analyze according to the following steps:

    1. Scene Analysis:
    Please carefully analyze these frames:
    - Changes in player behavior and status
    - Changes in environment and objectives
    - Occurrence of key events

    2. Completion Assessment:
    Based on the above scene analysis, determine:
    - Whether the task objective was achieved
    - Provide a clear complete/incomplete judgment
    - Provide specific basis for judgment

    3. Cause Analysis:
    If task completed:
    - Which aspects of the task design were appropriate
    - What factors contributed to the task's success

    If task failed:
    - What problems existed in the task design
    - What were the specific reasons for failure

    4. Improvement Suggestions:
    Based on the above analysis:
    - Specific directions for improvement
    - Task design suggestions for similar scenarios in the future

    Please output in the following format:
    Scene Analysis: [Concise analysis of scene changes]
    Completion Status: [Complete/Incomplete, with judgment basis]
    Cause Analysis: [Task design analysis and success/failure reasons]
    Improvement Suggestions: [Specific actionable suggestions]
    
    Example Output:
    Scene Analysis: Player attempted to open the chest but was interrupted by enemy attack
    Completion Status: Incomplete - chest remains closed in final frame
    Cause Analysis: Task timing was poor, should check for nearby enemies first
    Improvement Suggestions: Add enemy check before initiating chest opening action
    """

    return prompt

def Prompt_reflection_action(task_content, action_list, action_code):
    prompt = f"""
    Task Content:
    {task_content}

    Designed Action List:
    {action_list}

    Action Code (Key-Mouse Mapping):
    {action_code}

    The keyframe image has been uploaded for each action's completion moment.

    Please analyze according to the following steps:

    1. Scene Analysis:
    Please carefully analyze these frames:
    - Effect of each executed action
    - Changes in game state after actions
    - Any unexpected outcomes or side effects

    2. Action List Design Assessment:
    Evaluate the action sequence design:
    - Whether the action decomposition is appropriate for the task
    - If the action sequence is logical and efficient
    - Whether any necessary actions are missing or redundant

    3. Action Code Effectiveness:
    Analyze the key-mouse mapping implementation:
    - Whether the controls achieved their intended effects
    - Timing and coordination of inputs
    - Any execution issues or control problems

    4. Improvement Suggestions:
    Based on the above analysis:
    - How to optimize the action list design
    - How to improve the action code 
    - Better approaches for similar tasks in future

    Please output in the following format:
    Scene Analysis: [Concise analysis of each action's effect]
    Action Design Analysis: [Evaluation of action list design]
    Code Implementation Analysis: [Assessment of control mapping effectiveness]
    Improvement Suggestions: [Specific actionable suggestions for both design and implementation]
    """

    return prompt
def self_reflection_task(frame_list, env_state_info, reason_and_task):
    prompt=Prompt_reflection_task(env_state_info, reason_and_task)
    task_reflection=MultiTurnChatService(system_prompt=task_sys_prompt)
    content=make_pic_content(prompt, frame_list)
    response = task_reflection.chat({"role": "user", "content": content})
    token_use = task_reflection.get_token_usage()

    return response,token_use

def self_reflection_action(frame_list, reason_and_task, action_list, action_code):
    prompt = Prompt_reflection_action(reason_and_task, action_list, action_code)
    action_reflection = MultiTurnChatService(system_prompt=action_sys_prompt)
    content = make_pic_content(prompt, frame_list)
    response = action_reflection.chat({"role": "user", "content": content})
    token_use = action_reflection.get_token_usage()

    return response, token_use

task_new_sys_prompt='''
'You are an expert game analyst specializing in task reflection and evaluation. Your role is to:
1. Analyze all gameplay screenshots and state changes to understand what happened during task execution
2. Evaluate task completion status with concrete evidence
3. Identify and analyze issues at task design, action planning, and execution levels
4. Provide specific recommendations when needed

Always provide detailed, objective analysis following the exact format requested in the prompt.'''
def generate_task_level_prompt(pass_task_info, pass_env_info, current_env_info, pass_action_code, has_map=False):
    base_prompt = f"""Analyze the previous task execution using the following information:

    1. Task Information:
    {pass_task_info}

    2. Environment States:
    - Before task execution: {pass_env_info}
    - After task execution: {current_env_info}

    3. Action Design:
    - Planned action list and Execution code: 
    {pass_action_code}

    Please conduct your analysis in these sequential steps and provide a detailed response in the following format:

    1. VISUAL ANALYSIS
    Provide a clear description of:
    - What happened during the task execution based on all the gameplay screenshots
    - Key UI changes (if in UI screens), character movements, interactions observed, and Notable changes in environment states
    {" - Changes between initial and final maps (The last two pictures)" if has_map else ""} 

    2. TASK COMPLETION EVALUATION
    State clearly:
    - Whether the task was successfully completed
    - Specific evidence from screenshots or state changes supporting your conclusion

    3. ISSUE ANALYSIS (if any problems occurred)
    Analyze at three levels:
    a) Task Design Level
        - Any issues with task design given the game state
        - Problems with task objectives or prerequisites

    b) Action Planning Level
        - Issues with the planned action sequence
        - Problems with action strategy or logic

    c) Action Execution Level
        - Problems with specific control inputs
        - **Issues with duration of actions**

    4. NEXT STEP RECOMMENDATION
    If task failed:
    - Specific suggestions to complete the task in the **CURRENT** state

    If task succeeded:
    - Simply state that the task was completed successfully and no modifications are needed

    Please provide your analysis in the following format:
VISUAL ANALYSIS:
<Describe the sequence of events observed in gameplay screenshots, including UI changes (if in UI screens), character actions, and any significant state changes>
{" <Describe any relevant changes observed between initial and final maps>" if has_map else ""}

TASK COMPLETION EVALUATION:
Status: <SUCCESS/FAILURE>
Evidence: <List specific observations from screenshots or state changes that support your status determination>

ISSUE ANALYSIS:
Task Design Level:
<Evaluate if there are any issues with how the task was designed and specified. If no issues, explicitly state that>

Action Planning Level:
<Analyze if the planned sequence of actions was appropriate and complete. Identify any logical gaps or problems>

Action Execution Level:
<Assess if there were any issues with the specific implementation of actions, such as timing or input problems>

NEXT STEP RECOMMENDATION:
<If task failed: Provide specific suggestions for task completion given the current state>
<If task succeeded: Simply state that the task was completed successfully and no modifications are needed>

"""

    return base_prompt

action_new_sys_prompt='''
You are an expert game action analyst specializing in analyzing and improving game control implementations. Your role is to:
1. Analyze gameplay screenshots to understand the execution effects of each action
2. Evaluate action code design and implementation quality
3. Provide reusable insights for similar actions in the future
4. Suggest specific improvements for action code design

Always provide detailed, objective analysis following the exact format requested in the prompt.
'''

def new_task_reflection(pass_task_frames, maps_task_eval, pass_task_info, pass_env_info,current_env_info,pass_action_code):
    task_reflection_agent = MultiTurnChatService(system_prompt=task_new_sys_prompt)

    # Check if we have map information
    has_map = maps_task_eval is not None
    if has_map:
        frames = pass_task_frames + maps_task_eval
        print('we use map frame')
    else:
        frames = pass_task_frames
        print('we do not use map frame')

    # Generate base prompt with all necessary parameters
    prompt_gather = generate_task_level_prompt(
        pass_task_info=pass_task_info,
        pass_env_info=pass_env_info,
        current_env_info=current_env_info,
        pass_action_code=pass_action_code,
        has_map=has_map
    )
    #print(prompt_gather)
    content = make_pic_content(prompt_gather, frames)
    response = task_reflection_agent.chat({"role": "user", "content": content})
    token_usage = task_reflection_agent.get_token_usage()

    return response, token_usage


#- Planned actions: {pass_action_list}

def generate_action_level_prompt(pass_task_info, pass_action_code):
    base_prompt = f"""Analyze the previous action execution using the following information:

    1. Screenshot Sequence Rules:
    - For WASD movement actions lasting over 2 seconds:
        * Screenshots are captured every 2 seconds during the movement
    - For all other key/mouse actions:
        * Only two screenshots are captured: one before and one after the action
    This helps track continuous movements and precise action effects.

    2. Task Context:
    {pass_task_info}

    3. Action plan and code list:
    {pass_action_code}

    Please conduct your analysis in these sequential steps and provide a detailed response in the following format:

    1. ACTION EXECUTION ANALYSIS
    For each action in the sequence, analyze:
    - Initial state and final state from screenshots
    - Whether the action achieved its intended effect
    - Timing and smoothness of execution
    - Any unexpected behaviors or side effects

    2. ACTION CODE EVALUATION
    For each action implementation, evaluate:
    - Appropriateness of key/mouse mapping choices
    - Timing duration settings
    - Action sequence coordination
    - Code efficiency and reliability

    3. SUCCESS/FAILURE ANALYSIS
    For each action, determine:
    - Whether it succeeded or failed
    - Root causes of any failures:
        a) Input mapping issues
        b) Timing problems
        c) Sequence coordination issues
        d) Environmental factors

    4. REUSABILITY ANALYSIS
    Analyze each action's potential for reuse:
    - Common scenarios where this action pattern could apply
    - Required prerequisites and conditions
    - Potential adaptations needed for different contexts
    - Limitations and constraints

    5. IMPROVEMENT RECOMMENDATIONS
    Provide specific suggestions for:
    - Better key/mouse mapping choices
    - Optimal timing parameters
    - Enhanced sequence coordination
    - More robust implementation patterns
    
    Note that:
1. output will be directly evaluated using Python eval(), so it must be a valid Python list of dicts
2. No additional text or explanation should be added between or after these sections
    After completing your analysis, output a list of dictionaries in the following format:

    ```python
    [
        {{
            "action_name_description": "<original action description from action_name_description>",
            "action_code": "<corresponding action code tuple from action_code>",
            "reflection": {{
                "execution_analysis": "<summary of execution analysis>",
                "code_evaluation": {{
                    "status": "<SUCCESS/PARTIAL SUCCESS/FAILURE>",
                    "quality_analysis": "<implementation quality summary>"
                }},
                "success_failure_analysis": "<detailed analysis of what worked/didn't work>",
                "reusability": {{
                    "applicable_scenarios": "<list of potential reuse cases>",
                    "prerequisites": "<required conditions>",
                    "limitations": "<known constraints>"
                }},
                "improvements": "<specific suggestions for implementation improvements>"
            }}
        }},
        # ... repeat for each action
    ]
    ```

    Ensure your response ends with this structured list for easy parsing. Format it exactly as shown above.
    """
    return base_prompt


def new_action_reflection(pass_action_frames, pass_task_info, pass_action_code,max_retries=4):
    action_reflection_agent = MultiTurnChatService(system_prompt=action_new_sys_prompt)

    prompt = generate_action_level_prompt(
        pass_task_info=pass_task_info,
        pass_action_code=pass_action_code
    )


    content = make_pic_content(prompt, pass_action_frames)
    response = action_reflection_agent.chat({"role": "user", "content": content})

    retries = 0
    while retries < max_retries:
        try:

            import re
            pattern = r"```python\s*([\s\S]*?)\s*```"
            match = re.search(pattern, response)

            if match:
                dict_list = eval(match.group(1))

                for item in dict_list:
                    required_keys = {'action_name_description', 'action_code', 'reflection'}
                    if not all(key in item for key in required_keys):
                        raise ValueError("Missing required keys in dictionary")

                token_usage = action_reflection_agent.get_token_usage()
                return dict_list, token_usage

            correction_prompts = [
                """Your previous response didn't contain a properly formatted Python dictionary list. 
                Please reformat your analysis as a Python list of dictionaries exactly as specified in the original prompt. 
                Include only the dictionary list, enclosed in ```python ``` code blocks.""",

                """I still cannot parse your response. Please ensure your response contains ONLY a Python list of dictionaries
                with exactly this structure for each action:
                {
                    "action_name_description": "<action description>",
                    "action_code": "<action code>",
                    "reflection": {
                        "execution_analysis": "<analysis>",
                        "code_evaluation": {"status": "<status>", "quality_analysis": "<analysis>"},
                        "success_failure_analysis": "<analysis>",
                        "reusability": {"applicable_scenarios": "<scenarios>", "prerequisites": "<conditions>", "limitations": "<limits>"},
                        "improvements": "<suggestions>"
                    }
                }""",

                """Final attempt: Please output ONLY a Python list of dictionaries. Nothing else. 
                Start with ```python and end with ```. Ensure all dictionary keys and values are properly formatted."""
            ]

            correction_prompt = correction_prompts[min(retries, len(correction_prompts) - 1)]
            response = action_reflection_agent.chat({"role": "user", "content": correction_prompt})

        except Exception as e:
            print(f"Attempt {retries + 1} failed with error: {e}")

        retries += 1


    raise ValueError(f"Failed to extract valid dictionary list after {max_retries} attempts")


