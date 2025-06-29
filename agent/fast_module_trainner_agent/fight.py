
# action_optimizer.py

import os
import ast
from utils.ask_model import MultiTurnChatService
from utils.picture2chat import process_image_path_to_content 

def validate_code_syntax(code: str):
    """
    Validates if the Python code has correct syntax.
    :param code: The Python code string to validate.
    :return: A tuple (is_valid, error_message).
    """
    try:
        ast.parse(code)  # Try to parse the code into an Abstract Syntax Tree
        return True, None
    except SyntaxError as e:
        return False, str(e)

def generate_action_analysis_prompt(goal, code):
    """Generates the initial analysis prompt for the LLM."""
    return f"""
As a world-class expert in game AI and strategy, your task is to analyze an agent's combat performance and suggest improvements to its core logic.

**Primary Goal:**
{goal}

**Current Combat Logic (Python Code):**
```python
{code}
```

**Your Task:**
 **Analyze the Python code:** Review the current `execute_fight` function. What is its strategy? Does it seem simplistic, repetitive, or inefficient?
"""

def opt_action(fight_logic_code, screenshot_dir, suggest_pass=None):
    """
    Optimizes the fighting action logic using an LLM based on performance screenshots.

    :param fight_logic_code: A string containing the current 'execute_fight' function code.
    :param screenshot_dir: The directory path containing screenshots from the last run.
    :param suggest_pass: Optional. Analysis from a previous, failed optimization attempt.
    :return: A tuple containing (new_optimized_code, suggestions_from_llm).
    """
    # --- Define the Goal and System Prompt ---
    optimization_goal = """
    The agent must learn to defeat a powerful boss in a video game. The primary objective is to **maximize damage dealt to the boss** while **minimizing damage received**. An effective strategy involves learning the boss's attack patterns, dodging incoming attacks, and counter-attacking during windows of opportunity. Simple, repetitive attacks are unlikely to succeed.
    """
    
    system_prompt = "You are an expert game AI strategist. Your goal is to refine an agent's Python-based combat logic by analyzing its performance through code and screenshots."

    # --- Initialize the Chat Service ---
    opt_chat = MultiTurnChatService(system_prompt=system_prompt)

    # --- Prepare the Multi-modal Input ---
    # 1. Generate the text prompt
    analysis_prompt = generate_action_analysis_prompt(optimization_goal, fight_logic_code)
    if suggest_pass:
        analysis_prompt = (
            "A previous analysis was provided. Please consider these points as well:\n"
            + suggest_pass + "\n\n" + analysis_prompt
        )
    
    # 2. Process images from the screenshot directory
    image_contents = []
    if os.path.exists(screenshot_dir):
        all_files = sorted([f for f in os.listdir(screenshot_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # To avoid overwhelming the model, let's select a maximum of 15 representative images
        sample_files = all_files[::len(all_files)//15] if len(all_files) > 15 else all_files
        
        print(f"Found {len(all_files)} screenshots. Sampling {len(sample_files)} for analysis.")

        for filename in sample_files:
            image_path = os.path.join(screenshot_dir, filename)
            # Use the provided utility to convert image to the required chat format
            image_content = process_image_path_to_content(image_path)
            if image_content:
                image_contents.append(image_content)

    else:
        print(f"Warning: Screenshot directory not found at '{screenshot_dir}'. Proceeding without visual analysis.")

    # 3. Assemble the full context for the LLM
    context = [{"type": "text", "text": analysis_prompt}] + image_contents

    # --- First LLM Call: Get Strategic Suggestions ---
    print("\n--- Sending analysis request to LLM... ---")
    suggestions = opt_chat.chat({"role": "user", "content": context})
    print("\n--- LLM Suggestions Received: ---")
    print(suggestions)

    # --- Second LLM Call: Generate New Code Based on Suggestions ---
    code_generation_prompt = f"""
Based on the following strategic analysis, please rewrite the Python function `execute_fight`.

**Analysis & Suggestions:**
{suggestions}

**Original Code:**
```python
{fight_logic_code}
```

**Instructions:**
- Your response MUST be only the raw Python code for the new function.
- Do NOT include any explanations, markdown formatting like "```python", or any text other than the code itself.
- Ensure the function signature `def execute_fight(fight_controller, detector, time):` remains unchanged.
- The code should be complete, syntactically correct, and ready to be written directly to a .py file.
"""
    
    print("\n--- Requesting updated code from LLM... ---")
    new_code = opt_chat.chat({"role": "user", "content": code_generation_prompt})

    # --- Validate the Generated Code in a Loop ---
    is_valid, error_msg = validate_code_syntax(new_code)
    retry_count = 0
    while not is_valid and retry_count < 3:
        retry_count += 1
        print(f"\nGenerated code has a syntax error: {error_msg}")
        print(f"Retrying... (Attempt {retry_count}/3)")
        
        correction_prompt = f"""
The code you previously provided had a syntax error: {error_msg}.
Please fix the error and provide only the raw, complete, and syntactically correct Python code for the `execute_fight` function. Do not add any extra text or markdown.
"""
        new_code = opt_chat.chat({"role": "user", "content": correction_prompt})
        is_valid, error_msg = validate_code_syntax(new_code)

    if not is_valid:
        print("\nCRITICAL: Failed to generate valid code after multiple retries. Aborting optimization.")
        return None, suggestions # Return None for code if it's still invalid

    print("\n--- Successfully generated and validated new code! ---")
    # print(new_code) # Uncomment for debugging
    
    return new_code, suggestions
