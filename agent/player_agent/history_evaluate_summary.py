
from utils.ask_model import MultiTurnChatService
from utils.picture2chat import make_pic_content
history_summary_sys_prompt = '''
You are an expert game historian. Your role is to synthesize gameplay history into a concise, informative narrative paragraph that captures key events, strategies, and insights relevant for future decision-making.
'''

def generate_history_summary_prompt(history_logs):
    base_prompt = f"""
Based on the following game history logs, generate a single coherent paragraph (approximately 150 words) that:
- Summarizes the key events chronologically
- Highlights critical decisions and their outcomes
- Identifies important patterns or strategies
- Notes any significant environmental changes
- Includes relevant insights for future tasks

Game History Logs:
{history_logs}

Your summary should be clear, concise, and focused on information that will be most valuable for future task reasoning.
"""
    return base_prompt

def history_evaluate_summary(history_str):
    history_summary_agent = MultiTurnChatService(system_prompt=history_summary_sys_prompt)
    prompt = generate_history_summary_prompt(history_str)
    response = history_summary_agent.chat({"role": "user", "content": prompt})
    token_usage = history_summary_agent.get_token_usage()

    return response, token_usage