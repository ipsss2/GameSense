
from utils.ask_model import MultiTurnChatService
import ast
import json

def validate_code_syntax(code: str):

    try:
        ast.parse(code)  
        return True, None
    except SyntaxError as e:
        return False, str(e)


q_net="""
class QNetwork(nn.Module):
    def __init__(self, embedding_dim, state_dim, action_dim, dropout_rate=0.1):
        super(QNetwork, self).__init__()
        self.embedding_fc1 = nn.Linear(embedding_dim, 512)
        self.embedding_fc2 = nn.Linear(512, 256)
        self.state_fc1 = nn.Linear(state_dim, 64)
        self.state_fc2 = nn.Linear(64, 64)
        self.action_embedding = nn.Embedding(
            num_embeddings=action_dim + 1,
            embedding_dim=32
        )
        self.action_lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        self.layer_norm = nn.LayerNorm(64)
        combined_dim = 256 + 64 + 64
        self.batch_norm = nn.BatchNorm1d(combined_dim)
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, embedding, state_info, action_history):
        embedding_out = torch.relu(self.embedding_fc1(embedding))
        embedding_out = torch.relu(self.embedding_fc2(embedding_out))
        state_out = torch.relu(self.state_fc1(state_info))
        state_out = torch.relu(self.state_fc2(state_out))
        action_embedded = self.action_embedding(action_history)
        action_out, _ = self.action_lstm(action_embedded)
        action_out = action_out[:, -1, :]  
        action_out = self.layer_norm(action_out)  

        combined = torch.cat([embedding_out, state_out, action_out], dim=1)
        combined = self.batch_norm(combined)  
        x = torch.relu(self.fc1(combined))
        return self.fc2(x)
"""
def design_reward(action_state_changes):
    action_length = len(action_state_changes)
    prompt=[f"""
---

### Task Description

I am designing a reinforcement learning (RL) environment to train AI in learning optimal strategies during boss battles. To guide AI learning, a reward function system needs to be designed including immediate rewards, combination rewards, and long-term goal rewards.
---
### RL Network Information
This reward is designed to work with the following qnet-based double DQN network code:
{q_net}

---
### Environment Information

#### State Format Description

The format of `prev_state` and `next_state` is as follows:

```python
{{
    "blood_percentage": <float>,   #Player's health percentage, range [0,1]
    "boss_percentage": <float>,   #Boss's health percentage, range [0,1]
    "mana_percentage": <float>,   #Player's mana percentage, range [0,1]
    "stamina_percentage": <float>, #Player's stamina percentage, range [0,1]
    "potion_percentage": <float>  #Player's remaining medication percentage, range [0,1]
}}
```

- **`prev_state`**: The state before executing the action.
- **`next_state`**: The state after executing the action.

#### Action Index Description

`action_idx` is the index of the action in the action space (of type `int`). For example, `action_idx=0` corresponds to the action `Move Forward`.


#### Temporal Information
- `action_history`: A queue of the last 20 actions
- `episode_start_time`: The start time of the episode
- `step_time`: The time of the current step
- `step`: The current step number

### Action Space and Expected Effects

Below is the complete action space and the expected state changes for each action:
action_state_changes={action_state_changes}

---

### Task Goal

Please design a **reward function** based on the following requirements:
1. **Training Goal**：
   - The most **important** goal is to make the boss's health decrease quickly


2. **Reward Logic**：
   - Reward the AI for actions that help achieve the goal (making the boss's health decrease), and penalize harmful or ineffective actions.
   - The reward design should include multiple reward structures:
       - 1.boss health decrease reward: Reward or penalize based on the change in boss health.
       - 2.player health change reward: Reward or penalize based on the change in player health.
       - 3.player health unchanged reward: Only for dodge actions, design carefully to avoid the model learning only to dodge
       - 4.combination reward: Reward or penalize based on the action usage history.

3. **Special Cases**：
   - When the game ends (`done=True`), reward or penalize based on the completion degree of the training goal, with fine-grained linear rewards or penalties.
   - The boss will not be defeated during training and the boss's health will always be **greater than 0**.
   - Attack success rewards should be higher.

---

### Output Requirements

Please generate a complete Python function `reward_function` based on the above information, in the following format:

```
def reward_function(prev_state, next_state, action_idx, done, action_history, action_state_changes, episode_start_time, step_time, step):
    # Initialize reward
    reward = 0.0

    # Game end logic
    if done:
        # Reward based on the reduction in boss health
        if prev_state["boss_percentage"]<0.05:
            reward += 0
        else:
            boss_health_reduction = 1-prev_state["boss_percentage"]
            if boss_health_reduction >= 0.5:
                reward +=   
            elif boss_health_reduction >= 0.2:
                reward +=   
            elif boss_health_reduction >= 0.1:
                reward +=  
            else:
                reward -=   
        return reward

    # boss health decrease reward: Reward or penalize based on the change in boss health.
    boss_health_change = prev_state["boss_percentage"] - next_state["boss_percentage"]
    if boss_health_change > 0.0:
        boss_blood_reward=   
        reward += min(boss_blood_reward,20)


    # player health change reward: Reward or penalize based on the change in player health.
    player_health_change = next_state["blood_percentage"] - prev_state["blood_percentage"]
    reward +=   

    # player health unchanged reward: Only for dodge actions, design carefully to avoid the model learning only to dodge
    action = action_state_changes[action_idx]
    if action["action_name"] == "Dodge":
        if player_health_change == 0:
            reward +=   
        else:
            reward -= 

    if action["action_name"] == "Drink Health Potion":
        if prev_state["blood_percentage"]>0.7:
            reward -= 

        if prev_state["potion_percentage"] == 0.0:
            reward -= 

    # combination reward: Reward or penalize based on the action usage history.
    def calculate_combo_reward(action_history):
        combo_reward = 0
        # Reward for four consecutive light attacks
        if action_history[-2:] == [4, 4]:
            combo_reward += 
        # Penalize excessive dodging
        if action_history.count(6) > 10:
            combo_reward -= 
        # Penalize frequent potion usage
        if action_history.count(7) > 4 or action_history[-2:]==[7,7]:
            combo_reward -= 
        return combo_reward

    reward += calculate_combo_reward(action_history)

    return reward
```

---

### Insertion Instructions

1. **Action Space**：If using action_state_changes, make sure there are {action_length} actions in total, and they **must correspond to the content provided above**, otherwise the program will not run.
2. **Reward Mechanism**：Design reasonable reward values and logic based on your goals; the reward mechanism must be close to the training goal and compatible with the double DQN and the provided qnet.

""",'I accessed you through an API, so I converted your answer into a form that only includes code and comments, and does not contain elements such as “```python”. This way, I can directly write into a py file run:']
    return prompt

def get_org_reward_function(action_state_changes):
    reward_designer=MultiTurnChatService(system_prompt='You are an AI training expert specializing in reinforcement learning reward function design. ')
    promptlist=design_reward(action_state_changes)
    code=''
    for prompt in promptlist:
        print(prompt)
        code=reward_designer.chat({"role": "user", "content": prompt})
        print(code)
    code_true,message=validate_code_syntax(code)
    while code_true is False:
        code=reward_designer.chat({"role": "user", "content": promptlist[-1]+message})
        code_true,message=validate_code_syntax(code)
    return code



def opt_reward(reward_code, action_space_description,history_info,suggest_pass=None):
    training_goal='''
1. **Training Goal**：
   - The most **important** goal is to make the boss's health decrease quickly

2. **Current Reward Logic**：
   - Reward the AI for actions that help achieve the goal (making the boss's health decrease), and penalize harmful or ineffective actions.
   - The reward design should include multiple reward structures:
       - 1.boss health decrease reward: Reward or penalize based on the change in boss health.
       - 2.player health change reward: Reward or penalize based on the change in player health.
       - 3.player health unchanged reward: Only for dodge actions, design carefully to avoid the model learning only to dodge
       - 4.combination reward: Reward or penalize based on the action usage history.

3. **Special Cases**：
   - When the game ends (`done=True`), reward or penalize based on the completion degree of the training goal, with fine-grained linear rewards or penalties.
   - The boss will not be defeated during training and the boss's health will always be **greater than 0**.
   - Attack success rewards should be higher.
'''
    opt_chat=MultiTurnChatService(system_prompt='You are an expert in the field of reinforcement learning, skilled in analyzing agent training objectives, designing reward functions, and matching action spaces with strategies. Your task is to assist users in analyzing reward designs in reinforcement learning systems, Determine whether the reward logic is reasonable, consistent with the agent\'s training objectives, and whether there are potential issues caused by reward design (e.g, strategy bias, reward sparsity, etc.). You also need to combine user-provided training statistics and charts, identify potential issues, and provide specific suggestions for improvement.')
    suggest_prompt=generate_rl_analysis_prompt(training_goal,reward_code,action_space_description,history_info)
    if suggest_pass is not None:
        suggest_prompt= 'The problem analysis and suggestions for the reward design given in the previous round are as follows：\n' + suggest_pass +'\n'+suggest_prompt


    print(suggest_prompt)
    total_reward_curve_base64 = history_info['latest_file_visualizations']['total_reward_curve']
    total_steps_reward_trend_base64 = history_info['total_steps_reward_trend_base64']
    boss_percentage_trend_visualization = history_info['boss_percentage_trend_visualization']

    context=[
        {
            "type": "text",
            "text": suggest_prompt
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{total_reward_curve_base64}",
                "detail": "low"
            }
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{total_steps_reward_trend_base64}",
                "detail": "low"
            }
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{boss_percentage_trend_visualization}",
                "detail": "low"
            }
        }
    ]

    #print(suggest_prompt)
    suggest=opt_chat.chat({"role": "user", "content": context})
    print(suggest)

    code_prompt=f'''
    Please revise reward_function() according to these suggestions：\n
    {reward_code}
    Pay attention to keeping the input and output of reward_function() consistent so that it can run. I am accessing you through an API, so I will convert your answer to a form that only contains code and comments, and does not contain elements such as "python". This way, you can directly write it to a py file to run:
    '''
    code=opt_chat.chat({"role": "user", "content": code_prompt})
    print(code)
    code_true, message = validate_code_syntax(code)
    while code_true is False:
        code = opt_chat.chat({"role": "user", "content": 'Not pure code, rewrite a pure code without elements such as “```python”, so that it can be directly written to a py file for execution'})
        code_true, message = validate_code_syntax(code)
        print(code)
    return code,suggest





def generate_rl_analysis_prompt(training_goal, reward_design_code, action_space_description, history_info):


    training_statistics=history_info["latest_file_statistics"]
    action_frequency = training_statistics.get("action_frequency", {})
    action_avg_rewards = training_statistics.get("action_average_rewards", {})
    action_total_rewards = training_statistics.get("action_total_rewards", {})

    action_frequency_str = json.dumps(action_frequency, indent=4, ensure_ascii=False)
    action_avg_rewards_str = json.dumps(action_avg_rewards, indent=4, ensure_ascii=False)
    action_total_rewards_str = json.dumps(action_total_rewards, indent=4, ensure_ascii=False)

    prompt = f"""
We are training a reinforcement learning (RL) agent with the goal of enabling the agent to complete tasks in the following environment:

### RL Network Information
This reward is designed to adapt to the double DQN network composed of QNETs as shown in the following code:
{q_net}

### 1. Training Targets
{training_goal}

---

### 2. Reward Design
The following is our current reward design logic, with the following code:

```python
{reward_design_code}
```

---

### 3. Action Space
The following is the action space of the intelligent agent and the design intention of each action:
{action_space_description}

---

### 4. Statistics and Visualization of Training Data
The following are statistical data and charts extracted from the last ten training logs:

- **Distribution of action frequency**：
```json
{action_frequency_str}
```

- **The average reward for each action**：
```json
{action_avg_rewards_str}
```

- **The overall reward for each action**：
```json
{action_total_rewards_str}
```

- **Visual charts**：
The uploaded images are the total reward curve for the last ten training sessions, the total number of training steps for each round, and the line graph of the overall reward. The final health and reward changes of the boss after dozens of rounds of training are shown in the images

---
### 5. analysis requirements

Please analyze based on the above information:

1. **Is the reward design reasonable**：
   - Is the reward logic consistent with the training objectives?
   - Is there a potential conflict caused by reward design? For example:
     - Is it easy for rewards to lead intelligent agents to adopt extreme behaviors (such as excessive evasion, neglect of resource management, etc.)?
     - Is the reward too sparse or lacking gradient, making it difficult to optimize the current policy network?
   - Is the current reward logic reasonable and comprehensive, and is there a need to adjust or add reward logic

2. **Match between action space and reward design**：
   - Does the average reward of an action reflect the design intent of the action?
   - Is the design of reward values for actions reasonable?
   - Whether the total reward value of certain actions has deviations (such as negative values or being too low) may conflict with the reward design.

3. **Data and Chart Analysis**：
   - Can reward design issues be identified from the training curve?
   - Are there any suggestions for improving reward design to better guide agents to learn strategies that align with their goals?
   - Can we achieve the only important training goal of reducing the boss's health?

4. **Improvement suggestions**：
   - If there are any issues with the reward design, please provide detailed improvement suggestions, with a focus on:
     - How to adjust the reward logic or weight?
     - Do we need to redesign rewards to better balance action use or strategy learning?

---

Please answer the above questions in detail and provide specific improvement suggestions.
    """
    return prompt.strip()




