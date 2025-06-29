#reward设计与优化
#首先根据要求与暴露出来的状态，reward
#回顾历史约束reward
from utils.ask_model import MultiTurnChatService
import ast
import json

def validate_code_syntax(code: str):
    """
    校验代码的语法是否正确。
    :param code: 待校验的 Python 代码字符串
    :return: 是否合法（True/False），以及错误信息（如果有）
    """
    try:
        ast.parse(code)  # 尝试解析代码为 AST
        return True, None
    except SyntaxError as e:
        return False, str(e)

'''
---

### 任务描述

我正在设计一个强化学习（RL）环境，目标是训练 AI 在与 Boss 战斗的场景中学习最佳策略。AI 的行为由动作空间控制，每个动作会对环境状态产生特定的变化。为了引导 AI 学习，需要设计一个 **reward 函数**，基于动作的预期效果和状态变化给予奖励或惩罚。

---

### 环境信息

#### 状态格式说明

`prev_state` 和 `next_state` 的格式如下：

```python
{
    "blood_percentage": <float>,   # 玩家的血量百分比，范围 [0, 1]
    "boss_percentage": <float>,   # Boss 的血量百分比，范围 [0, 1]
    "mana_percentage": <float>,   # 玩家的法力百分比，范围 [0, 1]
    "stamina_percentage": <float>,# 玩家的体力百分比，范围 [0, 1]
    "potion_percentage": <float>  # 玩家的药物剩余量百分比，范围 [0, 1]
}
```

- **`prev_state`**: 执行动作前的状态。
- **`next_state`**: 执行动作后的状态。

#### 动作索引说明

`action_idx` 是动作空间中动作的索引（`int` 类型）。例如，`action_idx=0` 对应动作 `Move Forward`。


---

### 动作空间及预期效果

以下是完整的动作空间和对应的状态预期变化：

action_state_changes = [
    # 基础移动操作
    {
        "action_name": "Move Forward",
        "state_changes": {
            "血量条": "无变化",
            "法力条": "无变化",
            "体力条": "无变化",
            "药物剩余量": "无变化",
            "Boss血量": "无变化"
        },
        "reason": "基础移动操作，不涉及资源消耗或攻击行为。"
    },
    {
        "action_name": "Move Backward",
        "state_changes": {
            "血量条": "无变化",
            "法力条": "无变化",
            "体力条": "无变化",
            "药物剩余量": "无变化",
            "Boss血量": "无变化"
        },
        "reason": "基础移动操作，不涉及资源消耗或攻击行为。"
    },
    {
        "action_name": "Move Left",
        "state_changes": {
            "血量条": "无变化",
            "法力条": "无变化",
            "体力条": "无变化",
            "药物剩余量": "无变化",
            "Boss血量": "无变化"
        },
        "reason": "基础移动操作，不涉及资源消耗或攻击行为。"
    },
    {
        "action_name": "Move Right",
        "state_changes": {
            "血量条": "无变化",
            "法力条": "无变化",
            "体力条": "无变化",
            "药物剩余量": "无变化",
            "Boss血量": "无变化"
        },
        "reason": "基础移动操作，不涉及资源消耗或攻击行为。"
    },

    # 基础战斗动作
    {
        "action_name": "Light Attack",
        "state_changes": {
            "血量条": "无变化",
            "法力条": "无变化",
            "体力条": "减少（消耗体力）",
            "药物剩余量": "无变化",
            "Boss血量": "减少（造成轻微伤害）"
        },
        "reason": "轻攻击会对 Boss 造成伤害，但需要消耗一定体力。"
    },
    {
        "action_name": "Heavy Attack",
        "state_changes": {
            "血量条": "可能减少（容易被打断）",
            "法力条": "无变化",
            "体力条": "减少（消耗较多体力）",
            "药物剩余量": "无变化",
            "Boss血量": "减少（造成较高伤害）"
        },
        "reason": "重攻击需要蓄力，容易被打断，但造成的伤害较高，适合寻找机会施放。"
    },

    # 闪避操作
    {
        "action_name": "Dodge",
        "state_changes": {
            "血量条": "无变化（成功躲避）/可能减少（未成功）",
            "法力条": "无变化",
            "体力条": "减少（消耗体力）",
            "药物剩余量": "无变化",
            "Boss血量": "无变化"
        },
        "reason": "闪避用于躲避攻击，成功时避免血量减少，但会消耗体力。"
    },
    {
        "action_name": "Forward Dodge",
        "state_changes": {
            "血量条": "无变化（成功躲避）/可能减少（未成功）",
            "法力条": "无变化",
            "体力条": "减少（消耗体力）",
            "药物剩余量": "无变化",
            "Boss血量": "无变化"
        },
        "reason": "向前闪避用于快速接近敌人，适合进攻时调整位置。"
    },
    {
        "action_name": "Backward Dodge",
        "state_changes": {
            "血量条": "无变化（成功躲避）/可能减少（未成功）",
            "法力条": "无变化",
            "体力条": "减少（消耗体力）",
            "药物剩余量": "无变化",
            "Boss血量": "无变化"
        },
        "reason": "向后闪避用于拉开距离，适合防守时调整位置。"
    },
    {
        "action_name": "Left Dodge",
        "state_changes": {
            "血量条": "无变化（成功躲避）/可能减少（未成功）",
            "法力条": "无变化",
            "体力条": "减少（消耗体力）",
            "药物剩余量": "无变化",
            "Boss血量": "无变化"
        },
        "reason": "向左闪避用于调整横向站位，适应敌人攻击方向。"
    },
    {
        "action_name": "Right Dodge",
        "state_changes": {
            "血量条": "无变化（成功躲避）/可能减少（未成功）",
            "法力条": "无变化",
            "体力条": "减少（消耗体力）",
            "药物剩余量": "无变化",
            "Boss血量": "无变化"
        },
        "reason": "向右闪避用于调整横向站位，适应敌人攻击方向。"
    },

    # 特殊技能
    {
        "action_name": "Drink Health Potion",
        "state_changes": {
            "血量条": "增加（回复血量）",
            "法力条": "无变化",
            "体力条": "无变化",
            "药物剩余量": "减少（消耗药物）",
            "Boss血量": "无变化"
        },
        "reason": "喝药用于回复血量，但会消耗药物存量。适合在血量过低时使用。"
    },
    {
        "action_name": "Cast Body Fixing",
        "state_changes": {
            "血量条": "无变化",
            "法力条": "减少（消耗法力）",
            "体力条": "无变化",
            "药物剩余量": "无变化",
            "Boss血量": "无变化（Boss被控制）"
        },
        "reason": "施放定身术需要法力条，能控制 Boss 一段时间，创造输出机会。"
    },

    # 组合攻击动作
    {
        "action_name": "Dodge Attack",
        "state_changes": {
            "血量条": "无变化（成功躲避）/可能减少（未成功）",
            "法力条": "无变化",
            "体力条": "减少（消耗体力）",
            "药物剩余量": "无变化",
            "Boss血量": "减少（造成小额伤害）"
        },
        "reason": "在闪避过程中进行轻攻击，适合高机动时的快速反击。"
    },
    {
        "action_name": "Heavy Dodge Attack",
        "state_changes": {
            "血量条": "可能减少（被打断时）",
            "法力条": "无变化",
            "体力条": "减少（消耗较多体力）",
            "药物剩余量": "无变化",
            "Boss血量": "减少（造成较高伤害）"
        },
        "reason": "在闪避后进行重攻击，适合抓住敌人破绽时进行高伤害输出。"
    },

    # 高级动作（需要资源管理）
    {
        "action_name": "Forward Heavy Attack",
        "state_changes": {
            "血量条": "可能减少（容易被打断）",
            "法力条": "无变化",
            "体力条": "减少（消耗较多体力）",
            "药物剩余量": "无变化",
            "Boss血量": "减少（造成较高伤害）"
        },
        "reason": "向前移动并蓄力攻击，适合在进攻时快速接近敌人并造成高伤害。"
    },
    {
        "action_name": "Dodge Cast Body Fixing",
        "state_changes": {
            "血量条": "无变化（成功躲避）/可能减少（未成功）",
            "法力条": "减少（消耗法力）",
            "体力条": "减少（消耗体力）",
            "药物剩余量": "无变化",
            "Boss血量": "无变化（Boss被控制）"
        },
        "reason": "在闪避过程中施放定身术，可在规避攻击的同时控制敌人。"
    },
    {
        "action_name": "Drink Health Potion While Dodging",
        "state_changes": {
            "血量条": "增加（回复血量）",
            "法力条": "无变化",
            "体力条": "减少（消耗体力）",
            "药物剩余量": "减少（消耗药物）",
            "Boss血量": "无变化"
        },
        "reason": "在闪避中喝药，适合紧急情况下回血并规避敌人攻击。"
    }
]
```

---

### 任务目标

请基于以下要求设计一个 **reward 函数**：
1. **训练目标**：
   - 训练的最**重要**目标就是让boss血量下降
   - 期望模型能够更快地击败boss（最好在3min内）

2. **奖励逻辑**：
   - 奖励 AI 执行能帮助达成目标（让 Boss血量下降）的动作。
   - 惩罚那些浪费资源（如体力、法力或药物）的无效动作。
   - 根据动作的预期效果和实际状态变化，给予奖励或惩罚。
   - 添加特殊奖励：done的时候根据，boss剩余血量设计特殊较度奖励或者惩罚。

3. **特殊情况**：
   - 游戏结束时（`done=True`），根据训练目标的完成度，细粒度地线性给予不同档次奖励或者惩罚。
   - 训练过程中boss不会被击败且boss血量一定**大于0**。
   - 如果某些动作显著违背预期效果（如在血量已满时喝药），给予负奖励。

3. **映射动作索引**：
   - 使用 `action_idx` 映射到对应的动作名称，并根据动作的预期效果设计奖励逻辑。
   - action_history是一个长度为10的历史action_idx队列，可以考虑合理利用它

4. **状态变量**：
   - 使用 `prev_state` 和 `next_state` 中的变量（如 `blood_percentage`, `boss_percentage`, `mana_percentage`, `stamina_percentage`, `potion_percentage`）判断动作是否符合预期效果。

---

### 输出要求

请基于上述信息生成一个完整的 Python 函数 `reward_function`，格式如下：

```python
def reward = reward_function(prev_state, next_state, action_idx, done, action_history, episode_start_time, step_time):
    # 初始化奖励
    reward = 0.0

    # 游戏结束逻辑
    if done:
        ......
        return reward


    # 动作空间定义
    action_state_changes = [
        {
            "action_name": "Move Forward",
            "state_changes": {
                "血量条": "无变化",
                "法力条": "无变化",
                "体力条": "无变化",
                "药物剩余量": "无变化",
                "Boss血量": "无变化"
            },
            "reason": "基础移动操作，不涉及资源消耗或攻击行为。"
        },
        # 剩余所有动作......
    ]

    # 根据动作索引获取当前动作
    action = action_state_changes[action_idx]

    # 奖励逻辑示例
    if action["action_name"] == "Move Forward":
        # 移动动作无明显资源变化，无奖励或惩罚
        reward += 0

    elif action["action_name"] == "Light Attack":
        if next_state["boss_percentage"] < prev_state["boss_percentage"]:
            reward += 2  # 成功对 Boss 造成伤害
        else:
            reward -= 1  # 攻击未命中

    elif action["action_name"] == "Drink Health Potion":
        if prev_state["blood_percentage"] < next_state["blood_percentage"]:
            reward += 3  # 成功回血
        else:
            reward -= 2  # 喝药无效（如血量已满）

    # 其他动作逻辑...
    return reward
```

---

### 插入说明

1. **动作空间**：在 action_state_changes 中补全你的动作列表，一定要注意action_idx一共9个，一定要对应上，否则程序无法运行。
2. **状态逻辑**：根据预期变化（如血量减少、体力消耗）完善奖励逻辑。
3. **奖励机制**：根据你的目标，调整奖励数值和逻辑；奖励机制一定要贴近于训练目标。

'''


q_net="""
class QNetwork(nn.Module):
    def __init__(self, embedding_dim, state_dim, action_dim, dropout_rate=0.1):
        super(QNetwork, self).__init__()
        # 图片embedding分支
        self.embedding_fc1 = nn.Linear(embedding_dim, 512)
        self.embedding_fc2 = nn.Linear(512, 256)
        # 状态条分支
        self.state_fc1 = nn.Linear(state_dim, 64)
        self.state_fc2 = nn.Linear(64, 64)
        # 动作历史处理分支
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
        self.layer_norm = nn.LayerNorm(64)  # LSTM输出的LayerNorm
        # 融合层
        combined_dim = 256 + 64 + 64
        self.batch_norm = nn.BatchNorm1d(combined_dim)  # 特征拼接后的BatchNorm
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, embedding, state_info, action_history):
        # 处理图片embedding
        embedding_out = torch.relu(self.embedding_fc1(embedding))
        embedding_out = torch.relu(self.embedding_fc2(embedding_out))
        # 处理状态条信息
        state_out = torch.relu(self.state_fc1(state_info))
        state_out = torch.relu(self.state_fc2(state_out))
        # 处理动作历史
        action_embedded = self.action_embedding(action_history)
        action_out, _ = self.action_lstm(action_embedded)
        action_out = action_out[:, -1, :]  # 只使用最后一个时间步的输出
        action_out = self.layer_norm(action_out)  # 对LSTM输出做LayerNorm
        # 融合所有信息
        combined = torch.cat([embedding_out, state_out, action_out], dim=1)
        combined = self.batch_norm(combined)  # 对拼接后的特征做BatchNorm
        x = torch.relu(self.fc1(combined))
        return self.fc2(x)
"""
def design_reward(action_state_changes):
    action_length = len(action_state_changes)
    prompt=[f"""
---

### 任务描述

我正在设计一个强化学习（RL）环境,目标是训练 AI 在与 Boss 战斗的场景中学习最佳策略。为了引导 AI 学习,需要设计一个reward函数系统,包含即时奖励、组合奖励和长期目标奖励。
---
### RL网络信息
该reward是为了适配如下代码展示的qnet组成的double dqn网络的：
{q_net}

---
### 环境信息

#### 状态格式说明

`prev_state` 和 `next_state` 的格式如下：

```python
{{
    "blood_percentage": <float>,   # 玩家的血量百分比，范围 [0, 1]
    "boss_percentage": <float>,   # Boss 的血量百分比，范围 [0, 1]
    "mana_percentage": <float>,   # 玩家的法力百分比，范围 [0, 1]
    "stamina_percentage": <float>,# 玩家的体力百分比，范围 [0, 1]
    "potion_percentage": <float>  # 玩家的药物剩余量百分比，范围 [0, 1]
}}
```

- **`prev_state`**: 执行动作前的状态。
- **`next_state`**: 执行动作后的状态。

#### 动作索引说明

`action_idx` 是动作空间中动作的索引（`int` 类型）。例如，`action_idx=0` 对应动作 `Move Forward`。


#### 时序信息
- `action_history`: 长度为20的历史动作队列
- `episode_start_time`: 回合开始时间
- `step_time`: 当前步骤时间
- `step`: 当前步数

### 动作空间及预期效果

以下是完整的动作空间和对应的状态预期变化：
action_state_changes={action_state_changes}

---

### 任务目标

请基于以下要求设计一个 **reward 函数**：
1. **训练目标**：
   - 训练的最**重要**目标就是让boss血量快速下降


2. **奖励逻辑**：
   - 奖励 AI 执行能帮助达成目标（让 Boss血量下降）的动作，惩罚有害与无效的动作。
   - 奖励设计应该是多种奖励结构：
       - 1.boss血量下降奖励：根据boss血量变化量，给予奖励或惩罚。
       - 2.自身血量变化奖励: 根据自己血量变化量，给予奖励或惩罚。
       - 3.自身血量不变奖励：只针对闪避动作的奖励，谨慎设计，有可能模型只学会了闪避
       - 4.组合奖励：对于动作使用历史，给予奖励或惩罚。

3. **特殊情况**：
   - 游戏结束时（`done=True`），根据训练目标的完成度，细粒度地线性给予不同档次奖励或者惩罚。
   - 训练过程中boss不会被击败且boss血量一定**大于0**。
   - 攻击成功奖励要高一些。

---

### 输出要求

请基于上述信息生成一个完整的 Python 函数 `reward_function`，格式如下：

```python
def reward_function(prev_state, next_state, action_idx, done, action_history, action_state_changes, episode_start_time, step_time, step):
    # 初始化奖励
    reward = 0.0

    # 游戏结束逻辑
    if done:
        # 根据boss血量的减少程度给予奖励
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

    # boss血量下降奖励：根据boss血量变化量，给予奖励或惩罚。
    boss_health_change = prev_state["boss_percentage"] - next_state["boss_percentage"]
    if boss_health_change > 0.0:
        boss_blood_reward=   
        reward += min(boss_blood_reward,20)


    # 自身血量变化奖励: 根据自己血量变化量，给予奖励或惩罚。
    player_health_change = next_state["blood_percentage"] - prev_state["blood_percentage"]
    reward +=   

    # 自身血量不变奖励：只针对闪避动作的奖励，谨慎设计，有可能模型只学会了闪避
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

    # 组合奖励：对于连续或者频繁使用的动作，给予奖励或惩罚。
    def calculate_combo_reward(action_history):
        combo_reward = 0
        # 奖励四连轻攻击
        if action_history[-2:] == [4, 4]:
            combo_reward += 
        # 惩罚过于频繁闪避
        if action_history.count(6) > 10:
            combo_reward -= 
        # 惩罚频繁喝药
        if action_history.count(7) > 4 or action_history[-2:]==[7,7]:
            combo_reward -= 
        return combo_reward

    reward += calculate_combo_reward(action_history)

    return reward
```

---

### 插入说明

1. **动作空间**：如果使用 action_state_changes，一定要注意action_idx一共{action_length}个，且**必须要对应上前面给出的内容**，否则程序无法运行。
2. **奖励机制**：根据你的目标，设计合理的奖励数值和逻辑；奖励机制一定要贴近于训练目标,也要适配于double dqn和给出的qnet的特性。

""",'我是通过api访问你的,所以将你的回答转换为只包含代码与注释的形式，且不包含“```python”等元素，这样可以直接写入py文件运行：']
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
        print('失败')
        code=reward_designer.chat({"role": "user", "content": promptlist[-1]+message})
        code_true,message=validate_code_syntax(code)
    return code



def opt_reward(reward_code, action_space_description,history_info,suggest_pass=None):
    training_goal='''
1. **训练目标**：
   - 训练的最**重要**目标就是让boss血量快速下降

2. **当前的奖励逻辑**：
   - 奖励 AI 执行能帮助达成目标（让 Boss血量下降）的动作，惩罚有害与无效的动作。
   - 奖励设计应该是多种奖励结构：
       - 1.boss血量下降奖励：根据boss血量变化量，给予奖励或惩罚。
       - 2.自身血量变化奖励: 根据自己血量变化量，给予奖励或惩罚。
       - 3.自身血量不变奖励：只针对闪避动作的奖励，谨慎设计，有可能模型只学会了闪避
       - 4.组合奖励：对于动作使用历史，给予奖励或惩罚。

3. **特殊情况**：
   - 游戏结束时（`done=True`），根据训练目标的完成度，细粒度地线性给予不同档次奖励或者惩罚。
   - 训练过程中boss不会被击败且boss血量一定**大于0**。
   - 攻击成功奖励要高一些。
'''
    opt_chat=MultiTurnChatService(system_prompt='你是一位强化学习（Reinforcement Learning, RL）领域的专家，擅长分析智能体训练目标、奖励函数设计以及动作空间与策略的匹配性。你的任务是帮助用户分析强化学习系统中的奖励设计，判断奖励逻辑是否合理，是否与智能体的训练目标一致，以及是否存在奖励设计引发的潜在问题（如策略偏差、奖励稀疏等）。你还需要结合用户提供的训练统计数据和图表，识别可能的问题，并提出具体的改进建议。')
    suggest_prompt=generate_rl_analysis_prompt(training_goal,reward_code,action_space_description,history_info)
    if suggest_pass is not None:
        suggest_prompt= '上一轮给出的Reward设计问题分析及建议如下：\n' + suggest_pass +'\n'+suggest_prompt


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
    请依据这些建议修改reward_function()代码：\n
    {reward_code}
    注意保持reward_function()输入输出一致，这样才能运行
    我是通过api访问你的,所以将你的回答转换为只包含代码与注释的形式，且不包含“```python”等元素，这样可以直接写入py文件运行：
    '''
    code=opt_chat.chat({"role": "user", "content": code_prompt})
    print(code)
    code_true, message = validate_code_syntax(code)
    while code_true is False:
        print('失败')
        code = opt_chat.chat({"role": "user", "content": '不是纯code,重新写一个纯code，不包含“```python”等元素，这样可以直接写入py文件运行'})
        code_true, message = validate_code_syntax(code)
        print(code)
    return code,suggest





def generate_rl_analysis_prompt(training_goal, reward_design_code, action_space_description, history_info):
    """
    生成用于分析 RL 奖励设计问题的 Prompt。

    :param training_goal: str，智能体的训练目标描述。
    :param reward_design_code: str，奖励设计的代码内容（代码字符串）。
    :param action_space: list，动作空间的列表，每个动作用字典描述，例如：
        [{"name": "Attack", "description": "主动攻击敌人，造成伤害但消耗法力值或耐力。"}, ...]
    :param training_statistics: dict，训练数据的统计信息，包含动作频率、平均奖励等，例如：
        {
            "action_frequency": {"Attack": 500, "Defend": 300},
            "action_average_rewards": {"Attack": 0.8, "Defend": 0.5},
            "action_total_rewards": {"Attack": 400.0, "Defend": 150.0}
        }
    :param base64_visualizations: dict，可视化图表的 Base64 编码，例如：
        {
            "action_frequency": "data:image/png;base64,...",
            "action_avg_reward": "data:image/png;base64,...",
            "total_reward_curve": "data:image/png;base64,..."
        }
    :return: str，生成的 Prompt。
    """


    # 统计数据格式化

    training_statistics=history_info["latest_file_statistics"]
    action_frequency = training_statistics.get("action_frequency", {})
    action_avg_rewards = training_statistics.get("action_average_rewards", {})
    action_total_rewards = training_statistics.get("action_total_rewards", {})

    action_frequency_str = json.dumps(action_frequency, indent=4, ensure_ascii=False)
    action_avg_rewards_str = json.dumps(action_avg_rewards, indent=4, ensure_ascii=False)
    action_total_rewards_str = json.dumps(action_total_rewards, indent=4, ensure_ascii=False)


    # 拼接 Prompt
    prompt = f"""
我们正在训练一个强化学习（RL）智能体，目标是让智能体在以下环境中完成任务：

### RL网络信息
该reward是为了适配如下代码展示的qnet组成的double dqn网络的：
{q_net}

### 1. 训练目标
{training_goal}

---

### 2. 奖励设计
以下是我们当前的奖励设计逻辑，代码如下：

```python
{reward_design_code}
```

---

### 3. 动作空间
以下是智能体的动作空间及每个动作的设计意图：
{action_space_description}

---

### 4. 训练数据的统计与可视化
以下是从最后十次训练日志中提取的统计数据和图表：

- **动作频率分布**：
```json
{action_frequency_str}
```

- **每种动作的平均奖励**：
```json
{action_avg_rewards_str}
```

- **每种动作的总体奖励**：
```json
{action_total_rewards_str}
```

- **可视化图表**：
上传的图片分别为，最后十次训练总奖励曲线，训练每一个轮次的总共训练step数量和总体reward的折线图,训练几十个轮次boss最终血量与最终reward变化图片

---
### 5. 分析需求

请根据以上信息进行分析：

1. **奖励设计是否合理**：
   - 奖励逻辑是否与训练目标一致？
   - 是否存在奖励设计引发的潜在冲突？例如：
     - 奖励是否容易导致智能体采取极端行为（如过度闪避、忽视资源管理等）？
     - 奖励是否过于稀疏或缺乏梯度，导致目前的策略网络难以优化？
   - 当前的奖励逻辑是否合理与全面，是否需要调整或者添加奖励逻辑

2. **动作空间与奖励设计的匹配性**：
   - 动作的平均奖励是否反映动作的设计意图？
   - 动作的奖励数值的设计是否合理？
   - 某些动作的总奖励值是否具有偏差（如负值或过低），这可能与奖励设计冲突。

3. **数据与图表分析**：
   - 从训练曲线中，是否可以发现奖励设计问题？
   - 是否有改进奖励设计的建议，以更好地引导智能体学习符合目标的策略？
   - 训练的reward和step的关系是否平衡？
   - 是否能够达到训练的唯一重要目标，让boss血量下降?

4. **改进建议**：
   - 如果奖励设计存在问题，请提供详细的改进建议，重点包括：
     - 如何调整奖励逻辑或权重？
     - 是否需要重新设计奖励，以更好地平衡动作使用或策略学习？

---

请详细回答以上问题，并提供具体的改进建议。
    """
    return prompt.strip()




