from action_manager.model.wukong_trained.ResNet_boss_model import ResNet50_boss

from boss_env import boss_env
from models.new_model import DoubleDQNAgent,train
import torch
from agent.fast_module_trainner_agent.reward import get_org_reward_function,opt_reward
from utils.train_data_analysis import ana_all_data

import inspect

def func_to_str(func):
    return inspect.getsource(func)


def str_to_func(func_str):
    try:

        func_str = func_str.strip()
        if not func_str.startswith('def '):
            raise ValueError("Function string must start with 'def'")

        namespace = {}

        print("Executing function string...")
        exec(func_str, namespace)
        print("Namespace keys:", list(namespace.keys()))


        func_name = func_str[4:func_str.index('(')].strip()
        print(f"Extracted function name: {func_name}")

        if func_name not in namespace:
            raise ValueError(f"Function {func_name} not found in namespace")

        return namespace[func_name]

    except Exception as e:
        print(f"Debug - Function string starting characters: {func_str[:50]}...")
        raise ValueError(f"Cannot convert string to function: {str(e)}")

device = torch.device("cuda")

model_resnet_boss = ResNet50_boss(num_classes=10)
model_resnet_boss.load_state_dict(torch.load('action_manager\\model\\wukong_trained\\boss_model.pkl'))
print('success load')
model_resnet_boss.to(device)
model_resnet_boss.eval()
for param in model_resnet_boss.parameters():
    param.requires_grad = False



def save_string_list_to_file(string_list, file_name):

    with open(file_name, 'w', encoding='utf-8') as file:
        for item in string_list:
            file.write(item + '\n')

action_o_space = [

    {
        "action_name": "Move Forward",
        "key_combination": ["MOVE_FORWARD"],
        "design_reason": "用于向前移动，是最基础的位移操作之一。适合接近敌人或调整站位。"
    },
    {
        "action_name": "Move Backward",
        "key_combination": ["MOVE_BACKWARD"],
        "design_reason": "用于后退，是基础的防守位移操作之一，适合拉开距离或规避攻击。"
    },
    {
        "action_name": "Move Left",
        "key_combination": ["MOVE_LEFT"],
        "design_reason": "向左移动，用于调整横向站位或躲避攻击。"
    },
    {
        "action_name": "Move Right",
        "key_combination": ["MOVE_RIGHT"],
        "design_reason": "向右移动，用于调整横向站位或躲避攻击。"
    },

    {
        "action_name": "Light Attack",
        "key_combination": ["LIGHT_ATTACK"],
        "design_reason": "基础攻击动作，适合在敌人无防备时快速输出伤害。"
    },
    {
        "action_name": "Heavy Attack",
        "key_combination": ["HEAVY_ATTACK"],
        "design_reason": "重攻击，造成较高伤害但需要蓄力，适合在敌人被控制或暴露破绽时使用。"
    },


    {
        "action_name": "Dodge",
        "key_combination": ["DODGE"],
        "design_reason": "基础的闪避操作，用于躲避敌人的攻击。"
    },


    {
        "action_name": "Drink Health Potion",
        "key_combination": ["DRINK_restore_blood_volume"],
        "design_reason": "使用药剂回复血量，适合在血量过低时使用。"
    },
    {
        "action_name": "Cast Body Fixing",
        "key_combination": ["Body_Fixing"],
        "design_reason": "施放定身术，将敌人控制在原地，适合在敌人进行攻击或移动时打断其行动。"
    }

]
action_space = [
    {
        "action_name": "Move Forward",
        "key_combination": ["MOVE_FORWARD"],
        "design_reason": "用于向前移动，是最基础的位移操作之一。适合接近敌人或调整站位。"
    },
    {
        "action_name": "Move Backward",
        "key_combination": ["MOVE_BACKWARD"],
        "design_reason": "用于后退，是基础的防守位移操作之一，适合拉开距离或规避攻击。"
    },
    {
        "action_name": "Move Left",
        "key_combination": ["MOVE_LEFT"],
        "design_reason": "向左移动，用于调整横向站位或躲避攻击。"
    },
    {
        "action_name": "Move Right",
        "key_combination": ["MOVE_RIGHT"],
        "design_reason": "向右移动，用于调整横向站位或躲避攻击。"
    },
    {
        "action_name": "Light Attack",
        "key_combination": ["LIGHT_ATTACK"],
        "design_reason": "基础攻击动作，适合在敌人无防备时快速输出伤害。"
    },
    {
        "action_name": "Heavy Attack",
        "key_combination": ["HEAVY_ATTACK"],
        "design_reason": "重攻击，造成较高伤害但需要蓄力，适合在敌人被控制或暴露破绽时使用。"
    },
    {
        "action_name": "Dodge",
        "key_combination": ["DODGE"],
        "design_reason": "基础的闪避操作，用于躲避敌人的攻击。"
    },

    {
        "action_name": "Drink Health Potion",
        "key_combination": ["DRINK_restore_blood_volume"],
        "design_reason": "使用药剂回复血量，适合在血量过低时使用。"
    },
    {
        "action_name": "Cast Body Fixing",
        "key_combination": ["Body_Fixing"],
        "design_reason": "施放定身术，将敌人控制在原地，适合在敌人进行攻击或移动时打断其行动。"
    }
]


action_state_changes = [

    {
        "action_name": "Move Forward",
        "reason": "基础移动操作，不涉及资源消耗或攻击行为。",
        "idx": 0

    },
    {
        "action_name": "Move Backward",
        "reason": "基础移动操作，不涉及资源消耗或攻击行为。",
        "idx": 1
    },
    {
        "action_name": "Move Left",
        "reason": "基础移动操作，不涉及资源消耗或攻击行为。",
        "idx": 2
    },
    {
        "action_name": "Move Right",
        "reason": "基础移动操作，不涉及资源消耗或攻击行为。",
        "idx": 3
    },


    {
        "action_name": "Light Attack",
        "reason": "轻攻击会对 Boss 造成伤害，但需要消耗一定体力。",
        "idx": 4
    },
    {
        "action_name": "Heavy Attack",
        "reason": "重攻击需要蓄力，容易被打断，但造成的伤害较高，适合寻找机会施放。",
        "idx": 5
    },
    {
        "action_name": "Dodge",
        "reason": "闪避用于躲避攻击，成功时避免血量减少，但会消耗体力。",
        "idx": 6
    },
    {
        "action_name": "Drink Health Potion",
        "reason": "喝药用于回复血量，但会消耗药物存量。适合在血量过低时使用。",
        "idx": 7
    },
    {
        "action_name": "Cast Body Fixing",
        "reason": "施放定身术需要法力条，能控制 Boss 一段时间，创造输出机会。",
        "idx": 8
    }
]

code=get_org_reward_function(action_state_changes)


reward_function=str_to_func(code)

action_dim=len(action_space)

boss_env = boss_env()

agent = DoubleDQNAgent(embedding_dim=256,state_dim=5,action_dim=action_dim,device=device)

suggests=[]
code_history=[]
code_history.append(code)

suggest=None


for i in range(4):
    agent.load("dqn_training_model.pth")
    train(env=boss_env, agent=agent, resnet_model=model_resnet_boss, reward_function=reward_function, action_list=action_space,num_episodes=50, device=device)
    reward_code_str= code
    history_info = ana_all_data('logs')
    code,suggest = opt_reward(reward_code_str,action_state_changes,history_info,suggest)
    code_history.append(code)
    suggests.append(suggest)
    reward_function = str_to_func(code)

save_string_list_to_file(suggests,'suggests.txt')
save_string_list_to_file(code_history,'code_history.txt')

print('final reward code is:' , reward_function)
agent.load("dqn_training_model.pth")
train(env=boss_env, agent=agent, resnet_model=model_resnet_boss, reward_function=reward_function, action_list=action_space,num_episodes=400, device=device)

