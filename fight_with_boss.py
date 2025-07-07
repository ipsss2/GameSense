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

action_space = [
    {
        "action_name": "Move Forward",
        "key_combination": ["MOVE_FORWARD"],
        "design_reason": "Used for forward movement, it is one of the most basic displacement operations. Suitable for approaching enemies or adjusting positions."
    },
    {
        "action_name": "Move Backward",
        "key_combination": ["MOVE_BACKWARD"],
        "design_reason": "Used for retreating, it is one of the basic defensive displacement operations, suitable for pulling distance or avoiding attacks."
    },
    {
        "action_name": "Move Left",
        "key_combination": ["MOVE_LEFT"],
        "design_reason": "Move to the left, used to adjust lateral positioning or evade attacks."
    },
    {
        "action_name": "Move Right",
        "key_combination": ["MOVE_RIGHT"],
        "design_reason": "Move to the right, used to adjust lateral positioning or avoid attacks."
    },
    {
        "action_name": "Light Attack",
        "key_combination": ["LIGHT_ATTACK"],
        "design_reason": "Basic attack action, suitable for quickly dealing damage when the enemy is unguarded."
    },
    {
        "action_name": "Heavy Attack",
        "key_combination": ["HEAVY_ATTACK"],
        "design_reason": "Heavy attack, dealing high damage but requiring power, suitable for use when the enemy is controlled or exposed."
    },
    {
        "action_name": "Dodge",
        "key_combination": ["DODGE"],
        "design_reason": "Basic evasion operations used to avoid enemy attacks."
    },

    {
        "action_name": "Drink Health Potion",
        "key_combination": ["DRINK_restore_blood_volume"],
        "design_reason": "Use medication to restore blood volume, suitable for use when blood volume is too low."
    },
    {
        "action_name": "Cast Body Fixing",
        "key_combination": ["Body_Fixing"],
        "design_reason": "Use the technique of immobilizing to keep the enemy in place, suitable for interrupting their movements when they are attacking or moving."
    }
]


action_state_changes = [

    {
        "action_name": "Move Forward",
        "reason": "Basic mobile operations that do not involve resource consumption or attack behavior.",
        "idx": 0

    },
    {
        "action_name": "Move Backward",
        "reason": "Basic mobile operations that do not involve resource consumption or attack behavior.",
        "idx": 1
    },
    {
        "action_name": "Move Left",
        "reason": "Basic mobile operations that do not involve resource consumption or attack behavior.",
        "idx": 2
    },
    {
        "action_name": "Move Right",
        "reason": "Basic mobile operations that do not involve resource consumption or attack behavior.",
        "idx": 3
    },


    {
        "action_name": "Light Attack",
        "reason": "Light attacks can cause damage to bosses, but require a certain amount of physical exertion.",
        "idx": 4
    },
    {
        "action_name": "Heavy Attack",
        "reason": "Heavy attacks require energy accumulation and are easily interrupted, but they deal high damage and are suitable for seeking opportunities to unleash.",
        "idx": 5
    },
    {
        "action_name": "Dodge",
        "reason": "Evasion is used to evade attacks and avoid a decrease in health when successful, but it consumes physical energy.",
        "idx": 6
    },
    {
        "action_name": "Drink Health Potion",
        "reason": "Drinking medication is used to restore blood volume, but it will consume the stock of medication. Suitable for use when blood volume is too low.",
        "idx": 7
    },
    {
        "action_name": "Cast Body Fixing",
        "reason": "To cast a fixed body spell, a mana bar is required, which can control the boss for a period of time and create output opportunities.",
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

