import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from collections import defaultdict


import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from collections import defaultdict


def plot_to_base64(fig):

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    base64_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return base64_str


def get_latest_log_json(directory, num_files=10):

    json_files = glob.glob(os.path.join(directory, "episode_*.json"))
    if not json_files:
        return None


    def extract_episode_number(file_name):
        try:
            base_name = os.path.basename(file_name)  
            number = int(base_name.split("_")[1].split(".")[0]) 
            return number
        except (IndexError, ValueError):
            return -1  

    json_files.sort(key=extract_episode_number, reverse=True)  
    print(json_files)
    return json_files[:num_files]  


def get_nonzero_boss_percentage(directory, target_index=2):

    json_files = glob.glob(os.path.join(directory, "episode_*.json"))

    def extract_episode_number(file_name):
        try:
            base_name = os.path.basename(file_name)
            number = int(base_name.split("_")[1].split(".")[0])
            return number
        except (IndexError, ValueError):
            return -1

    json_files.sort(key=extract_episode_number)  

    results = []
    steps = []
    reward = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            nonzero_count = 0  
            for entry in reversed(data): 
                if "state" in entry and "boss_percentage" in entry["state"]:
                    boss_percentage = entry["state"]["boss_percentage"]
                    step=entry["step"]
                    reward_s=entry["total_reward"]

                    if boss_percentage != 0:
                        nonzero_count += 1  
                        if nonzero_count == target_index:  
                            results.append(boss_percentage)  
                            steps.append(step)
                            reward.append(reward_s)
                            break  # 停止处理当前文件
        except (json.JSONDecodeError, KeyError) as e:
            print(f" {json_file}, error: {e}")
            continue

    return results, steps, reward


def plot_training_trends(steps, reward):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1, color2 = '#2ecc71', '#e74c3c' 

    ax1.plot(range(len(steps)), steps, marker='o', markersize=4,
             color=color1, label='Steps', alpha=0.8)
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Steps', color=color1, fontsize=10)
    ax1.tick_params(axis='y', labelcolor=color1)


    steps_min, steps_max = min(steps), max(steps)
    steps_margin = (steps_max - steps_min) * 0.1
    ax1.set_ylim(steps_min - steps_margin, steps_max + steps_margin)

    ax2 = ax1.twinx()
    ax2.plot(range(len(reward)), reward, marker='s', markersize=4,
             color=color2, label='Reward', alpha=0.8)
    ax2.set_ylabel('Reward', color=color2, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color2)

    reward_min, reward_max = min(reward), max(reward)
    reward_margin = (reward_max - reward_min) * 0.1
    ax2.set_ylim(reward_min - reward_margin, reward_max + reward_margin)


    ax1.grid(True, alpha=0.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Training Progress: Steps and Reward per Episode', pad=20)

    plt.tight_layout()

    plt.show()

    total_steps_reward_trend_base64 = plot_to_base64(fig)
    plt.close(fig)

    return total_steps_reward_trend_base64

def plot_training_blood_reward(bloods, reward):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1, color2 = '#2ecc71', '#e74c3c'  

    ax1.plot(range(len(bloods)), bloods, marker='o', markersize=4,
             color=color1, label='Bloods', alpha=0.8)
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Bloods', color=color1, fontsize=10)
    ax1.tick_params(axis='y', labelcolor=color1)

    steps_min, steps_max = min(bloods), max(bloods)
    steps_margin = (steps_max - steps_min) * 0.1
    ax1.set_ylim(steps_min - steps_margin, steps_max + steps_margin)


    ax2 = ax1.twinx()
    ax2.plot(range(len(reward)), reward, marker='s', markersize=4,
             color=color2, label='Reward', alpha=0.8)
    ax2.set_ylabel('Reward', color=color2, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color2)

    reward_min, reward_max = min(reward), max(reward)
    reward_margin = (reward_max - reward_min) * 0.1
    ax2.set_ylim(reward_min - reward_margin, reward_max + reward_margin)

    ax1.grid(True, alpha=0.2)


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')


    plt.title('Training Progress: Bloods and Reward per Episode', pad=20)


    plt.tight_layout()

    plt.show()


    total_trend_base64 = plot_to_base64(fig)
    plt.close(fig)

    return total_trend_base64


def analyze_training_data(file_paths):

    data = {
        "step": [],
        "blood_percentage": [],
        "boss_percentage": [],
        "mana_percentage": [],
        "stamina_percentage": [],
        "potion_percentage": [],
        "action_name": [],
        "reward": [],
        "total_reward": [],
        "elapsed_time": []
    }
    action_rewards = defaultdict(list)  

    cumulative_step_offset = 0


    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                logs = json.load(f)

            for log in logs:
                data["step"].append(log["step"] + cumulative_step_offset)  
                data["blood_percentage"].append(log["state"]["blood_percentage"])
                data["boss_percentage"].append(log["state"]["boss_percentage"])
                data["mana_percentage"].append(log["state"]["mana_percentage"])
                data["stamina_percentage"].append(log["state"]["stamina_percentage"])
                data["potion_percentage"].append(log["state"]["potion_percentage"])
                data["action_name"].append(log["action"]["action_name"])
                data["reward"].append(log["reward"])
                data["total_reward"].append(log["total_reward"])
                data["elapsed_time"].append(log["elapsed_time"])

                action_rewards[log["action"]["action_name"]].append(log["reward"])

            cumulative_step_offset += logs[-1]["step"] + 1 
        except (json.JSONDecodeError, KeyError) as e:
            print(f" {file_path}, error: {e}")
            continue


    df = pd.DataFrame(data)

    action_counts = {action: len(rewards) for action, rewards in action_rewards.items()}

    action_avg_rewards = {action: sum(rewards) / len(rewards) for action, rewards in action_rewards.items()}
    action_total_rewards = {action: sum(rewards) for action, rewards in action_rewards.items()}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["step"], df["total_reward"], label="Total Reward", color="blue")
    ax.set_title("Total Reward over Time (Latest 10 Files)")
    ax.set_xlabel("Cumulative Step")
    ax.set_ylabel("Total Reward")
    ax.legend()


    total_reward_curve_base64 = plot_to_base64(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["step"], df["blood_percentage"], label="Blood Percentage", color="red")
    ax.plot(df["step"], df["boss_percentage"], label="Boss Percentage", color="green")
    ax.plot(df["step"], df["mana_percentage"], label="Mana Percentage", color="purple")
    ax.plot(df["step"], df["stamina_percentage"], label="Stamina Percentage", color="orange")
    ax.plot(df["step"], df["potion_percentage"], label="Potion Percentage", color="brown")
    ax.set_title("State Variables over Time (Latest 10 Files)")
    ax.set_xlabel("Cumulative Step")
    ax.set_ylabel("Percentage")
    ax.legend()

    state_variables_curve_base64 = plot_to_base64(fig)

 
    statistics = {
        "total_steps": len(df),
        "average_reward": df["reward"].mean(),
        "total_reward": df["total_reward"].iloc[-1] if len(df) > 0 else 0,
        "average_elapsed_time": df["elapsed_time"].mean(),
        "most_frequent_action": max(action_counts, key=action_counts.get) if action_counts else None,
        "action_frequency": action_counts,
        "action_average_rewards": action_avg_rewards,
        "action_total_rewards": action_total_rewards
    }


    visualizations = {
        "total_reward_curve": total_reward_curve_base64,
        "state_variables_curve": state_variables_curve_base64,
    }

    return statistics, visualizations


def ana_all_data(log_directory):

    latest_json = get_latest_log_json(log_directory)
    if latest_json:
        print(f"JSON: {latest_json}")
        stats, visualizations = analyze_training_data(latest_json)
    else:
        print("No JSON！")
        stats, visualizations = {}, {}


    boss_percentage_trend,steps,reward = get_nonzero_boss_percentage(log_directory)


    if boss_percentage_trend:

        boss_percentage_trend_base64 = plot_training_blood_reward(boss_percentage_trend,reward)
        total_steps_reward_trend_base64= plot_training_trends(steps, reward)
    else:
        boss_percentage_trend_base64 = None
        total_steps_reward_trend_base64 = None

    print(stats)
    print(boss_percentage_trend)
    return {
        "latest_file_statistics": stats,
        "latest_file_visualizations": visualizations,
        "boss_percentage_trend": boss_percentage_trend,
        "boss_percentage_trend_visualization": boss_percentage_trend_base64,
        'total_steps_reward_trend_base64':total_steps_reward_trend_base64
    }

