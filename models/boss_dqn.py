import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
import json
from utils.img2resnet import process_img_to_obs
from utils.frame_recorder import ScreenshotManager



class QNetwork(nn.Module):
    def __init__(self, embedding_dim, state_dim, action_dim):
        super(QNetwork, self).__init__()


        self.embedding_fc1 = nn.Linear(embedding_dim, 512) 
        self.embedding_fc2 = nn.Linear(512, 256)


        self.state_fc1 = nn.Linear(state_dim, 64)
        self.state_fc2 = nn.Linear(64, 32)


        self.fc1 = nn.Linear(256 + 32, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, embedding, state_info):

        embedding_out = torch.relu(self.embedding_fc1(embedding))
        embedding_out = torch.relu(self.embedding_fc2(embedding_out))


        state_out = torch.relu(self.state_fc1(state_info))
        state_out = torch.relu(self.state_fc2(state_out))

        combined = torch.cat([embedding_out, state_out], dim=1)


        x = torch.relu(self.fc1(combined))
        return self.fc2(x)



class DoubleDQNAgent:
    def __init__(self, embedding_dim, state_dim, action_dim, lr=0.001, gamma=0.99, tau=0.01, device='cuda'):
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=10000)  
        self.batch_size = 64
        self.lr = lr
        self.device = device  

    
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  


        self.q_network = QNetwork(embedding_dim, state_dim, action_dim).to(self.device)
        self.target_q_network = QNetwork(embedding_dim, state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)


        self.loss_fn = nn.MSELoss()


        self.target_q_network.load_state_dict(self.q_network.state_dict())



    def act(self, embedding, state_info):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)


        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        state_info = torch.FloatTensor(state_info).unsqueeze(0).to(self.device)


        with torch.no_grad():
            q_values = self.q_network(embedding, state_info)
        return q_values.argmax().item()



    def remember(self, embedding, state_info, action, reward, next_embedding, next_state_info, done):

        embedding_cpu = embedding.squeeze(0).cpu().numpy()  
        next_embedding_cpu = next_embedding.squeeze(0).cpu().numpy()
        action = int(action)

        state_info = torch.FloatTensor(state_info).cpu().numpy()
        next_state_info = torch.FloatTensor(next_state_info).cpu().numpy()



        self.memory.append((
            embedding_cpu,
            state_info,  
            action,
            reward,
            next_embedding_cpu,
            next_state_info,
            done
        ))

    def replay(self):

        if len(self.memory) < self.batch_size * 5:  
            return

        batch = random.sample(self.memory, self.batch_size)
        embeddings, state_infos, actions, rewards, next_embeddings, next_state_infos, dones = zip(*batch)

        embeddings = torch.FloatTensor(embeddings).to(self.device)
        state_infos = torch.FloatTensor(state_infos).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_embeddings = torch.FloatTensor(next_embeddings).to(self.device)
        next_state_infos = torch.FloatTensor(next_state_infos).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)  


        current_q_values = self.q_network(embeddings, state_infos).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_q_network(next_embeddings, next_state_infos).detach()
        next_q_actions = torch.argmax(self.q_network(next_embeddings, next_state_infos), dim=1)
        next_q_values = next_q_values.gather(1, next_q_actions.unsqueeze(1)).squeeze(1)

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()


        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.soft_update(self.q_network, self.target_q_network)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)



def save_episode_log(log_dir, episode, episode_logs):

    log_file = os.path.join(log_dir, f"episode_{episode + 1}.json")
    with open(log_file, "w") as f:
        json.dump(episode_logs, f, indent=4)

def save_overall_log(log_dir, logs):

    log_file = os.path.join(log_dir, f"over_all.json")
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)


def check_complex_action(action_list):
    complex_action = []
    for i, action in enumerate(action_list):

        if len(action["key_combination"]) > 1:
            complex_action.append(i)

    return complex_action

import time
from collections import deque
import json
import os
def train(env, agent, resnet_model, reward_function, action_list, device,num_episodes=30, max_steps=2000,log_dir="logs",load=False):
    resnet_model.eval()  


    if load:
        pass

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    boss_final_blood = []
    for episode in range(num_episodes):

        next_state, obs_img = env.restart()


        obs_img_tensor = process_img_to_obs(obs_img)
        input_tensor = obs_img_tensor.to(device)
        with torch.no_grad():
            _, embedding = resnet_model(input_tensor) 
            #print(embedding.shape)
            # print(type(embedding))

        state_info = list(next_state.values()) 
        # print(state_info)
        # s=torch.FloatTensor(state_info)
        # print(s.shape)
        # s_o=s.unsqueeze(0)
        # print(s_o.shape)

        total_reward = 0
        prev_state = next_state

        action_history = deque(maxlen=10)
        episode_start_time = time.time()

        episode_logs = []  

        boss_blood=0
        for step in range(max_steps):

            step_time = time.time()

            action_idx = agent.act(embedding, state_info) 
            action = action_list[action_idx]["key_combination"]


            action_history.append(action_idx)
            next_state, obs_img, done, success = env.step(action)

            obs_img_tensor = process_img_to_obs(obs_img)
            input_tensor = obs_img_tensor.to(device)
            with torch.no_grad():
                _, next_embedding = resnet_model(input_tensor)  


            next_state_info = list(next_state.values())

            # if screenshot_manager.recording:
            #     screenshot_manager.stop_recording()
            #     print(
            #         f"Step {step}: Complex action completed. Duration: {action_end_time - action_start_time:.2f}s"
            #     )
            boss_blood=prev_state['boss_percentage']

            reward = reward_function(prev_state, next_state, action_idx, done, action_history, episode_start_time, step_time,step)
            total_reward += reward


            prev_state = next_state


            episode_logs.append({
                "step": step,
                "state": next_state,
                "action": action_list[action_idx],
                "reward": reward,
                "total_reward": total_reward,
                "elapsed_time": step_time - episode_start_time,
                "action_history": list(action_history),
                'done':done,
                'success': success
            })

            agent.remember(embedding, state_info, action_idx, reward, next_embedding, next_state_info, done)

 
            embedding = next_embedding
            state_info = next_state_info


            if success:
                boss_final_blood.append(next_state['boss_percentage'])
                print(f"Episode {episode + 1}: Success! Total Reward: {total_reward}")
                agent.save("dqn_model.pth")
                return

            if done:
                boss_final_blood.append({'boss_finial_blood':next_state['boss_percentage']})
                print(f"Episode {episode + 1}: Done. Total Reward: {total_reward}. Boss percentage {boss_blood}")
                break

        agent.replay()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        save_episode_log(log_dir, episode, episode_logs)
    save_overall_log(log_dir,boss_final_blood)
    print(boss_final_blood)
    print("Training completed.")