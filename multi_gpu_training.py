# Importing dependencies

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from Trading_Env import TradingEnv
from Agents import DQNAgent, QNetwork
from tqdm.auto import tqdm

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

torch.backends.cudnn.benchmark = True

# Set device and multi-GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_multi_gpu = torch.cuda.device_count() > 1

print(f"Using device: {device}")
print(f"Multi-GPU training: {use_multi_gpu}")

# Creating the environment
data = pd.read_csv('Data/processed_reliance_data.csv')
env = TradingEnv(data=data)

# Creating the agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

# Setup multi-GPU if available
if use_multi_gpu:
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    agent.q_network = nn.DataParallel(agent.q_network)
    agent.target_network = nn.DataParallel(agent.target_network)

    # Update the target network to match the main network
    if use_multi_gpu:
        agent.target_network.load_state_dict(agent.q_network.state_dict())

    # Modify the agent's act method to handle DataParallel
    original_act = agent.act


    def multi_gpu_act(state):
        # Epsilon greedy selection
        if random.random() < agent.epsilon:
            return random.choice(range(agent.action_size))
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.inference_mode():
                q_values = agent.q_network(state)
            return q_values.argmax().item()


    agent.act = multi_gpu_act

    # Modify the learn method to handle multi-GPU training
    original_learn = agent.learn


    def multi_gpu_learn():
        if len(agent.memory) < agent.batch_size:
            return

        batch = random.sample(agent.memory, agent.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays first, then to tensors to be more efficient.
        states = torch.FloatTensor(np.array(states)).to(agent.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(agent.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(agent.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(agent.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(agent.device)

        # Current Q-Values
        q_values = agent.q_network(states).gather(1, actions)

        # Target Q-Values
        with torch.no_grad():
            next_q_values = agent.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (agent.gamma * next_q_values * (1 - dones))

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize
        agent.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability in multi-GPU training
        torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), max_norm=1.0)

        agent.optimizer.step()

        # Update target network
        agent.step_count += 1
        if agent.step_count % agent.update_every == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())


    agent.learn = multi_gpu_learn

# Training
num_episodes = 3000
reward_history = []
epsilon_history = []
min_reward = float("-inf")

done = False

# Enable mixed precision for better GPU utilization
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

for episode in tqdm(range(num_episodes)):
    state, _ = env.reset()
    episode_reward = 0

    done = False
    while not done:
        action = agent.act(state)

        next_state, reward, done, _, _ = env.step(action)

        agent.memorize(state, action, reward, next_state, done)

        # Use mixed precision training if available and beneficial
        if scaler is not None and use_multi_gpu:
            with torch.cuda.amp.autocast():
                agent.learn()
        else:
            agent.learn()

        state = next_state
        episode_reward += reward
        # env.render()  Can be uncommented to view the info about the trades

    # Epsilon decay
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    reward_history.append(episode_reward)
    epsilon_history.append(agent.epsilon)

    # Saving the model every 50 episodes
    if episode_reward > min_reward:
        # Handle DataParallel model saving
        model_state_dict = agent.q_network.module.state_dict() if use_multi_gpu else agent.q_network.state_dict()
        torch.save(model_state_dict, f"Models/dqn_trading_model_reward_({episode_reward}).pth")
        min_reward = episode_reward

    # Plot updates (moved outside of frequent saving to reduce I/O overhead)
    if episode % 10 == 0:  # Update plots every 10 episodes instead of every episode
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plotting Reward History
        axs[0].plot(reward_history)
        axs[0].set_title("Reward Trend over Episodes")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Total Reward")

        # Plotting Epsilon Decay
        axs[1].plot(epsilon_history)
        axs[1].set_title("Epsilon Decay over Episodes")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Epsilon Value")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the combined plot
        plt.savefig("Plots/Training_Metrics.png")
        plt.close()

print("Training completed!")

# Clean up GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()