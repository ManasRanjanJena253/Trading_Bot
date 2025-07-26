# Importing dependencies

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Trading_Env import TradingEnv
from Agents import DQNAgent, QNetwork
from tqdm.auto import tqdm

print(torch.cuda.is_available())
torch.backends.cudnn.benchmark = True

# Creating the environment
data = pd.read_csv('Data/processed_reliance_data.csv')
env = TradingEnv(data = data)

# Creating the agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

# Training
num_episodes = 3000
reward_history = []
epsilon_history = []
min_reward = float("-inf")

done = False

for episode in tqdm(range(num_episodes)):
    state, _ = env.reset()
    episode_reward = 0

    done = False
    while not done:
        action = agent.act(state)

        next_state, reward, done, _, _ = env.step(action)

        agent.memorize(state, action, reward, next_state, done)

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
        torch.save(agent.q_network.state_dict(), f"Models/dqn_trading_model_reward_({episode_reward}).pth")
        min_reward = episode_reward

    fig, axs = plt.subplots(1, 2, figsize = (12, 5))

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




