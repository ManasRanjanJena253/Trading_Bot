# Importing dependencies
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size : int = 64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features = state_size,
                             out_features = hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(in_features = hidden_size,
                             out_features = 128)
        self.fc3 = nn.Linear(in_features = 128,
                             out_features = 256)
        self.bn2 = nn.BatchNorm1d(num_features = 256)
        self.fc4 = nn.Linear(in_features = 256,
                             out_features = hidden_size)
        self.fc5 = nn.Linear(in_features = hidden_size,
                             out_features = action_size)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.leaky_relu(self.fc3(self.fc2(x)))
        x = self.leaky_relu(self.fc4(self.bn2(x)))
        return self.fc5(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Q-Network and Target Network
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = 1e-3)

        # Replay Buffer
        self.memory = deque(maxlen = 100000)

        # Hyperparameter
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.update_every = 500   # Steps to update target network
        self.step_count = 0  # For updating target network

    def act(self, state):
        # Epsilon greedy selection
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))   # random action
        else :
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                q_values = self.q_network(state)
            return q_values.argmax().item()   # Best action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:   # Checking whether the memory is more than the batch_size, to enable stable learning.
            return                               # If the memory is really less than learning is skipped to lessen the noise.

        batch = random.sample(self.memory, self.batch_size)   # Randomly sampling a batch of memories from the memory to break correlation between consecutive experiences.
        states, actions, rewards, next_states, dones = zip(*batch)

        # Converting all the parameters into torch tensors.
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-Values
        q_values = self.q_network(states).gather(1, actions)

        # Target Q-Values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)   # Picks the maximum Q-value among all actions for each next state .
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
