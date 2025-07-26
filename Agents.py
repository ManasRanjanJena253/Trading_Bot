# Importing dependencies
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Q-Network
import torch.nn as nn
import torch

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size: int = 128):
        super(QNetwork, self).__init__()

        # Initial projection layer: From state space to higher dimension
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)

        # Deeper network with more layers and higher capacity
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.bn2 = nn.LayerNorm(hidden_size * 4)

        self.fc4 = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.dropout1 = nn.Dropout(0.2)                            # Regularization and GPU overhead
        self.fc5 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.bn3 = nn.LayerNorm(hidden_size * 2)

        self.fc6 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.fc7 = nn.Linear(hidden_size, action_size)             # Final output layer

        # Activation function
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))                 # Layer 1
        x = self.leaky_relu(self.fc2(x))                           # Layer 2
        x = self.leaky_relu(self.bn2(self.fc3(x)))                 # Layer 3
        x = self.leaky_relu(self.fc4(x))                           # Layer 4
        x = self.dropout1(x)                                       # Regularization
        x = self.leaky_relu(self.bn3(self.fc5(x)))                 # Layer 5
        x = self.leaky_relu(self.fc6(x))                           # Layer 6
        x = self.dropout2(x)                                       # Regularization
        return self.fc7(x)                                         # Final output


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
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = 3e-3)

        # Replay Buffer
        self.memory = deque(maxlen = 200000)

        # Hyperparameter
        self.batch_size = 512
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_every = 100   # Steps to update target network
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
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays first, then to tensors to be more efficient.
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # Current Q-Values
        q_values = self.q_network(states).gather(1, actions)

        # Target Q-Values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
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
