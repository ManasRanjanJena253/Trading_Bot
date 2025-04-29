# Features contained inside our environment
# State = Current stock technical indicators + price movement
# Action = Buy / Sell / Hold
# Reward = Profit / Loss after an action
# Episode = One full trading day / period
# Goal = Maximize total reward (i.e maximize profit)

# Importing dependencies
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance : int = 5000, stop_loss_pct : float = 0.01, take_profit_pct : float = 0.025):
        super(TradingEnv, self).__init__()
        # super() is used to call the parent class constructor, which properly initialize the gym.Env functionality
        # Without this, our custom environment might not work properly with Gym's functions

        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.lev_balance = 5 * self.current_balance
        self.stop_loss_pct = stop_loss_pct   # 1% stop loss
        self.take_profit_pct = take_profit_pct  # 2.5 % take profit
        self.stocks_held = 0   # The no. of stocks that are held

        self.current_step = 0
        self.position = 0    # +1 for long, -1 for short, 0 for no position
        self.entry_price = 0

        # Action_space : 0 = Hold, 1 = Buy(Long), 2 = Sell (short)
        self.action_space = spaces.Discrete(3)

        # Observation space = : The data plus the current balance
        self.observation_space = spaces.Box(
            low = np.inf,
            high = np.inf,
            shape = (len(self.data.columns) + 1,),
            dtype = np.float32
        )

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        # The reset function brings the environment back to the starting point
        # This is called at the beginning of each new episode

        self.initial_balance = self.current_balance   # Carry forward the balance from previous episode / day
        self.lev_balance = 5 * self.current_balance
        self.current_step = 0
        self.position = 0
        self.entry_price = 0

        return self.get_obs(), {}

    def get_obs(self):
        obs = np.append(self.data.iloc[self.current_step].values, self.initial_balance)
        return obs

    def step(self, action):
        done = False
        reward = 0

        current_price = self.data.iloc[self.current_step]['Close']

        if action == 1 and self.position == 0:    # Open long position
            self.position = 1
            self.entry_price = current_price
            self.stocks_held = self.lev_balance // current_price

        elif action == 2 and self.position == 0: # Open short position
            self.position = -1
            self.entry_price = current_price
            self.stocks_held = self.lev_balance // current_price

        if self.position != 0 :   # Checking if there is an open position
            price_change = (current_price - self.entry_price) / self.entry_price    # Price change percentage. If profit its positive and if loss its negative.

            if self.position == -1:
                price_change = -price_change  # Inverting the position for short positions

            if price_change <= -self.stop_loss_pct :
                # Close Position for loss
                if self.position == -1:
                    loss = self.stocks_held * (self.entry_price - current_price)
                elif self.position == 1:
                    loss = self.stocks_held * (current_price - self.entry_price)
                self.current_balance += loss

                reward = -1

                self.position = 0
                self.entry_price = 0

            elif price_change >= self.take_profit_pct :
                # Close Position for profit
                if self.position == -1:
                    profit = self.stocks_held * (self.entry_price - current_price)
                elif self.position == 1:
                    profit = self.stocks_held * (current_price - self.entry_price)
                self.current_balance += profit

                reward = 1

                self.position = 0
                self.entry_price = 0

        self.current_step += 1

        if self.current_step >= len(self.data) - 1 or self.current_balance <= 0:
            done = True

        return self.get_obs(), reward, done, False, {}

    def render(self):
        print(f"Step : {self.current_step}, Balance : {self.current_balance}, Position : {self.position}")



