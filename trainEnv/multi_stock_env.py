import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# shares normalization factor
# 20 shares maximum per trade
HMAX_NORMALIZE = 20
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1000000
# total number of stocks in our portfolio
STOCK_DIM = 1
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4


class MultiEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, features, offset=200, day=0, verbose=False):
        self.day = day
        self.df = df[offset:].reset_index(drop=True)
        self.features = features
        self.verbose = verbose

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # Shape = 11: [balance] + [price 1] + [shares 1]
        # + [macd 1] + [macds 1] + [rsi 1] + [ema_50 1] + [ema_20 1] + [atr 1] + [cci 1] + [dx 1]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(11,))  # state space
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]  # grab the data of the initial day
        self.terminal = False
        # initialize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + self.data.close.values.tolist() + [0] * STOCK_DIM
        for f in self.features:
            self.state += self.data[f].values.tolist()

        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action | sell k shares
        # self.state[index + STOCK_DIM + 1]: number of shares of the stock
        if self.state[index + STOCK_DIM + 1] > 0:
            # update balance
            self.state[0] += \
                self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * \
                (1 - TRANSACTION_FEE_PERCENT)

            self.state[index + STOCK_DIM + 1] -= min(abs(action), self.state[index + STOCK_DIM + 1])
            self.cost += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass

    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action | buy k shares
        available_amount = self.state[0] // self.state[index + 1]

        # update balance
        self.state[0] -= self.state[index + 1] * min(available_amount, action) * \
                         (1 + TRANSACTION_FEE_PERCENT)

        self.state[index + STOCK_DIM + 1] += min(available_amount, action)

        self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
        self.trades += 1

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            if self.verbose:
                print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train.csv')
            if self.verbose:
                print("total_cost: ", self.cost)
                print("total_trades: ", self.trades)

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            if self.verbose:
                print("Sharpe: ", sharpe)
                print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('results/account_rewards_train.csv')

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * HMAX_NORMALIZE  # [-HMAX_NORMALIZE, HMAX_NORMALIZE]
            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                    self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :]
            # load next state
            self.state = [self.state[0]] + self.data.close.values.tolist() + \
                         list(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
            for f in self.features:
                self.state += self.data[f].values.tolist()

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            self.asset_memory.append(end_total_asset)

            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        # initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + [self.data.close] + [0] * STOCK_DIM
        for f in self.features:
            self.state += [self.data[f]]
        # iteration += 1
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
