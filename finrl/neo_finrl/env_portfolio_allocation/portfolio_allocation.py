import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
import decimal  

matplotlib.use("Agg")


class StockPortfolioEnv(gym.Env):


    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        min_trans_amount=[],
        day=0
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.cost = 0
        self.trades = 0
        self.weights = [[1] + [0] * self.stock_dim]
        self.min_trans_amount = min_trans_amount

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stock_dim + len(self.tech_indicator_list), self.stock_dim),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[self.initial_amount] + [0] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.asset_memory)
            df.columns = ["account_value"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.close()

            plt.plot(self.asset_memory, "r")
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.asset_memory[-1]))

            df_account_value = pd.DataFrame(self.asset_memory)
            df_account_value.columns = ["account_value"]
            if df_account_value["account_value"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_account_value["account_value"].mean()
                    / df_account_value["account_value"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)

            asset_distribution = self.asset_memory[-1] * weights
            close_values = np.append([1], self.data.close.values.tolist())
            target_holding = np.divide(asset_distribution, close_values)
            for i in range(self.stock_dim):
                target_holding[i + 1] = self.round_down(target_holding[i + 1], self.min_trans_amount[i])
            trading = np.array(self.actions_memory[-1]) - target_holding
            trading = trading[1:]
            self.trades += len(trading[trading != 0]) 

            argsort_actions = np.argsort(trading)

            sell_index = argsort_actions[: np.where(trading < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(trading > 0)[0].shape[0]]

            trans_cost = 0

            for index in sell_index:
                trans_cost += close_values[index + 1] * trading[index] * self.transaction_cost_pct * -1

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                trans_cost += close_values[index + 1] * trading[index] * self.transaction_cost_pct   
            
            target_holding[0] -= trans_cost
            self.cost += trans_cost

            if target_holding[0] < 0:
                penalty = target_holding[0]
                target_holding[0] = 0   
            else:
                penalty = 0    

             # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )

            close_values = np.append([1], self.data.close.values.tolist())
            self.reward = np.sum(np.multiply(target_holding, close_values)) - self.asset_memory[-1] + penalty
            self.asset_memory.append(np.sum(np.multiply(target_holding, close_values)))

            self.actions_memory.append(target_holding)
            self.weights.append(weights)
           
            self.date_memory.append(self.data.date.unique()[0])
            # self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            # self.reward = new_portfolio_value
            # print("Step reward: ", self.reward)
            self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.weights = [[1] + [0] * self.stock_dim]
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.portfolio_value = self.initial_amount
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[self.initial_amount] + [0] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        weights_list = self.weights
        # action_list = self.weights
        df_actions = pd.DataFrame(np.append(action_list, weights_list, axis=1))
        df_actions.columns = ['cash'] + self.data.tic.values.tolist() + ['cashh', 'ada', 'algo', 'btc', 'eth']
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def round_down(self, value, decimals):
        with decimal.localcontext() as ctx:
            d = decimal.Decimal(value)
            ctx.rounding = decimal.ROUND_DOWN
            return round(d, decimals)