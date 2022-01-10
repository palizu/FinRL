import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common import logger


class CryptoTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots=False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        min_buy_amount=[],
        min_transaction_amount=0,
        stoploss_penalty=0.9,
        profit_loss_ratio=2,
        cash_penalty_proportion=0.1,
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.assets = df.tic.unique()
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.stoploss_penalty = stoploss_penalty
        self.profit_loss_ratio = profit_loss_ratio
        self.min_profit_penalty = 1 + profit_loss_ratio * (1 - self.stoploss_penalty)
        self.cash_penalty_proportion = cash_penalty_proportion
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.min_buy_amount= [int(i) for i in min_buy_amount]
        self.min_transaction_amount = min_transaction_amount
        self.peak = self.initial_amount
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    # if self.min_buy_amount != []:
                    #     r_action = int(action / self.min_buy_amount[index])
                    #     r_action = r_action * self.min_buy_amount[index]

                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )

                    if self.min_buy_amount != []:
                        if sell_num_shares < round(sell_num_shares, self.min_buy_amount[index]): 
                            sell_num_shares = round(sell_num_shares, self.min_buy_amount[index]) - 10**(-self.min_buy_amount[index])
                        else:
                            sell_num_shares = round(sell_num_shares, self.min_buy_amount[index])

                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct)
                    )
                    if sell_amount < self.min_transaction_amount: 
                        return 0
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.state[index + self.stock_dim + 1] = round(self.state[index + self.stock_dim + 1], self.min_buy_amount[index])
                    self.cost += (
                        self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0
            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct)
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] / (self.state[index + 1] * (1 + self.buy_cost_pct))
                # if self.min_buy_amount != []:
                #     available_amount = int(available_amount / self.min_buy_amount[index])
                #     available_amounts = available_amount * self.min_buy_amount[index]
                #     r_action = int(action / self.min_buy_amount[index])
                #     r_action = r_action * self.min_buy_amount[index]

                # update balance
                buy_num_shares = min(available_amount, action)
                if self.min_buy_amount != []:
                    if buy_num_shares < round(buy_num_shares, self.min_buy_amount[index]):
                        buy_num_shares = round(buy_num_shares, self.min_buy_amount[index]) - 10**(-self.min_buy_amount[index])
                    else:
                        buy_num_shares = round(buy_num_shares, self.min_buy_amount[index])
                    
                buy_amount = (
                    self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct)
                )
                if buy_amount < self.min_transaction_amount:
                    return 0
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.state[index + self.stock_dim + 1] = round(self.state[index + self.stock_dim + 1], self.min_buy_amount[index])

                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct
                self.trades += 1
            else:
                buy_num_shares = 0
            # print("buy_num_shares ", buy_num_shares)

            return buy_num_shares

        # perform buy action based on the sign of the action
        # print("_buy_stock ", index, " ",action)
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig("results/account_value_trade_{}.png".format(self.episode))
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.initial_amount
            )
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}

        else:
            self.prev_holdings = np.array(self.state[(self.stock_dim) + 1 : (self.stock_dim*2 + 1)])
            closings = np.array(self.state[1 : (self.stock_dim + 1)])
            actions = np.array(actions * self.hmax)  # actions initially is scaled between 0 to 1
            self.reward = self.get_reward()
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            self.begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            self.closing_diff_avg_buy = closings - (self.stoploss_penalty * self.avg_buy_price)
            if self.begin_total_asset >= self.stoploss_penalty * self.peak:
                # clear out position if stop-loss criteria is met
                actions = np.where(
                    self.closing_diff_avg_buy < 0, -np.array(self.prev_holdings), actions
                )

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            # r_actions = actions.copy()

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            holdings = np.array(self.state[(self.stock_dim) + 1 : (self.stock_dim*2 + 1)])

            sells = -np.clip(actions, -np.inf, 0)
            buys = np.clip(actions, 0, np.inf)

            sell_closing_price = np.where(sells > 0, closings, 0) 
            profit_sell = np.where(sell_closing_price > 0, 1, 0)
            self.profit_sell_diff_avg_buy = np.where(
                profit_sell == 1, 
                closings - (self.min_profit_penalty * self.avg_buy_price), 
                0
            )

            buys = np.sign(buys)
            self.n_buys += buys
            self.avg_buy_price = np.where(
                buys > 0,
                self.avg_buy_price + ((closings - self.avg_buy_price) / self.n_buys),
                self.avg_buy_price,
            )

            self.n_buys = np.where(holdings > 0, self.n_buys, 0)
            self.avg_buy_price = np.where(holdings > 0, self.avg_buy_price, 0)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            if self.peak < end_total_asset:
                self.peak = end_total_asset 

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def get_reward(self):
        if self.day == 0:
            return 0
        else:
            total_assets = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            cash = self.state[0]
            holdings = self.state[(self.stock_dim + 1) : (self.stock_dim*2 + 1)]
            neg_closing_diff_avg_buy = np.clip(self.closing_diff_avg_buy, -np.inf, 0)
            neg_profit_sell_diff_avg_buy = np.clip(
                self.profit_sell_diff_avg_buy, -np.inf, 0
            )
            pos_profit_sell_diff_avg_buy = np.clip(
                self.profit_sell_diff_avg_buy, 0, np.inf
            )

            cash_penalty = max(0, (total_assets * self.cash_penalty_proportion - cash))
            if self.day > 1:
                stop_loss_penalty = -1 * np.dot(
                    np.array(self.prev_holdings), neg_closing_diff_avg_buy
                )
            else:
                stop_loss_penalty = 0
            low_profit_penalty = -1 * np.dot(
                np.array(holdings), neg_profit_sell_diff_avg_buy
            )
            total_penalty = stop_loss_penalty + low_profit_penalty 
            # + cash_penalty

            additional_reward = np.dot(np.array(holdings), pos_profit_sell_diff_avg_buy)

            # reward = (
            #     (total_assets - total_penalty + additional_reward) / self.initial_amount
            # ) - 1
            reward = total_assets - total_penalty + additional_reward - self.begin_total_asset

            return reward

    def reset(self):
        # initiate state
        self.state = self._initiate_state() 
        self.closing_diff_avg_buy = np.zeros(len(self.assets))
        self.profit_sell_diff_avg_buy = np.zeros(len(self.assets))
        self.n_buys = np.zeros(len(self.assets))
        self.avg_buy_price = np.zeros(len(self.assets))
        self.prev_holdings = np.zeros(len(self.assets))
        
        if self.initial:
            self.asset_memory = [self.initial_amount]
            self.peak = self.initial_amount
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]
            self.peak = self.previous_state[self.stock_dim * 3 + 1]

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + [0] * self.stock_dim
                    + [0] * self.stock_dim
                    + [self.initial_amount]
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [], 
                    )
                )
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + [0] * self.stock_dim
                    + [self.initial_amount]
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + self.previous_state[
                        (self.stock_dim * 2 + 1) : (self.stock_dim * 3 + 1)
                    ]
                    + [self.previous_state[self.stock_dim * 3 + 1]]
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )
                state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)] = [
                    1-(x < 1e-5) for x in state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                ]
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + self.previous_state[
                        (self.stock_dim * 2 + 1) : (self.stock_dim * 3 + 1)
                    ]
                    + [self.previous_state[self.stock_dim * 3 + 1]]
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + list(self.avg_buy_price)
                + [self.peak]
                + sum(
                    [
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ],
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + list(self.avg_buy_price)
                + [self.peak]
                + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

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
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
