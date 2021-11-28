import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class CryptoTradingEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """

    # metadata = {"render.modes": ["human"]}

    # def __init__(
    #     self,
    #     df,
    #     stock_dim,
    #     hmax,
    #     initial_amount,
    #     transaction_cost_pct,
    #     reward_scaling,
    #     state_space,
    #     action_space,
    #     tech_indicator_list,
    #     turbulence_threshold=None,
    #     lookback=252,
    #     day=0,
    # ):
    #     # super(StockEnv, self).__init__()
    #     # money = 10 , scope = 1
    #     self.day = day
    #     self.lookback = lookback
    #     self.df = df
    #     self.stock_dim = stock_dim
    #     self.hmax = hmax
    #     self.initial_amount = initial_amount
    #     self.transaction_cost_pct = transaction_cost_pct
    #     self.reward_scaling = reward_scaling
    #     self.state_space = state_space
    #     self.action_space = action_space
    #     self.tech_indicator_list = tech_indicator_list

    #     # action_space normalization and shape is self.stock_dim
    #     self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
    #     # Shape = (34, 30)
    #     # covariance matrix + technical indicators
    #     self.observation_space = spaces.Box(
    #         low=-np.inf,
    #         high=np.inf,
    #         shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
    #     )

    #     # load data from a pandas dataframe
    #     self.data = self.df.loc[self.day, :]
    #     self.covs = self.data["cov_list"].values[0]
    #     self.state = np.append(
    #         np.array(self.covs),
    #         [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
    #         axis=0,
    #     )
    #     self.terminal = False
    #     self.turbulence_threshold = turbulence_threshold
    #     # initalize state: inital portfolio return + individual stock return + individual weights
    #     self.portfolio_value = self.initial_amount

    #     # memorize portfolio value each step
    #     self.asset_memory = [self.initial_amount]
    #     # memorize portfolio return each step
    #     self.portfolio_return_memory = [0]
    #     self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
    #     self.date_memory = [self.data.date.unique()[0]]

    # def step(self, actions):
    #     # print(self.day)
    #     self.terminal = self.day >= len(self.df.index.unique()) - 1
    #     # print(actions)

    #     if self.terminal:
    #         df = pd.DataFrame(self.portfolio_return_memory)
    #         df.columns = ["daily_return"]
    #         plt.plot(df.daily_return.cumsum(), "r")
    #         plt.savefig("results/cumulative_reward.png")
    #         plt.close()

    #         plt.plot(self.portfolio_return_memory, "r")
    #         plt.savefig("results/rewards.png")
    #         plt.close()

    #         print("=================================")
    #         print("begin_total_asset:{}".format(self.asset_memory[0]))
    #         print("end_total_asset:{}".format(self.portfolio_value))

    #         df_daily_return = pd.DataFrame(self.portfolio_return_memory)
    #         df_daily_return.columns = ["daily_return"]
    #         if df_daily_return["daily_return"].std() != 0:
    #             sharpe = (
    #                 (252 ** 0.5)
    #                 * df_daily_return["daily_return"].mean()
    #                 / df_daily_return["daily_return"].std()
    #             )
    #             print("Sharpe: ", sharpe)
    #         print("=================================")

    #         return self.state, self.reward, self.terminal, {}

    #     else:
    #         # print("Model actions: ",actions)
    #         # actions are the portfolio weight
    #         # normalize to sum of 1
    #         # if (np.array(actions) - np.array(actions).min()).sum() != 0:
    #         #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
    #         # else:
    #         #  norm_actions = actions
    #         weights = self.softmax_normalization(actions)
    #         # print("Normalized actions: ", weights)
    #         self.actions_memory.append(weights)
    #         last_day_memory = self.data

    #         # load next state
    #         self.day += 1
    #         self.data = self.df.loc[self.day, :]
    #         self.covs = self.data["cov_list"].values[0]
    #         self.state = np.append(
    #             np.array(self.covs),
    #             [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
    #             axis=0,
    #         )
    #         # print(self.state)
    #         # calcualte portfolio return
    #         # individual stocks' return * weight
    #         portfolio_return = sum(
    #             ((self.data.close.values / last_day_memory.close.values) - 1) * weights
    #         )
    #         # update portfolio value
    #         new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
    #         self.portfolio_value = new_portfolio_value

    #         # save into memory
    #         self.portfolio_return_memory.append(portfolio_return)
    #         self.date_memory.append(self.data.date.unique()[0])
    #         self.asset_memory.append(new_portfolio_value)

    #         # the reward is the new portfolio value or end portfolo value
    #         self.reward = new_portfolio_value
    #         # print("Step reward: ", self.reward)
    #         # self.reward = self.reward*self.reward_scaling

    #     return self.state, self.reward, self.terminal, {}

    # def reset(self):
    #     self.asset_memory = [self.initial_amount]
    #     self.day = 0
    #     self.data = self.df.loc[self.day, :]
    #     # load states
    #     self.covs = self.data["cov_list"].values[0]
    #     self.state = np.append(
    #         np.array(self.covs),
    #         [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
    #         axis=0,
    #     )
    #     self.portfolio_value = self.initial_amount
    #     # self.cost = 0
    #     # self.trades = 0
    #     self.terminal = False
    #     self.portfolio_return_memory = [0]
    #     self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
    #     self.date_memory = [self.data.date.unique()[0]]
    #     return self.state

    # def render(self, mode="human"):
    #     return self.state

    # def softmax_normalization(self, actions):
    #     numerator = np.exp(actions)
    #     denominator = np.sum(np.exp(actions))
    #     softmax_output = numerator / denominator
    #     return softmax_output

    # def save_asset_memory(self):
    #     date_list = self.date_memory
    #     portfolio_return = self.portfolio_return_memory
    #     # print(len(date_list))
    #     # print(len(asset_list))
    #     df_account_value = pd.DataFrame(
    #         {"date": date_list, "daily_return": portfolio_return}
    #     )
    #     return df_account_value

    # def save_action_memory(self):
    #     # date and close price length must match actions length
    #     date_list = self.date_memory
    #     df_date = pd.DataFrame(date_list)
    #     df_date.columns = ["date"]

    #     action_list = self.actions_memory
    #     df_actions = pd.DataFrame(action_list)
    #     df_actions.columns = self.data.tic.values
    #     df_actions.index = df_date.date
    #     # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
    #     return df_actions

    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    # def get_sb_env(self):
    #     e = DummyVecEnv([lambda: self])
    #     obs = e.reset()
    #     return e, obs

    def __init__(self, df, 
                 time_frequency = 15, start = None, mid1 = 172197, mid2 = 216837, 
                 end = None, initial_amount=1e6, hmax=1e2, 
                 transaction_fee_percent=1e-3,  stock_attributes=None, mode='train',gamma = 0.99, make_plots = False, print_verbosity = 10, 
                turbulence_threshold=None,
                risk_indicator_col='turbulence',
                initial=True,
                previous_state=[],
                iteration=''):
        
        self.day = 0
        self.df = df
        self.stock_dim = self.df.tic.unique().shape[0];
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_fee_percent = transaction_fee_percent
        # self.max_stock = 1 //useless

        if stock_attributes is None:
            self.stock_attributes = list(self.df.columns)
            self.stock_attributes.remove("date")
            self.stock_attributes.remove("tic")
            self.stock_attributes.remove("close")
            # self.stock_attributes = ["high", "turbulence"] #hard code
        else:
            self.stock_attributes = stock_attributes

        self.action_space = spaces.Box(low = -1, high = 1, shape = (self.stock_dim, ))
        self.observation_space = spaces.Box (low=-np.inf, high=np.inf,shape= (1 + self.stock_dim * (2 + len(self.stock_attributes)), ))

        self.data = self.df.loc[self.day,:]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = ""

        self.mode = mode
        self.iteration=iteration
        # initalize state
        self.state = self._initiate_state()
        
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.gamma = gamma
        self.max_step = self.df.date.unique().shape[0];
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        #self.reset()
        self._seed()


        # print("df.columns: ",self.df.columns);
        # print("stock_dim", self.stock_dim)
        # print("max_step", self.max_step)
        # print("stock_attributes", self.stock_attributes)

        # reset
        # self.initial_account__reset = self.initial_account
        # self.account = self.initial_account__reset

        # self.stocks = [0.0]  # multi-stack

        # self.total_asset = self.account + self.day_price[0] * self.stocks
        # self.episode_return = 0.0  
        # self.gamma_return = 0.0
        

        '''env information'''
        self.env_name = 'CryptoTradingEnv' 
        # self.state_dim = 1 + 1 + self.price_ary.shape[1] + self.tech_ary.shape[1]
        # self.action_dim = 1
        # self.if_discrete = False //useless
        # self.target_return = 10 //useless

        


    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index+1]>0: 
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index+self.stock_dim+1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action),self.state[index+self.stock_dim+1])
                    sell_amount = self.state[index+1] * sell_num_shares * (1- self.transaction_fee_percent)
                    #update balance
                    self.state[0] += sell_amount

                    self.state[index+self.stock_dim+1] -= sell_num_shares
                    self.cost +=self.state[index+1] * sell_num_shares * self.transaction_fee_percent
                    self.trades+=1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares
            
        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence>=self.turbulence_threshold:
                if self.state[index+1]>0: 
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions 
                    if self.state[index+self.stock_dim+1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index+self.stock_dim+1]
                        sell_amount = self.state[index+1]*sell_num_shares* (1- self.transaction_fee_percent)
                        #update balance
                        self.state[0] += sell_amount
                        self.state[index+self.stock_dim+1] =0
                        self.cost += self.state[index+1]*sell_num_shares* \
                                    self.transaction_fee_percent
                        self.trades+=1
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
            if self.state[index+1]>0: 
                #Buy only if the price is > 0 (no missing data in this particular date)       
                available_amount = self.state[0] / self.state[index+1]
                # print('available_amount:{}'.format(available_amount))
                
                #update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = self.state[index+1] * buy_num_shares * (1+ self.transaction_fee_percent)
                self.state[0] -= buy_amount

                self.state[index+self.stock_dim+1] += buy_num_shares
                
                self.cost+=self.state[index+1] * buy_num_shares * self.transaction_fee_percent
                self.trades+=1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence< self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory,'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def step(self, actions) -> (np.ndarray, float, bool, None):
        self.terminal = self.day >= self.max_step - 1
        if self.terminal:
            if self.make_plots:
                self._make_plot()            
            end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = self.state[0]+sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))- self.initial_amount 
            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            df_total_value['daily_return']=df_total_value['account_value'].pct_change(1)
            if df_total_value['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ['account_rewards']
            df_rewards['date'] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value['daily_return'].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name!='') and (self.mode!=''):
                df_actions = self.save_action_memory()
                df_actions.to_csv('results/actions_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration))
                df_total_value.to_csv('results/account_value_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                df_rewards.to_csv('results/account_rewards_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.plot(self.asset_memory,'r')
                plt.savefig('results/account_value_{}_{}_{}.png'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.close()

            # Add outputs to logger interface
            #logger.record("environment/portfolio_value", end_total_asset)
            #logger.record("environment/total_reward", tot_reward)
            #logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            #logger.record("environment/total_cost", self.cost)
            #logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}
        else:
            actions = actions * self.hmax
            if self.turbulence_threshold is not None:
                if self.turbulence>=self.turbulence_threshold:
                    actions=np.array([-self.hmax]*self.stock_dim)
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)
            
            #state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day,:]    
            if self.turbulence_threshold is not None:
                self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()
            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            # self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

        # stock_action = action[0] 
        # """buy or sell stock"""
        # adj = self.day_price[0]
        # if stock_action < 0:
        #     stock_action = max(0, min(-1*stock_action, 0.5*self.total_asset/adj + self.stocks))
        #     self.account += adj * stock_action * (1 - self.transaction_fee_percent)
        #     self.stocks -= stock_action
        # elif stock_action > 0: 
        #     max_amount = self.account / adj
        #     stock_action = min(stock_action,max_amount)
        #     self.account -= adj * stock_action * (1 + self.transaction_fee_percent)
        #     self.stocks += stock_action
            
        # """update day"""
        # self.day += 1
        # self.day_price = self.price_ary[self.day]
        # self.day_tech = self.tech_ary[self.day]
        # done = (self.day + 1) == self.max_step  
        # normalized_tech = [self.day_tech[0]*2**-1, self.day_tech[1]*2**-15, self.day_tech[2]*2**-15,
        #                    self.day_tech[3]*2**-6, self.day_tech[4]*2**-6,self.day_tech[5]*2**-15, self.day_tech[6]*2**-15]
        # state = np.hstack((self.account * 2 ** -18, self.day_price * 2 ** -15, normalized_tech, self.stocks * 2 ** -4,)).astype(np.float32)

        # next_total_asset = self.account + self.day_price[0]*self.stocks
        # reward = (next_total_asset - self.total_asset) * 2 ** -16  
        # self.total_asset = next_total_asset

        # self.gamma_return = self.gamma_return * self.gamma + reward 
        # if done:
        #     reward += self.gamma_return
        #     self.gamma_return = 0.0  
        #     self.episode_return = next_total_asset / self.initial_account  
        # return state, reward, done, None

    def render(self, mode='human',close=False):
        print(self.state[0])
        print(self.state[(1 + self.stock_dim): (self.stock_dim*2+1)])

        print(self.state)
        print(self.asset_memory)
        print(self.rewards_memory)

        return self.state

    def reset(self) -> np.ndarray:
        self.state = self._initiate_state()
        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory = [previous_total_asset]

        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        
        self.episode+=1

        return self.state
        # self.day = 0
        # # self.day_price = self.price_ary[self.day]
        # # self.day_tech = self.tech_ary[self.day]
        # self.state = self.initiate_state()
        # self.initial_account__reset = self.initial_account  # reset()
        # # self.account = self.initial_account__reset
        # # self.stocks = 0.0
        # self.state = self._initiate_state()
        # self.total_asset = self.account + self.day_price[0] * self.stocks

        # normalized_tech = [self.day_tech[0]*2**-1, self.day_tech[1]*2**-15, self.day_tech[2]*2**-15,
        #                    self.day_tech[3]*2**-6, self.day_tech[4]*2**-6,self.day_tech[5]*2**-15, self.day_tech[6]*2**-15]
        # state = np.hstack((self.account * 2 ** -18, self.day_price * 2 ** -15, normalized_tech, self.stocks * 2 ** -4,)).astype(np.float32)
        # return state

    def _initiate_state(self):
        if self.initial:
            state = [self.initial_amount] + \
                self.data.close.values.tolist() + \
                [0]*self.stock_dim  + \
                sum([self.data[tech].values.tolist() for tech in self.stock_attributes ], [])
        else:
            state = [self.previous_state[0]] + \
                self.data.close.values.tolist() + \
                self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
                sum([self.data[tech].values.tolist() for tech in self.stock_attributes ], [])
        return state

    def _update_state(self):
        state =  [self.state[0]] + \
                  self.data.close.values.tolist() + \
                  list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                  sum([self.data[tech].values.tolist() for tech in self.stock_attributes ], [])
    
        return state

    def _get_date(self):
        date = self.data.date.unique()[0]
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique())>1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
