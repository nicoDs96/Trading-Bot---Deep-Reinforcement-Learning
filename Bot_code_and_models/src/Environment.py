import torch
import numpy as np


# TODO: modify the reward st. we can choose between sharpe ratio reward or profit
# reward as shown in the paper.
class Environment:
    def __init__(self, data, reward):
        """
        Creates the environment. Note: Before using the environment you must call
        the Environment.reset() method.

        Args:
           data (:obj:`pd.DataFrane`): Time serie to be initialize the environment.
           reward (:obj:`str`): Type of reward function to use, either sharpe ratio
              "sr" or profit function "profit"
        """
        self.data = data
        self.reward_f = reward if reward == "sr" else "profit"
        self.reset()

    def reset(self):
        """
        Reset the environment or makes a further step of initialization if called
        on an environment never used before. It must always be called before .step()
        method to avoid errors.
        """
        self.t = 23
        self.done = False
        self.profits = [0 for e in range(len(self.data))]
        self.agent_positions = []
        self.agent_open_position_value = 0

        self.cumulative_return = [0 for e in range(len(self.data))]
        self.init_price = self.data.iloc[0, :]["Close"]

    def get_state(self):
        """
        Return the current state of the environment. NOTE: if called after
        Environment.step() it will return the next state.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.done:
            # print(type(self.data))
            # to_tensors_values = [el for el in self.data.iloc[self.t - 23 : self.t + 1, :][["Close", "Target"]].values]
            to_tensors_values = [el for el in self.data.iloc[self.t - 23 : self.t + 1, :]["Close"]]
            t1 = torch.tensor(
                to_tensors_values,
                device=device,
                dtype=torch.float,
            )
            # columns = ["Close", "Target", "SMA_13"]
            
            # columns = ["Close"]
            # to_tensors_values2 = [el for el in self.data.iloc[self.t - 23 : self.t + 1, :][columns].values]
            # t2 = torch.tensor(
            #     np.array(to_tensors_values2),  # Преобразуйте список NumPy.ndarray в один массив NumPy
            #     device=device,
            #     dtype=torch.float,
            # )
            # print(t1.shape, t2.shape)
            return t1

        else:
            return None

    def step(self, act):
        """
        Perform the action of the Agent on the environment, computes the reward
        and update some datastructures to keep track of some econometric indexes
        during time.

        Args:
           act (:obj:`int`): Action to be performed on the environment.

        Returns:
            reward (:obj:`torch.tensor` :dtype:`torch.float`): the reward of
                performing the action on the current env state.
            self.done (:obj:`bool`): A boolean flag telling if we are in a final
                state
            current_state (:obj:`torch.tensor` :dtype:`torch.float`):
                the state of the environment after the action execution.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        reward = 0
        # GET CURRENT STATE
        state = self.data.iloc[self.t, :]["Close"]

        # EXECUTE THE ACTION (act = 0: stay, 1: buy, 2: sell)
        if act == 0:  # Do Nothing
            pass

        # BUY
        if act == 1:
            # TODO: integration
            self.agent_positions.append(self.data.iloc[self.t, :]["Close"])

        # SELL
        sell_nothing = False
        if act == 2:
            # TODO: integration
            profits = 0
            if len(self.agent_positions) < 1:
                sell_nothing = True
            for position in self.agent_positions:
                profits += (
                    self.data.iloc[self.t, :]["Close"] - position
                )  # profit = close - my_position for each my_position "p"

            self.profits[self.t] = profits
            self.agent_positions = []
            # reward += profits

        self.agent_open_position_value = 0
        for position in self.agent_positions:
            self.agent_open_position_value += (
                self.data.iloc[self.t, :]["Close"] - position
            )
            # TO CHECK if the calculus is correct according to the definition
            self.cumulative_return[self.t] += (
                position - self.init_price
            ) / self.init_price

        # COLLECT THE REWARD
        if self.reward_f == "profit":
            p = self.profits[self.t]
            if p > 0:
                reward = 1
            elif p < 0:
                reward = -1
            elif p == 0:
                reward = 0

        if sell_nothing and (reward > -5):
            reward = -5

        # UPDATE THE STATE
        self.t += 1  # self.t - tick

        if self.t == len(self.data) - 1:
            self.done = True

        return (
            torch.tensor([reward], device=device, dtype=torch.float),
            self.done,
            torch.tensor([state], dtype=torch.float),
        )  # reward, done, current_state
