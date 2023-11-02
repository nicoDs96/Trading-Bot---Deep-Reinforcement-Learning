import torch
import numpy as np
import datetime
import time
from src.utils import load_data_ram, show_loader, clean_loader, demo_wait_tick

COMMISION = 0.0

# TODO: modify the reward st. we can choose between sharpe ratio reward or profit reward as shown in the paper.
class Environment:
    def __init__(self, data, reward, remote=False):
        """
        Creates the environment. Note: Before using the environment you must call
        the Environment.reset() method.

        Args:
           data (:obj:`pd.DataFrane`): Time serie to be initialize the environment.
           reward (:obj:`str`): Type of reward function to use, either sharpe ratio
              "sr" or profit function "profit"
        """
        self.remote = remote
        self.data = data
        self.reward_f = reward if reward == "sr" else "profit"
        self.demo_last_tick = None
        self.reset()

        self.action_number = 0
        self.demo_iterations = 120
        self.last_price = None

    def reset(self):
        """
        Reset the environment or makes a further step of initialization if called
        on an environment never used before. It must always be called before .step()
        method to avoid errors.
        """
        self.tick = 23
        self.done = False
        self.profits = [0 for e in range(len(self.data))]
        self.agent_positions = []
        self.agent_open_position_value = 0

        self.cumulative_return = [0 for e in range(len(self.data))]
        self.init_price = (
            self.data.iloc[0, :]["Close"]
            if not self.remote
            else self.data.iloc[-1, :]["Close"]
        )

    def get_state(
        self,
    ):
        """
        Return the current state of the environment. NOTE: if called after
        Environment.step() it will return the next state.
        """
        if self.remote:
            self.data, self.demo_last_tick = load_data_ram()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.done:
            if not self.remote:
                to_tensors_values = [
                    el
                    for el in self.data.iloc[self.tick - 23 : self.tick + 1, :]["Close"]
                ]
            else:
                # TODO: double check & fix it well or genetics can be used
                to_tensors_values = [
                    el
                    for el in self.data.iloc[self.tick - 23 : self.tick + 1, :]["Close"]
                ]

            t1 = torch.tensor(
                to_tensors_values,
                device=device,
                dtype=torch.float,
            )
            return t1

        else:
            return None

    def step(self, act, state=None):
        # TODO: this function can work differently, it can open sell orders and open buy orders
        # TODO: and exit with take profit and stop loss
        # TODO: it is strategy, we can change it        
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
        state = self.data.iloc[-1, :]["Close"]
        sell_nothing = False

        def execute_action(act):
            global sell_nothing

            # EXECUTE THE ACTION (act = 0: stay, 1: buy, 2: sell)
            def print_action():
                if act == 0:
                    print("Stay...")
                if act == 1:
                    print("Buy...")
                if act == 2 and len(self.agent_positions) > 0:
                    print("Sell...")
                else:
                    pass

            if act == 0:  # Do Nothing
                pass

            if self.remote:
                print_action()

            # BUY
            if act == 1:
                # TODO: integration
                self.agent_positions.append(state)
                if self.remote:
                    print("Add position: ", state, "at", self.action_number)

            # SELL
            if act == 2:
                # TODO: integration waiting...

                profits = 0
                if len(self.agent_positions) < 1:
                    sell_nothing = True
                for position in self.agent_positions:
                    profits += (
                        state - position
                    )  # profit = close - my_position for each my_position "p"

                if not self.remote:
                    self.profits[self.tick] = profits
                else:
                    self.profits[self.action_number] = profits

                if len(self.agent_positions) > 0:
                    if self.remote:
                        print("Sell position:", state, "at", self.action_number)
                        print("Profits: ", profits)
                    self.agent_positions = []                    
                else:
                    pass
                # reward += profits

        execute_action(act)

        if self.remote:
            # while self.demo_wait_tick(self.demo_last_tick):            
            while demo_wait_tick(self.demo_last_tick):            
                show_loader()
            clean_loader()

        reward = 0
        # GET CURRENT STATE
        if self.remote:
            self.data, last_tick = load_data_ram()
            state = self.data.iloc[-1, :]["Close"]
            self.last_price = state
            print("Updated data for last tick:", last_tick, "last price:", state)
        
        # TRAIN & TEST
        else:
            state = self.data.iloc[self.tick, :]["Close"]

        self.agent_open_position_value = 0
        for position in self.agent_positions:
            self.agent_open_position_value += state - position - COMMISION
            # TO CHECK if the calculus is correct according to the definition
            if self.remote:
                self.cumulative_return[self.action_number] += (
                    position - self.last_price
                ) / self.init_price
            else:
                self.cumulative_return[self.tick] += (
                    position - self.init_price
                ) / self.init_price

        # TODO: it is reward function we need te test it for any case
        # COLLECT THE REWARD
        if self.reward_f == "profit":
            if self.remote:
                p = self.profits[self.action_number]
            else:
                p = self.profits[self.tick]

            if p > 0:
                reward = 1
            elif p < 0:
                reward = -1
            elif p == 0:
                reward = 0

        if sell_nothing and (reward > -5):
            reward = -5

        # TODO: extract it in utils
        if self.remote:
            print('.........')
            print('Profit: ', sum(self.profits))
            print('Value:', self.agent_open_position_value)
            print('.........')

        # UPDATE THE STATE FOR NEXT TICK
        # TODO: solve remote or not attribute for demo
        if not self.remote:
            self.tick += 1  # self.t - tick
            if self.tick == len(self.data) - 1:
                self.done = True
        else:
            self.action_number += 1
            if self.action_number == self.demo_iterations:
                self.done = True
                # todo: check if it is ok for closing all open positions

        return (
            torch.tensor([reward], device=device, dtype=torch.float),
            self.done,
            torch.tensor([state], dtype=torch.float),
        )  # reward, done, current_state
