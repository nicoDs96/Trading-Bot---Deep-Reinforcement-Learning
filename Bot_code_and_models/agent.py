from prettytable import PrettyTable as PrettyTable
from time import time
import pandas as pd
import random
import os

from src.Agent import Agent
from src.Environment import Environment
from src.utils import load_data, print_stats, plot_multiple_conf_interval

# to yaml / .env config [w4]
N_TEST = 10
TRADING_PERIOD = 5000
MODEL_FILEPATH = "/home/alxy/Codes/Trading-Bot---Deep-Reinforcement-Learning/Bot_code_and_models/models/profit_reward_double_ddqn_model"
data_path = "./input/"
models_path = "./models/"
EPOCHS = 10


class RlPredictor:
    def __init__(self) -> None:
        self.double_dqn_agent = None
        self.df = self.init_data()
        self.index = random.randrange(len(self.df) - TRADING_PERIOD - 5)
        self.train_size = int(TRADING_PERIOD * 0.8)
        self.profit_ddqn_return = []
        self.init_agent()
        self.profit_train_env = None
        self.profit_test_env = None

    def init_data(self) -> pd.DataFrame:
        # it can be replace for train data loader from open api or ccxt
        df = load_data(data_path)
        return df

    def init_agent(self) -> Agent:
        # hyperparams
        REPLAY_MEM_SIZE = 10000
        BATCH_SIZE = 40
        GAMMA = 0.98
        EPS_START = 1
        EPS_END = 0.12
        EPS_STEPS = 300
        LEARNING_RATE = 0.001
        INPUT_DIM = 24
        HIDDEN_DIM = 120
        ACTION_NUMBER = 3
        TARGET_UPDATE = 10

        self.double_dqn_agent = Agent(
            REPLAY_MEM_SIZE,
            BATCH_SIZE,
            GAMMA,
            EPS_START,
            EPS_END,
            EPS_STEPS,
            LEARNING_RATE,
            INPUT_DIM,
            HIDDEN_DIM,
            ACTION_NUMBER,
            TARGET_UPDATE,
            MODEL="dqn",
            DOUBLE=True,
        )

    def init_model(self) -> list:
        model_filepath = MODEL_FILEPATH  # os.path.join(models_path, "profit_reward_double_dqn_model")
        print(f"search model in {model_filepath}")
        Train = not os.path.isfile(path=model_filepath)
        print("pretrainded model not exist:", Train)

        # For not ready model
        if Train:
            self.profit_train_env = Environment(
                self.df[self.index : self.index + self.train_size], "profit"
            )
            self.double_dqn_agent_test = self.double_dqn_agent.train(
                env=self.profit_train_env, path=model_filepath, num_episodes=EPOCHS
            )


        self.profit_test_env = Environment(
            self.df[self.index + self.train_size : self.index + TRADING_PERIOD],
            "profit",
        )
        # Profit Double DQN
        self.double_dqn_agent_test, _ = self.double_dqn_agent.test(
            env_test=self.profit_test_env,
            model_name="profit_reward_double_dqn_model",
            path=model_filepath,
        )
        self.profit_ddqn_return.append(self.profit_test_env.cumulative_return)

    def loop(self):
        while True:
            # ENV EVOLUTION HERE
            self.init_model()
            self.profit_ddqn_return = []

            if self.profit_test_env is not None:
                self.profit_test_env.reset()
            if self.profit_train_env is not None:
                self.profit_train_env.reset()
            
            i = 0
            while i < N_TEST:
                # When we retry?
                index = random.randrange(len(self.df) - TRADING_PERIOD - 1)
                print(f"Test nr. {str(i + 1)} for rand seed index: {index}")

                # Test ENV
                profit_test_env = Environment(
                    self.df[index + self.train_size : index + TRADING_PERIOD], "profit"
                )
                # Profit Double DQN
                self.double_dqn_agent_test, _ = self.double_dqn_agent.test(
                    profit_test_env,
                    model_name="profit_reward_double_dqn_model",
                    path=MODEL_FILEPATH,
                )

                # Comulative return for parallel bots
                self.profit_ddqn_return.append(profit_test_env.cumulative_return)
                print(f'Reward for {i}', sum(self.profit_ddqn_return[i]) / len(self.profit_ddqn_return[i]))
                profit_test_env.reset()
                i += 1

            # Reporting
            t = PrettyTable(
                [
                    "Trading System",
                    "Avg. Return (%)",
                    "Max Return (%)",
                    "Min Return (%)",
                    "Std. Dev.",
                ]
            )
            print(i)
            print_stats("ProfitDDQN", self.profit_ddqn_return, t)
            print(t)

            # os.remove(MODEL_FILEPATH)
            # while os.path.isfile(path=MODEL_FILEPATH):
            #    time.sleep(1)
            break


if __name__ == "__main__":
    agent_predictions = RlPredictor()
    # we can use model for train?
    # and use predictions for cpu?
    agent_predictions.loop()
