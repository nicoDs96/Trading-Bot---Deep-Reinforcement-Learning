from prettytable import PrettyTable as PrettyTable
from time import time
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os

from src.Agent import Agent
from src.Environment import Environment
from src.utils import (
    load_data,
    print_stats,
    plot_multiple_conf_interval,
    plot_conf_interval,
)
from dotenv import load_dotenv

from tensorboardX import SummaryWriter


# to yaml / .env config [w4]
docker = os.getenv("IN_DOCKER" or None)
if docker is None:
    load_dotenv()

TENSORBOARD_LOGS_DIR = os.getenv("TENSORBOARD_LOGS")
SAVED_MODEL_FILEPATH = os.getenv("TORCH_MODEL_FILEPATH")
TRAIN_DATA_FILEPATH = os.getenv("TRAIN_FILEPATH")

TRADING_PERIOD = 4000
TEST_SIMULATIONS = 3
TRAIN_EPOCHS = 40


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
        self.writer = SummaryWriter(
            log_dir=TENSORBOARD_LOGS_DIR
        )  # You can customize the log directory

    def init_data(self) -> pd.DataFrame:
        # it can be replace for train data loader from open api or ccxt
        df = load_data(TRAIN_DATA_FILEPATH)
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

    def init_model(self) -> None:
        model_filepath = SAVED_MODEL_FILEPATH  # os.path.join(models_path, "profit_reward_double_dqn_model")
        print(f"search model in {model_filepath}")
        Train = not os.path.isfile(path=model_filepath)
        print("pretrainded model not exist:", Train)

        # For not ready model
        if Train:
            self.profit_train_env = Environment(
                self.df[self.index : self.index + self.train_size], "profit"
            )
            self.double_dqn_agent_test = self.double_dqn_agent.train(
                env=self.profit_train_env,
                path=model_filepath,
                num_episodes=TRAIN_EPOCHS,
            )
            # TODO: may be next time we can store images into tensorboard

        # For ready model
        else:
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
            # self.profit_ddqn_return.append(self.profit_test_env.cumulative_return)

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
            while i < TEST_SIMULATIONS:
                # When we retry? # TODO: fix random to window...
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
                    path=SAVED_MODEL_FILEPATH,
                )

                # Comulative return for parallel bots
                self.profit_ddqn_return.append(profit_test_env.cumulative_return)

                avg_mean = sum(self.profit_ddqn_return[i]) / len(
                    self.profit_ddqn_return[i]
                )
                print(f"Reward for {i}", avg_mean)
                profit_test_env.reset()
                i += 1

                self.writer.add_scalar("Agent Test End", avg_mean, i)

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
            to_tensorboard = print_stats("ProfitDDQN", self.profit_ddqn_return, t)
            print(t)

            mean1 = to_tensorboard.get("mean")
            max1 = to_tensorboard.get("max")
            min1 = to_tensorboard.get("min")
            std1 = to_tensorboard.get("std")
            timestamp_now = datetime.now().timestamp()

            self.writer.add_scalar("Avg. Return (%)", mean1, timestamp_now)
            self.writer.add_scalar("Max Return (%)", max1, timestamp_now)
            self.writer.add_scalar("Min Return (%)", min1, timestamp_now)
            self.writer.add_scalar("Std. Dev", std1, timestamp_now)
            # plot_multiple_conf_interval()
            # os.remove(MODEL_FILEPATH)
            # while os.path.isfile(path=SAVED_MODEL_FILEPATH):
            #    pass
            input()


if __name__ == "__main__":
    agent_predictions = RlPredictor()
    # we can use gpu for train?
    # and use predictions for cpu?
    agent_predictions.loop()
