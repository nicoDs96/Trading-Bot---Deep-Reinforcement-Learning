import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../TradingAgent"))
sys.path.append(parent_dir)

import unittest
from src.Agent import Agent
from src.Environment import Environment
from src.Config import *
from src.Database import *
from src.Loader import *
from src.models.dqns import *

from src.utils import (
    load_data,
    load_data_ram,
    print_stats,
    plot_multiple_conf_interval,
    plot_conf_interval,
)


class TestBot(unittest.TestCase):
    def test_load_data(self):
        # Write your test case for the load_data function here
        # df = load_data()
        pass

    def test_load_data_remote(self):
        # Write your test case for the load_data_remote function here
        pass

    def test_get_environment_demo(self):
        # Write your test case for the get_environment function here
        data, last_candle = load_data_ram(days=14)
        env = Environment(data=data, reward="profit", remote=True, days=14)
        env.reset()
        pass

    def test_get_agent(self):
        # Write your test case for the get_agent function here
        agent = Agent()
        pass

    def test_train(self):
        # Write your test case for the train function here
        pass

    def test_test(self):
        # Write your test case for the test function here
        pass

    def test_demo(self):
        # Write your test case for the demo function here
        pass

    def test_print_stats(self):
        pass

    def test_plots(self):
        # plot1 = plot_conf_interval(name="test 1", cum_returns=[0.1, 0.2, 0.3, 0.4, 0.5])
        # plot2 = plot_multiple_conf_interval(
        #     name="test 1", cum_returns=[0.1, 0.2, 0.3, 0.4, 0.5]
        # )
        pass

    def test_print_stats(self):
        from prettytable import PrettyTable as PrettyTable

        table = PrettyTable()
        stats = print_stats("test", [1, 2, 3, 4, 5], table)
        print(stats)


if __name__ == "__main__":
    unittest.main()
