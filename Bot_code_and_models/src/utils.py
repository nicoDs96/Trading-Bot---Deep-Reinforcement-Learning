#  It essentially maps (state, action) pairs to their (next_state, reward) result,
#  with the state being the current stock price
from collections import namedtuple
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def print_stats(model, c_return, t):
    c_return = np.array(c_return).flatten()
    t.add_row(
        [
            str(model),
            "%.2f" % np.mean(c_return),
            "%.2f" % np.amax(c_return),
            "%.2f" % np.amin(c_return),
            "%.2f" % np.std(c_return),
            # TODO: more information
        ]
    )

    return {
        "mean": np.mean(c_return),
        "max": np.amax(c_return),
        "min": np.amin(c_return),
        "std": np.std(c_return),
        # TODO: more information
    }


def plot_conf_interval(name, cum_returns):
    """NB. cum_returns must be 2-dim"""
    # Mean
    M = np.mean(np.array(cum_returns), axis=0)
    # std dev
    S = np.std(np.array(cum_returns), axis=0)
    # upper and lower limit of confidence intervals
    LL = M - 0.95 * S
    UL = M + 0.95 * S

    plt.figure(figsize=(20, 5))
    plt.xlabel("Trading Instant (h)")
    plt.ylabel(name)
    plt.legend(["Cumulative Averadge Return (%)"], loc="upper left")
    plt.grid(True)
    plt.ylim(-5, 15)
    plt.plot(range(len(M)), M, linewidth=2)  # mean curve.
    plt.fill_between(range(len(M)), LL, UL, color="b", alpha=0.2)  # std curves.
    plt.show()

    plt.savefig()


def plot_multiple_conf_interval(names, cum_returns_list):
    """NB. cum_returns[i] must be 2-dim"""
    i = 1

    for cr in cum_returns_list:
        plt.subplot(len(cum_returns_list), 2, i)
        # Mean
        M = np.mean(np.array(cr), axis=0)
        # std dev
        S = np.std(np.array(cr), axis=0)
        # upper and lower limit of confidence intervals
        LL = M - 0.95 * S
        UL = M + 0.95 * S

        plt.xlabel("Trading Instant (h)")
        plt.ylabel(names[i - 1])
        plt.title("Cumulative Averadge Return (%)")
        plt.grid(True)
        plt.plot(range(len(M)), M, linewidth=2)  # mean curve.
        plt.fill_between(range(len(M)), LL, UL, color="b", alpha=0.2)  # std curves.
        i += 1

    plt.show()


def split_raw_to_timerange():
    # TODO: it can be data kitchen
    if False:
        df = pd.read_csv(path + "one_minute.csv")
        dfs_to_concat = []
        for count in range(0, len(df) - 300, 300):
            five_minute_interval = df.iloc[count : count + 300]
            aggregated_data = pd.DataFrame(
                {
                    "Open": [five_minute_interval["Open"].iloc[0]],
                    "High": [five_minute_interval["High"].max()],
                    "Low": [five_minute_interval["Low"].min()],
                    "Close": [five_minute_interval["Close"].iloc[-1]],
                }
            )
            dfs_to_concat.append(aggregated_data)
        df_five_minute_aggregated = pd.concat(dfs_to_concat, ignore_index=True)
        df_five_minute_aggregated.to_csv(
            path + "five_minute_aggregated_dataset.csv", index=False
        )
        df = df_five_minute_aggregated
        del df_five_minute_aggregated
        print(f"Shape for range {timerange} of aggregated dataset:", df.shape)
    if False:
        df = pd.read_csv(path + "one_minute.csv")
        dfs_to_concat = []

        for count in range(0, len(df) - 60, 60):
            hour_interval = df.iloc[count : count + 60]
            aggregated_data = pd.DataFrame(
                {
                    "Open": [hour_interval["Open"].iloc[0]],
                    "High": [hour_interval["High"].max()],
                    "Low": [hour_interval["Low"].min()],
                    "Close": [hour_interval["Close"].iloc[-1]],
                }
            )
            dfs_to_concat.append(aggregated_data)

        df_hourly_aggregated = pd.concat(dfs_to_concat, ignore_index=True)
        df_hourly_aggregated.to_csv(path + "hourly_aggregated_dataset.csv", index=False)
        df = df_hourly_aggregated
        del df_hourly_aggregated
        print(f"Shape for range {timerange} of aggregated dataset:", df.shape)
    raise Exception("NOT IMPLEMENTED YET!")


def load_data(filepath=None, timerange=None):
    print(f"Processing `minutes?` to {timerange} from {filepath}")
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath)
        print("Shape of aggregated dataset:", df.shape)
    else:
        raise Exception("WE NEED FILE!!!!")

    return df
