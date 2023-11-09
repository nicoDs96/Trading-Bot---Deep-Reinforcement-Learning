import os
from pathlib import Path

import pandas as pd

import sys
import ccxt
import csv
from datetime import datetime, timedelta

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(""))))
sys.path.append(root + "/python")

VERBOSE = "minimal"


def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if VERBOSE == "minimal":
            pass
        else:
            print(
                "Fetched",
                len(ohlcv),
                symbol,
                "candles from",
                exchange.iso8601(ohlcv[0][0]),
                "to",
                exchange.iso8601(ohlcv[-1][0]),
            )
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')


def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        fetch_since = earliest_timestamp - timedelta
        ohlcv = retry_fetch_ohlcv(
            exchange, max_retries, symbol, timeframe, fetch_since, limit
        )
        if ohlcv[0][0] >= earliest_timestamp:
            break
        earliest_timestamp = ohlcv[0][0]
        all_ohlcv = ohlcv + all_ohlcv
        print(
            len(all_ohlcv),
            symbol,
            "candles in total from",
            exchange.iso8601(all_ohlcv[0][0]),
            "to",
            exchange.iso8601(all_ohlcv[-1][0]),
        )
        if fetch_since < since:
            break
    return all_ohlcv


def write_to_csv(filename, exchange, data):
    p = Path(".")
    p.mkdir(parents=True, exist_ok=True)
    full_path = p / str(filename)
    with Path(full_path).open("w+", newline="") as output_file:
        csv_writer = csv.writer(
            output_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerows(data)


def load_to_memory(exchange_id, max_retries, symbol, timeframe, since, limit):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)(
        {
            "enableRateLimit": True,
        }
    )
    if isinstance(since, str):
        since = exchange.parse8601(since)
    exchange.load_markets()
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
    if not VERBOSE == "minimal":
        print(
            "Loaded",
            len(ohlcv),
            "candles from",
            exchange.iso8601(ohlcv[0][0]),
            "to",
            exchange.iso8601(ohlcv[-1][0]),
        )
    # Переименуйте столбцы
    df = pd.DataFrame(
        ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    # df = df.drop('Timestamp', axis=1)
    print("Loaded ohlcv:", df.shape)
    return df, ohlcv[-1][0]


def scrape_candles_to_csv(
    filename, exchange_id, max_retries, symbol, timeframe, since, limit
):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)(
        {
            "enableRateLimit": True,  # required by the Manual
        }
    )
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
    # save them to csv file
    write_to_csv(filename, exchange, ohlcv)
    print(
        "Saved",
        len(ohlcv),
        "candles from",
        exchange.iso8601(ohlcv[0][0]),
        "to",
        exchange.iso8601(ohlcv[-1][0]),
        "to",
        filename,
    )
    return filename


def get_date_before(days=1):
    today = datetime.now()
    one_month_ago = today - timedelta(days=days)
    formatted_date = one_month_ago.strftime("%Y-%m-%dT%H:%M:%SZ")
    return formatted_date


if __name__ == "__main__":
    ticker = "USDT/BTC"
    timeframe = "1m"
    exchange = "binance"
    date_one_day_ago = get_date_before()
    data = load_to_memory(
        exchange_id=exchange,
        max_retries=10,
        symbol=ticker,
        timeframe=timeframe,
        since=date_one_day_ago,
        limit=200,
    )
