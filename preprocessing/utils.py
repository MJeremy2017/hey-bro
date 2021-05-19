import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import requests
import stockstats


class Preprocess:
    def __init__(self):
        self.df = None

    def get_ohlc(self, symbol, period, size=10, toTs=None):
        from_coin = symbol
        to_coin = 'usd'
        cat = period
        limit = size

        cats = {"min": "histominute", "hour": "histohour", "day": "histoday"}
        factors = {"min": 60, "hour": 3600}
        if cat not in cats.keys():
            return False
        factor = factors[cat]
        if toTs is None: toTs = int(time.time()) // factor * factor
        url = "https://min-api.cryptocompare.com/data/" + cats[cat]
        if limit <= 2000:
            get_url = url + "?fsym={}".format(from_coin.upper()) + \
                      "&tsym={}".format(to_coin.upper()) + "&limit={}".format(limit) + "&toTs={}".format(toTs)
            print(get_url)
            resp = requests.get(get_url).json()
            if resp["Response"] == "Error":
                raise ValueError("Wrong request")
            print("Success")
            return resp["Data"]
        else:
            resps = []
            toTs = None
            while limit > 2000:
                resp = self.get_ohlc(symbol, period, size=2000, toTs=toTs)
                resps = resp + resps
                toTs = int(resp[0]["time"]) // factor * factor - factor
                limit -= 2000
            resp = self.get_ohlc(symbol, period, size=limit, toTs=toTs)
            resps = resp + resps
            return resps

    # get bnb kline from binance
    def get_bnb_kline(self, symbol, period, size, toTs=None):
        limit = size
        max_size = 1500;

        end_time = None
        if toTs:
            end_time = toTs * 1000

        if limit <= max_size:
            get_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}USDT&interval={period}&limit={limit}&endTime={end_time}"
            print(get_url)
            resp = requests.get(get_url).json()

            return [{"time": int(s[0] / 1000), "high": float(s[2]),
                     "low": float(s[3]), "open": float(s[1]),
                     "close": float(s[4]), "volume": float(s[5])} for s in resp]
        else:
            resps = []
            toTs = None
            while limit > max_size:
                resp = self.get_bnb_kline(symbol, period, size=max_size, toTs=toTs)
                resps = resp + resps
                toTs = resp[0]['time'] - 60
                limit -= max_size
            resp = self.get_bnb_kline(symbol, period, size=limit, toTs=toTs)
            resps = resp + resps
            return resps

    def get_bnb_data(self, symbol, period, size):
        feats = ['time', 'high', 'low', 'open', 'close']
        self.df = pd.DataFrame(self.get_bnb_kline(symbol, period, size))
        self.df = self.df[feats]
        self.df.rename(columns={"time": "start_time_unix"}, inplace=True)
        self.df["start_date"] = self.df["start_time_unix"].apply(
            lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
        self.df.sort_values("start_time_unix", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def get_more_data(self, symbol, period, size):
        feats = ['time', 'high', 'low', 'open', 'close']
        self.df = pd.DataFrame(self.get_ohlc(symbol, period, size))
        self.df = self.df[feats]
        self.df.rename(columns={"time": "start_time_unix"}, inplace=True)
        self.df["start_date"] = self.df["start_time_unix"].apply(
            lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
        self.df.sort_values("start_time_unix", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def get_features(self, feats, df=None):
        df = df if df else self.df
        df_copy = df.copy()
        sk = stockstats.StockDataFrame.retype(df_copy)
        for f in feats:
            df_copy[f] = sk.get(f)
        self.df = df_copy
        return self.df
