from preprocessing.utils import Preprocess
import talib


class DataLoader:
    def __init__(self, data=None):
        self.data = data
        self.close = None
        self.high = None
        self.low = None
        self.open = None

    def load_kline(self, symbol='BTC', period='1m', size=1000):
        """
        Load data from API
        :param period:
        :param symbol:
        :param size:
        :return:
        """
        preprocess = Preprocess()
        self.data = preprocess.get_bnb_data(symbol, period, size)
        self.init_close_high_low_open()
        return self.data

    def add_data(self, data):
        self.data = data
        self.init_close_high_low_open()

    def calculate_rsi(self, period=14):
        """
        Calculate rsi
        :return: return a numpy.ndarray
        """
        rsi = talib.RSI(self.close, timeperiod=period)
        return rsi

    def calculate_stochrsi(self, period=14):
        k, d = talib.STOCHRSI(self.close, timeperiod=period)
        return k, d

    def calculate_sma(self, period):
        sma = talib.SMA(self.close, timeperiod=period)
        return sma

    def calculate_ema(self, period):
        ema = talib.EMA(self.close, timeperiod=period)
        return ema

    def calculate_macd(self, fastp=12, slowp=26, signalp=9):
        macd, macds, macdh = talib.MACD(self.close, fastperiod=fastp, slowperiod=slowp, signalperiod=signalp)
        return macd, macds, macdh

    def rolling_low(self, period=5):
        rl = [0] * (period-1) + [min(self.low[i-period+1:i + 1]) for i in range(period-1, len(self.close))]
        return rl

    def calculate_roc(self, period=1):
        """
        Rate of change : ((price/prevPrice)-1)*100
        :return: 
        """
        real = talib.ROC(self.close, timeperiod=period)
        return real

    def calculate_general_features(self):
        rsi_14 = self.calculate_rsi(14)
        ema_50 = self.calculate_ema(50)
        ema_120 = self.calculate_ema(120)
        macd, macds, macdh = self.calculate_macd()
        macd_crossover = [0] + [1 if macdh[i - 1] < 0 < macdh[i] else 0 for i in range(1, len(macdh))]
        rsi_over_50 = [1 if i > 50 else 0 for i in rsi_14]
        close_over_ema_50 = [1 if self.close[i] > ema_50[i] else 0 for i in range(len(ema_50))]
        ema_50_over_120 = [1 if ema_50[i] > ema_120[i] else 0 for i in range(len(ema_50))]
        stochk, stochd = self.calculate_stochrsi()
        stochrsi_under_30 = [1 if stochk[i] < 30 and stochd[i] < 30 else 0 for i in range(len(stochk))]
        stochrsi_above_70 = [1 if stochk[i] > 70 and stochd[i] > 70 else 0 for i in range(len(stochk))]
        roc1 = self.calculate_roc(1)

        self.data['rsi_14'] = rsi_14
        self.data['ema_50'] = ema_50
        self.data['ema_120'] = ema_120
        self.data['macd'] = macd
        self.data['macdh'] = macdh
        self.data['macds'] = macds
        self.data['macd_crossover'] = macd_crossover
        self.data['ris_over_50'] = rsi_over_50
        self.data['close_over_ema_50'] = close_over_ema_50
        self.data['ema_50_over_120'] = ema_50_over_120
        self.data['stochk'] = stochk
        self.data['stochd'] = stochd
        self.data['stochrsi_under_30'] = stochrsi_under_30
        self.data['stochrsi_above_70'] = stochrsi_above_70
        self.data['rolling_low_5'] = self.rolling_low(5)
        self.data['roc1'] = roc1

        return self.data

    def init_close_high_low_open(self):
        self.close = self.data['close'].values
        self.high = self.data['high'].values
        self.low = self.data['low'].values
        self.open = self.data['open'].values


if __name__ == "__main__":
    import pandas as pd
    dl = DataLoader()
    dl.load_kline('BTC', '1m', size=300)
    data = dl.calculate_general_features()
    
    aa = talib.CDL3INSIDE(dl.open, dl.high, dl.low, dl.close)
    print(aa)
    # print('sma \n', a[-2:], b[-2:], c[-2:])
