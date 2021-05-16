from trainEnv.single_stock_env import SingleEnvTrain, SpaceType
from stable_baselines import PPO2, DQN
from stable_baselines.common.vec_env import DummyVecEnv
import pandas as pd
import time
import os
import gc


def run_model(model_func, space_type, data_path: str, model_name: str, model_path: str, verbose=False, timesteps=50000):
    df = pd.read_csv(data_path)
    feats = ['macd', 'macds', 'rsi_14', 'close_50_ema', 'close_20_ema', 'atr', 'cci_30', 'dx_30']
    norm = {
        'macd': 1000,
        'macds': 1000,
        'close_50_ema': 60000,
        'close_20_ema': 60000,
        'atr': 200,
        'rsi_14': 100,
        'cci_30': 100,
        'dx_30': 100,
        'close': 1
    }
    for col, val in norm.items():
        df[col] = df[col]/val

    env_train = DummyVecEnv([lambda: SingleEnvTrain(df, feats, verbose=verbose, space_type=space_type)])
    model = model_func(env_train, timesteps=timesteps)

    path = os.path.join(model_path, model_name)
    print(f"saving model {model_name} to {path}")
    model.save(path)
    return model


def train_PPO(env_train, timesteps=50000):
    """PPO model"""
    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef=0.005, nminibatches=8)

    model.learn(total_timesteps=timesteps)
    end = time.time()
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def train_DQN(env_train, timesteps=50000):
    """DQN model"""
    start = time.time()
    model = DQN('MlpPolicy', env_train, batch_size=32, tensorboard_log='/Users/yuezhang/PycharmProjects/hey-bro/results/log/')

    model.learn(total_timesteps=timesteps)
    end = time.time()
    print('Training time (DQN): ', (end - start) / 60, ' minutes')
    return model


if __name__ == "__main__":
    # from stable_baselines.common.env_checker import check_env
    data_path = "/Users/yuezhang/PycharmProjects/hey-bro/data/btc_5min_20000.csv"
    model_path = "trainedModels"

    # df = pd.read_csv(data_path)
    # feats = ['macd', 'macds', 'rsi_14', 'close_50_ema', 'close_20_ema', 'atr', 'cci_30', 'dx_30']
    #
    # env = SingleEnvTrain(df, feats, verbose=False, space_type=SpaceType.DISCRETE)
    # check_env(env)
    # print("env check completed")

    # model = run_PPO(data_path, "Model-PPO", model_path, timesteps=500000)
    model_name = 'Model_DQN'
    print(f"running model {model_name}")
    model = run_model(train_DQN, SpaceType.DISCRETE, data_path, model_name, model_path, timesteps=5000000)
    gc.collect()
