import random
import gc
import time
import datetime
import pickle
from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from datetime import datetime as dt


VERSION = 3
DEBUG = False
MODES=["validation", "all_train"]
SEED = 22
num_rounds = 10 if DEBUG else 1200

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'max_depth': 7,
    'num_leaves': 64,
    'max_bin': 31,
    'nthread': -1,
    'bagging_freq': 1,
    'verbose': -1,
    'seed': SEED,
}

data = pd.read_pickle(f"../data/train_v{VERSION}.pkl")
data["shipping_company"] = data["shipping_company"].astype("category").cat.codes

train = data[data["train"]].drop(columns="train")
test = data[~data["train"]].drop(columns=["train", "target", "send_timestamp"])
target = train["target"]
# train = train.drop(columns="target")

train["validation"] = False
train["send_timestamp"] = pd.to_datetime(train["send_timestamp"])
train.loc[train["send_timestamp"] > dt(2020, 4, 14), "validation"] = True

use_cols = [x for x in train.columns if x not in ["send_timestamp", "validation", "target"]]

best_iter = 0

for MODE in MODES:
    if MODE == "validation":
        d_train = lgb.Dataset(
            train.loc[~train["validation"], use_cols],
            label=train.loc[~train["validation"], "target"]
        )
        d_valid = lgb.Dataset(
            train.loc[train["validation"], use_cols],
            label=train.loc[train["validation"], "target"]
        )
        estimator = lgb.train(
            params=params,
            train_set=d_train,
            num_boost_round=num_rounds,
            valid_sets=[d_train, d_valid],
            verbose_eval=100,
            early_stopping_rounds=100
        )

        y_pred = estimator.predict(test) + 50

        best_iter = estimator.best_iteration
        # pred += y_pred / N_FOLDS

        # print(fold + 1, "done")

        # score += estimator.best_score["valid_1"]["rmse"] / N_FOLDS
        lgb.plot_importance(estimator, importance_type="gain", max_num_features=25)
        plt.show()
    else:
        d_train = lgb.Dataset(
            train[use_cols],
            label=train["target"]
        )

        estimator = lgb.train(
            params=params,
            train_set=d_train,
            num_boost_round=best_iter,
            valid_sets=[d_train],
            verbose_eval=100,
            early_stopping_rounds=1000
        )
        y_pred = estimator.predict(test)
        lgb.plot_importance(estimator, importance_type="gain", max_num_features=25)
        plt.show()

pd.Series(y_pred).to_csv("../outputs/lgbm_v2.csv", index=False, header=False)

