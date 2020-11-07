from functools import wraps

import time
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

TARGET = "gross_weight"
train = pd.read_csv("../data/Copy of train_3_4_pr.csv").iloc[:, 1:]
test = pd.read_csv("../data/submission_3.csv")["ID"]
target = train[TARGET]
time_col = "send_timestamp"

test_new = pd.DataFrame()
test_new["send_timestamp"] = pd.to_datetime(test.apply(lambda x: x[:-4]))
test_new["shipping_company"] = test.apply(lambda x: x[-3:])

memo = "send_timestamp"
# train[memo] = pd.to_datetime(train["send_timestamp"])
train["year"] = train[memo].apply(lambda x: x[:4])
train["month"] = train[memo].apply(lambda x: x[5:7])
train["day"] = train[memo].apply(lambda x: x[8:10])

train["y-m-d"] = train["year"].astype(str) + "-" + train["month"].astype(str) + "-" + train["day"].astype(str)

target = train.groupby(["y-m-d", "shipping_company"])["gross_weight"].count()

train_new = pd.DataFrame()
train_new["send_timestamp"] = [x[:10][0] for x in target.index]
train_new["shipping_company"] = [x[:10][1] for x in target.index]
train_new["gross_weight"] = target.values

print()
train_new.to_csv("../data/train_v1.csv", index=False)