from functools import wraps

import time
import pandas as pd
import numpy as np

train = pd.read_csv("../data/Copy of train_3_4_pr.csv").iloc[:, 1:]
test = pd.read_csv("../data/submission_3.csv")["ID"]
time_col = "send_timestamp"

test_new = pd.DataFrame()
test_new["send_timestamp"] = test.apply(lambda x: x[:-4])
test_new["shipping_company"] = test.apply(lambda x: x[-3:])
test_new["target"] = np.nan
test_new["train"] = False

train["y-m-d"] = train["send_timestamp"].apply(lambda x: x[:10])
train["number"] = 1
target = train.groupby(["y-m-d", "shipping_company"])["number"].count()
train_new = pd.DataFrame()
train_new["send_timestamp"] = [x[:10][0] for x in target.index]
train_new["shipping_company"] = [x[:10][1] for x in target.index]
train_new["target"] = target.values
# train_new["train"] = True

memo = pd.date_range("20190214", "20200613").astype("str")
sc1 = pd.DataFrame(memo.copy(), columns=["send_timestamp"])
sc2 = pd.DataFrame(memo.copy(), columns=["send_timestamp"])
sc3 = pd.DataFrame(memo.copy(), columns=["send_timestamp"])
sc1["shipping_company"] = "SC1"
sc2["shipping_company"] = "SC2"
sc3["shipping_company"] = "SC3"
data = pd.concat([sc1, pd.concat([sc2, sc3])]).sort_values(["send_timestamp", "shipping_company"]).reset_index(
    drop=True)

# data["target"] = 0

train_new = data.merge(train_new, on=["send_timestamp", "shipping_company"], how="left")
train_new["target"] = train_new["target"].fillna(0)
train_new["train"] = True
merged = pd.concat([train_new, test_new])

merged.to_pickle("../data/data_v2.pkl")
