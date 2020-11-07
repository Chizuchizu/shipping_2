# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, random

features = pd.read_pickle("../data/data_v2.pkl").reset_index(drop=True)

shift_days = [61, 90, 120]
roll_days = [7, 14, 21, 30]

for s in shift_days:
    for r in roll_days:
        features[f"roll_mean_{s}_{r}"] = features.groupby(["shipping_company"])["target"].transform(
            lambda x: x.shift(s).rolling(r).mean()).fillna(method="bfill")

        features[f"roll_std_{s}_{r}"] = features.groupby(["shipping_company"])["target"].transform(
            lambda x: x.shift(s).rolling(r).std()).fillna(method="bfill")

# time_df = pd.to_datetime(features["send_timestamp"])
# features["year"] = time_df.apply(lambda x: x.year)
# features["month"] = time_df.apply(lambda x: x.month)
# features["day"] = time_df.apply(lambda x: x.day)
# features["weekday"] = time_df.apply(lambda x: x.dayofweek)

# features = features.drop(columns="send_timestamp")
# features["dayofweek"] = time.

features.to_pickle("../data/train_v4.pkl")
