from fbprophet import Prophet
from fbprophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_squared_error as mse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.load_base_data import preprocessed_data


def load_data():
    data = preprocessed_data()[["send_timestamp", "target", "train", "shipping_company"]]
    data = data[data["train"]].drop(columns="train")
    data = data.rename(columns={"send_timestamp": "ds", "target": "y"})
    return data


def plot_model(m, forecast, pars, mode):
    fig = m.plot(forecast)
    ax1 = fig.add_subplot(111)
    ax1.set_title(f"Forecast {mode}_{pars}", fontsize=16)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Number", fontsize=12)

    fig2 = m.plot_components(forecast)

    plt.show()


train = load_data()
data = pd.DataFrame()
best_score = 0
use_all_data = True


def run_model(pars):
    all_score = 0
    for x in ["SC1", "SC2", "SC3"]:
        if not use_all_data:
            if x == "SC2":
                use_data = train.loc[train["shipping_company"] == x, ["ds", "y"]]
                periods = 61
            else:
                use_data = train.loc[train["shipping_company"] == x, ["ds", "y"]]
                use_data = use_data.loc[pd.to_datetime(use_data["ds"]) < pd.to_datetime("2019-12-01")]
                periods = 61 + 14 + 31 + 30 + 31 + 29 + 31 + 31
        else:
            use_data = train.loc[train["shipping_company"] == x, ["ds", "y"]]
            periods = 61

        model = build_model(pars)
        model.fit(use_data)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        oof = forecast["yhat"].iloc[:-periods]
        val_y = use_data["y"]

        score = np.exp(-np.sqrt(mse(oof, val_y)))
        print("score: ", score)
        all_score += score

        pred = forecast[["ds", "yhat"]]
        # if x == "SC2":
        pred = pred.iloc[-61:, :]
        # else:
        #     pred = pred.iloc[]
        pred["company"] = x

        plot_model(model, forecast, pars, x)

        if x == "SC1":
            data = pred
        else:
            data = pd.concat([data, pred])
    all_score /= 3

    data = data.sort_values(["ds", "company"])
    return all_score, data["yhat"]


def build_model(pars):
    year_list = [2019, 2020]
    holidays = make_holidays_df(year_list=year_list, country='UK')
    wseas, mseas, yseas, s_prior, h_prior, c_prior = pars
    m = Prophet(growth='linear',
                # holidays=holidays,
                # daily_seasonality="auto",
                # weekly_seasonality="auto",
                # yearly_seasonality=False,
                # seasonality_prior_scale=s_prior,
                # holidays_prior_scale=h_prior,
                # changepoint_prior_scale=2.5
                )

    m = m.add_seasonality(
        name='weekly',
        period=7,
        fourier_order=wseas)

    m = m.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=mseas)

    # m = m.add_seasonality(
    #     name='yearly',
    #     period=365.25,
    #     fourier_order=yseas)

    return m


params = [[15, 25, 50, 0.1, 0.1, 0.1]]
best_params = None
for pars in params:
    # m = build_model(pars)
    score, forecast = run_model(pars)

    if best_score < score:
        best_score = score
        best_params = pars
        best_forecast = forecast

print(best_score, best_params)
forecast[forecast < 0] = 0
# forecast.to_csv(f"../outputs/{round(best_score, 5)}_prophet_non_year.csv", index=False, header=False)
