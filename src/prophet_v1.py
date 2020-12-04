from fbprophet import Prophet
from fbprophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_squared_error as mse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import pickle

from src.load_base_data import preprocessed_data


class ProphetPos(Prophet):

    @staticmethod
    def piecewise_linear(t, deltas, k, m, changepoint_ts):
        """Evaluate the piecewise linear function, keeping the trend
        positive.

        Parameters
        ----------
        t: np.array of times on which the function is evaluated.
        deltas: np.array of rate changes at each changepoint.
        k: Float initial rate.
        m: Float initial offset.
        changepoint_ts: np.array of changepoint times.

        Returns
        -------
        Vector trend(t).
        """
        # Intercept changes
        gammas = -changepoint_ts * deltas
        # Get cumulative slope and intercept at each t
        k_t = k * np.ones_like(t)
        m_t = m * np.ones_like(t)
        for s, t_s in enumerate(changepoint_ts):
            indx = t >= t_s
            k_t[indx] += deltas[s]
            m_t[indx] += gammas[s]
        trend = k_t * t + m_t
        if max(t) <= 1:
            return trend
        # Add additional deltas to force future trend to be positive
        indx_future = np.argmax(t >= 1)
        while min(trend[indx_future:]) < 0:
            indx_neg = indx_future + np.argmax(trend[indx_future:] < 0)
            k_t[indx_neg:] -= k_t[indx_neg]
            m_t[indx_neg:] -= m_t[indx_neg]
            trend = k_t * t + m_t
        return trend

    def predict(self, df=None):
        fcst = super().predict(df=df)
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            fcst[col] = fcst[col].clip(lower=0.0)
        return fcst


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
best_score = 0
use_all_data = True
debug = True
rand = np.random.randint(0, 1000000)
experiment_name = f"debug_{rand}" if debug else f"{rand}"
"""
イギリスのロックダウン
https://www.bloomberg.co.jp/news/articles/2020-03-23/Q7NYY4T0AFB401
https://www.bbc.com/japanese/features-and-analysis-52612672
2020-03-23 〜　2020-05-11



    def __init__(
            self,
            growth='linear',
            changepoints=None,
            n_changepoints=25,  
            changepoint_range=0.8,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            holidays=None,
            seasonality_mode='additive',
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
            stan_backend=None
    ):
    multiplicative

https://github.com/facebook/prophet/issues/1668
"""
lockdown_holidays = pd.DataFrame()
lockdown_holidays["ds"] = pd.date_range("2020-03-24", "2020-05-13", freq="D")
lockdown_holidays["holiday"] = "covid-19"
params = {
    "growth": "logistic",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 30,
    "mcmc_samples": 0,
    "seasonality_mode": "multiplicative",
    # "daily_seasonality": True,
    #  "weekly_seasonality": True,
    # "changepoints": ["2020-03-23", "2020-05-11"],
    # "holidays": lockdown_holidays
    # "prophet_pos": multiplicative
    # "likelihood": "NegBinomial"
}

prophet_pos = False


def is_lockdown(ds):
    return (pd.to_datetime("2020-03-23") < pd.to_datetime(ds)) or (pd.to_datetime(ds) < pd.to_datetime("2020-05-12"))


def run_model():
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

        # mlflow.set_experiment(experiment_name)
        mlflow.set_experiment(x)

        # with mlflow.start_run(run_name=f"{x}"):
        with mlflow.start_run(run_name=f"{rand}"):
            model = build_model()

            # model.add_seasonality(name="lockdown", period=7, fourier_order=3, condition_name="lockdown")
            # model.add_seasonality(name="normal", period=7, fourier_order=3, condition_name="normal")
            #
            # use_data["lockdown"] = use_data["ds"].apply(is_lockdown)
            # use_data["normal"] = ~use_data["ds"].apply(is_lockdown)

            if params["growth"] == "logistic":
                use_data["cap"] = use_data["y"].max() * 1.2
            model.fit(use_data)
            future = model.make_future_dataframe(periods=periods)

            # future["lockdown"] = future["ds"].apply(is_lockdown)
            # future["normal"] = ~future["ds"].apply(is_lockdown)

            if params["growth"] == "logistic":
                future["cap"] = use_data["y"].max() * 1.2

            forecast = model.predict(future)

            oof = forecast["yhat"].iloc[:-periods]
            val_y = use_data["y"]

            score = np.exp(-np.sqrt(mse(oof, val_y)))
            print("score: ", score)
            all_score += score

            pred = forecast[["ds", "yhat"]]
            # if x == "SC2":
            pred = pred.iloc[-61:, :]

            # if x == "SC1":
            #     pred["yhat"] = pred["yhat"].round()

            # else:
            #     pred = pred.iloc[]
            pred["company"] = x

            # https://github.com/facebook/prophet/issues/725
            pkl_path = f"../models/{rand}_{x}_prophet.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)

            # plot_model(model, forecast, pars, x)
            fig1 = model.plot(forecast)
            plt.savefig("fig1.png")
            plt.clf()
            fig2 = model.plot_components(forecast)
            plt.savefig("fig2.png")
            plt.clf()

            if x == "SC1":
                data = pred
            else:
                data = pd.concat([data, pred])

            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("score", score)
            mlflow.log_artifact("fig1.png")
            mlflow.log_artifact("fig2.png")
            mlflow.log_param("ProphetPos", prophet_pos)

    all_score /= 3
    data.loc[data["yhat"] < 0, "yhat"] = 0
    sorted_data = data.sort_values(["ds", "company"])
    return all_score, sorted_data["yhat"], data


def build_model():
    year_list = [2019, 2020]
    holidays = make_holidays_df(year_list=year_list, country='IN')
    # wseas, mseas, yseas, s_prior, h_prior, c_prior = pars

    if prophet_pos:
        m = ProphetPos(**params)
    else:
        m = Prophet(**params)

    m = m.add_seasonality(
        name='weekly',
        period=7,
        fourier_order=15)

    m = m.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=25)

    # m = m.add_seasonality(
    #     name='yearly',
    #     period=365.25,
    #     fourier_order=yseas)

    return m


#
# params = [[15, 25, 50, 0.1, 0.1, 0.1]]
# best_params = None
# for pars in params:
#     # m = build_model(pars)
#     score, forecast = run_model(pars)
#
#     if best_score < score:
#         best_score = score
#         best_params = pars
#         best_forecast = forecast

score, forecast, data = run_model()

print(score)
# forecast[forecast < 0] = 0
forecast.to_csv(f"../outputs/{round(best_score, 5)}_{rand}_prophet.csv", index=False, header=False)

mlflow.set_experiment("all")
with mlflow.start_run(run_name=f"{rand}"):
    for k, v in params.items():
        mlflow.log_param(k, v)
    mlflow.log_param("score", score)
    mlflow.log_param("ProphetPos", prophet_pos)
    mlflow.log_artifact(f"../outputs/{round(best_score, 5)}_{rand}_prophet.csv")

    for x in ["SC1", "SC2", "SC3"]:
        memo = data.loc[data["company"] == x]
        for i, row in enumerate(memo.itertuples()):
            mlflow.log_metric(key=x, value=row.yhat, step=i)
