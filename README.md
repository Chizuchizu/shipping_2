# Shipping Optimization Challenge my solution

## how to run

### RUN

```shell script
python src/prophet_v1.py
```

### My environment

#### Python

- Python 3.7.9
- conda
- library(requirements.txt)

#### Computer

- Ubuntu 20.04
- CPU: i9 9900K
- Memory: 16 * 2 (GiB)

## solution

### Features

Since there is no data for days when there was no trade, I have added such data that such days would also be zero. (
Adding it increased the score compared to not adding it.)

I didn't do features engineering just to prepare the data.

### Modeling

I did the modeling based on prophet.

I made a "logistic" prediction because the corona can cause the trend to be negative. See the parameters for details.

The discussion I referred to ↓
https://github.com/facebook/prophet/issues/1668

```
params = {
    "growth": "logistic",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 30,
    "mcmc_samples": 0,
    "seasonality_mode": "multiplicative",
}
```

Holiday data were not included because they were a factor in over-fitting.

#### What didn't work

- LGBM modeling(ex. https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50)
- complex parameters
- features(on Prophet)
