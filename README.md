# Shipping Optimization Chalenge my solution

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

I didn't do features engineering just to prepare the data.

### Modeling
I did the modeling based on prophet.

We made a "logistic" prediction because the corona can cause the trend to be negative. See the parameters for details.

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

