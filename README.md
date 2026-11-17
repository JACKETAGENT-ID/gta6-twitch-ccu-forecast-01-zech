# gta6-twitch-ccu-forecast-01 (zech)

Forecast of GTA VI launch-day Twitch concurrent viewers, hours 2–24, as
five quantiles per hour.

## Approach

Per-hour log-linear regression, fitted at runtime from the comparables in
the case file — nothing is hardcoded:

```
log(viewers_h) = a_h + b_h·log(p1) + c_h·log(channels_1)
```

`p1` is the case's hour-1 print, `channels_1` its hour-1 channel count
(one-predictor fallback when channels are missing). The fit is local in
scale: comparables are weighted by a Gaussian kernel (bandwidth 2 in log
units) on their hour-1 print vs the case's, cut off beyond a factor of 10,
so games of a similar size drive the forecast. The median is the fitted
value exponentiated; the quantile spread comes from the weighted residual
standard deviation at that hour (floored at 0.15 in log units), with
standard-normal z-scores (−1.645, −0.674, 0, 0.674, 1.645).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no randomness:
deterministic, ~0.2 s per case, outputs ≤ 65,536 bytes.

## Local validation (leave-one-out over the 37 comparables)

- Mean pinball loss (log scale): **0.412** overall, **0.260** on
  comparables with hour-1 print ≥ 20k, **0.210** on those ≥ 100k
  (closest in scale to a GTA VI-sized launch).
- House baseline on the same LOO protocol: 0.930 overall, 0.321 on
  hour-1 ≥ 20k. Practice baseline per RULES: ~0.387.
