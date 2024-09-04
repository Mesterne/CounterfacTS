import argparse
import os
import sys

import numpy as np
from statsmodels.tsa.seasonal import STL

sys.path.append(".")
from src.utils.features import trend_determination, trend_slope, trend_linearity, seasonal_determination
from src.utils.data_loading import load_test_data
from src.utils.transformations import manipulate_trend_component, manipulate_seasonal_determination


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="A GluonTS dataset")
parser.add_argument("--sp", type=int, default=24, help="Seasonal periodicity of the dataset", required=True)
parser.add_argument("--context-length", type=int, default=168, help="Length of input window", required=True)
parser.add_argument("--prediction-length", type=int, default=24, help="Length of horizon", required=True)
parser.add_argument("--step-size", type=float, default=0.01, help="Step size for each subsequent transformation",
                    required=True)


def calculate_features(time_series, sp):
    decomp = STL(time_series, period=sp).fit()

    features = np.zeros(4)
    features[0] = trend_determination(decomp.trend, decomp.resid)
    features[1] = trend_slope(decomp.trend)
    features[2] = trend_linearity(decomp.trend)
    features[3] = seasonal_determination(decomp.seasonal, decomp.resid)
    return features, decomp


def brute_force_trend_str(ts, step, sp):
    features, decomp = calculate_features(ts, sp)

    f = 1
    ts_arr = []
    features_arr = []
    while 0.01 <= f <= 1.99:
        f += step
        trend = manipulate_trend_component(decomp.trend, f, g=1, h=1, m=0)

        ts = trend + decomp.seasonal + decomp.resid
        features, _ = calculate_features(ts, sp)

        ts_arr.append(ts)
        features_arr.append(features)

    return ts_arr, features_arr


def brute_force_trend_slope(ts, step, sp, context_length):
    features, decomp = calculate_features(ts, sp)

    m = 0
    ts_arr = []
    features_arr = []
    while abs(m) <= 0.99:
        m += step
        trend = manipulate_trend_component(decomp.trend, f=1, g=1, h=1, m=m / context_length)

        ts = trend + decomp.seasonal + decomp.resid
        features, _ = calculate_features(ts, sp)

        ts_arr.append(ts)
        features_arr.append(features)

    return ts_arr, features_arr


def brute_force_trend_lin(ts, step, sp):
    features, decomp = calculate_features(ts, sp)

    h = 1
    ts_arr = []
    features_arr = []
    while 0.01 <= h <= 1.99:
        h += step
        trend = manipulate_trend_component(decomp.trend, f=1, g=1, h=h, m=0)

        ts = trend + decomp.seasonal + decomp.resid
        features, _ = calculate_features(ts, sp)

        ts_arr.append(ts)
        features_arr.append(features)

    return ts_arr, features_arr


def brute_force_seasonal_str(ts, step, sp):
    features, decomp = calculate_features(ts, sp)

    k = 1
    ts_arr = []
    features_arr = []
    while 0.01 <= k <= 1.99:
        k += step
        seasonality = manipulate_seasonal_determination(decomp.seasonal, k)

        ts = decomp.trend + seasonality + decomp.resid
        features, _ = calculate_features(ts, sp)

        ts_arr.append(ts)
        features_arr.append(features)

    return ts_arr, features_arr


if __name__ == '__main__':
    args = vars(parser.parse_args())
    dataset = args["dataset"]
    sp = args["sp"]
    context_length = args["context_length"]
    prediction_length = args["prediction_length"]
    step = args["step_size"]
    save_path = f"./data/{dataset}/generated/test"
    os.makedirs(save_path, exist_ok=True)
    test_data = load_test_data(dataset, ts_length=context_length + prediction_length)

    for i, ts in enumerate(test_data):
        print(f"Generating variations for time series {i}")

        inc_trend_str_tss, inc_trend_str_feat = brute_force_trend_str(ts, step, sp)
        dec_trend_str_tss, dec_trend_str_feat = brute_force_trend_str(ts, -step, sp)
        trend_str_tss = [*inc_trend_str_tss, *dec_trend_str_tss]
        trend_str_feat = [*inc_trend_str_feat, *dec_trend_str_feat]
        np.save(f"{save_path}/ts_trend_str{i}.npy", np.array(trend_str_tss))
        np.save(f"{save_path}/feat_trend_str{i}.npy", np.array(trend_str_feat))

        inc_trend_slope_tss, inc_trend_slope_feat = brute_force_trend_slope(ts, step, sp, context_length)
        dec_trend_slope_tss, dec_trend_slope_feat = brute_force_trend_slope(ts, -step, sp, context_length)
        trend_slope_tss = [*inc_trend_slope_tss, *dec_trend_slope_tss]
        trend_slope_feat = [*inc_trend_slope_feat, *dec_trend_slope_feat]
        np.save(f"{save_path}/ts_trend_slope{i}.npy", np.array(trend_slope_tss))
        np.save(f"{save_path}/feat_trend_slope{i}.npy", np.array(trend_slope_feat))

        inc_trend_lin_tss, inc_trend_lin_feat = brute_force_trend_lin(ts, step, sp)
        dec_trend_lin_tss, dec_trend_lin_feat = brute_force_trend_lin(ts, -step, sp)
        trend_lin_tss = [*inc_trend_lin_tss, *dec_trend_lin_tss]
        trend_lin_feat = [*inc_trend_lin_feat, *dec_trend_lin_feat]
        np.save(f"{save_path}/ts_trend_lin{i}.npy", np.array(trend_lin_tss))
        np.save(f"{save_path}/feat_trend_lin{i}.npy", np.array(trend_lin_feat))

        inc_seasonal_str_tss, inc_seasonal_str_feat = brute_force_seasonal_str(ts, step, sp)
        dec_seasonal_str_tss, dec_seasonal_str_feat = brute_force_seasonal_str(ts, -step, sp)
        seasonal_str_tss = [*inc_seasonal_str_tss, *dec_seasonal_str_tss]
        seasonal_str_feat = [*inc_seasonal_str_feat, *dec_seasonal_str_feat]
        np.save(f"{save_path}/ts_seasonal_str{i}.npy", np.array(seasonal_str_tss))
        np.save(f"{save_path}/feat_seasonal_str{i}.npy", np.array(seasonal_str_feat))
