import argparse
import os
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.repository.datasets import get_dataset
from matplotlib import cm
from tqdm import tqdm

sys.path.append(".")
from src.utils.evaluation import score_batch
from src.models.utils import get_model
from src.utils.data_loading import load_score

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="Path to a config file")
parser.add_argument("feature_name", type=str, help="Name of feature (in filename) to be loaded")
parser.add_argument("mode", type=str, choices=["all", "clip", "remove"], default="all",
                    help="How time series with observations outside the min and max values are filtered."
                         "'all' to apply no filter,"
                         "'clip' to clip observations between the min and max seen in test data,"
                         "'remove' to remove time series with observations higher than seen in test data.")


def load_test_data(dataset, ts_length):
    dataset = get_dataset(dataset).test
    data = []
    max_val = float("-inf")
    min_val = float("inf")
    for ts in tqdm(dataset):
        if max_val < ts["target"].max():
            max_val = ts["target"].max()
        if min_val > ts["target"].min():
            min_val = ts["target"].min()

        values = ts["target"][-ts_length:]
        index = pd.date_range(ts["start"], periods=len(values), freq=ts["start"].freq)
        ts = pd.Series(data=values, index=index)
        data.append(ts)

    return data, max_val, min_val


def load_generated_test_data(path, prefix, test_data_len):
    data = [0 for i in range(test_data_len)]
    for f in tqdm(os.listdir(path)):
        f_name = f.split(".")[0]
        if f_name.startswith(prefix):
            index = int(f_name[len(prefix):])
            arr = np.load(os.path.join(path, f))
            data[index] = arr
    return np.vstack(data)


def predict_generated_data(model, data, config):
    mape = []
    smape = []
    mase = []
    seasonal_mase = []
    mse = []
    for i in tqdm(range(0, len(data), 128)):
        batch = np.expand_dims(data[i:i + 128], axis=-1)
        context = batch[:, :config["context_length"], :]
        target = batch[:, -config["prediction_length"]:, 0]
        preds = model.predict(context)[:, :, 0]

        scores = score_batch(target, preds, context, config["sp"])
        mape.append(scores[0])
        smape.append(scores[1])
        mase.append(scores[2])
        seasonal_mase.append(scores[3])
        mse.append(scores[4])

    mape = np.vstack(mape)
    smape = np.vstack(smape)
    mase = np.vstack(mase)
    seasonal_mase = np.vstack(seasonal_mase)
    mse = np.vstack(mse)

    return mape, smape, mase, seasonal_mase, mse


def save_generated_data_metric(path, feature, metric, array):
    np.save(os.path.join(path, f"generated_{feature}_{metric}.npy"), array)


def test_and_save(model, data, feature, config, path):
    metrics = predict_generated_data(model, data, config)

    names = ["mape", "smape", "mase", "smase", "mse"]
    for metric, name in zip(metrics, names):
        save_generated_data_metric(path, feature, name, metric)


def load_generated_test_scores(path, feature):
    mape = np.load(os.path.join(path, f"generated_{feature}_mape.npy"))
    smape = np.load(os.path.join(path, f"generated_{feature}_smape.npy"))
    mase = np.load(os.path.join(path, f"generated_{feature}_mase.npy"))
    seasonal_mase = np.load(os.path.join(path, f"generated_{feature}_smase.npy"))
    mse = np.load(os.path.join(path, f"generated_{feature}_mse.npy"))

    return mape, smape, mase, seasonal_mase, mse


def plot_average_error_vs_feature_val(metric_scores, feature_vals, mean, std, metric_name, feature_name, path,
                                      num_bins="auto", limit_x=False):
    hist, bins = np.histogram(feature_vals, density=True, bins=num_bins)

    binned_metric = [[] for i in range(len(bins))]
    for score, val in zip(metric_scores, feature_vals):
        bin_num = np.where(val >= bins)[0][-1]
        binned_metric[bin_num].append(score.mean(axis=-1))

    binned_metric = [np.array(metric_bin).mean() if len(metric_bin) > 0 else np.nan for metric_bin in binned_metric]

    plt.plot(bins, binned_metric, label=f"average {metric_name}")
    plt.vlines(mean, ymin=min(binned_metric), ymax=max(binned_metric), color="C1", label="mean feature value")
    if mean - std >= min(bins):
        plt.vlines(mean - std, ymin=min(binned_metric), ymax=max(binned_metric), color="C1", linestyles=":",
                   label="mean \u00B1 1 std")
    if mean - (2 * std) >= min(bins):
        plt.vlines(mean - (2 * std), ymin=min(binned_metric), ymax=max(binned_metric), color="C1", linestyles="--",
                   label="mean \u00B1 2 std")
    if mean + std <= max(bins):
        # trick to avoid two legends for the lines for the standard deviations
        label = "mean \u00B1 1 std" if mean - std <= min(bins) else ""
        plt.vlines(mean + std, ymin=min(binned_metric), ymax=max(binned_metric), color="C1", linestyles=":",
                   label=label)
    if mean + (2 * std) <= max(bins):
        label = "mean \u00B1 2 std" if mean - (2 * std) <= min(bins) else ""
        plt.vlines(mean + (2 * std), ymin=min(binned_metric), ymax=max(binned_metric), color="C1", linestyles="--",
                   label=label)

    if limit_x:
        plt.xlim([mean - (3 * std), mean + (3 * std)])

    plt.title(metric_name)
    plt.xlabel(feature_name)
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(
        os.path.join(path, f"{feature_name}_{metric_name}_error_vs_feature_val{'_limit' if limit_x else ''}.png")
    )
    plt.clf()


def compare_steps_with_baseline(new_scores, original_scores, nsteps, prediction_length):
    inc_comparisons = np.empty((original_scores.shape[0], nsteps, prediction_length))
    for i, baseline in tqdm(enumerate(original_scores)):
        for j in range(nsteps):
            score = new_scores[i * (nsteps * 2) + j]
            inc_comparisons[i, j] = score - baseline

    dec_comparisons = np.empty((original_scores.shape[0], nsteps, prediction_length))
    for i, baseline in tqdm(enumerate(original_scores)):
        for j in range(nsteps, nsteps * 2):
            score = new_scores[i * (nsteps * 2) + j]
            dec_comparisons[i, j - nsteps] = score - baseline

    return inc_comparisons, dec_comparisons


def plot_horizon_error_change_per_step(inc_comparisons, dec_comparisons, feature_name, metric_name, path):
    def plot_comparisons(ax, comparisons):
        for i, avg_comp in enumerate(comparisons):
            ax.plot(avg_comp, color=viridis(i), alpha=0.5)

    viridis = cm.get_cmap('viridis', nsteps)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_title(f"Increasing {feature_name} {metric_name}")
    plot_comparisons(ax1, inc_comparisons)

    ax2.set_title(f"Decreasing {feature_name} {metric_name}")
    plot_comparisons(ax2, dec_comparisons)

    sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, label="distance from original time series")
    plt.savefig(os.path.join(path, f"{feature_name}_{metric_name}_horizon_change.png"))
    plt.clf()


def plot_error_change_per_step(inc_mean_step_horizon_error, dec_mean_step_horizon_error, feature_name, metric_name, path):
    plt.plot(inc_mean_step_horizon_error, label=f"Increasing {feature_name}")
    plt.plot(dec_mean_step_horizon_error, label=f"Decreasing {feature_name}")
    plt.legend()
    plt.xlabel("Num steps from original")
    plt.ylabel(f"Change in {metric_name}")
    plt.title(metric_name)
    plt.savefig(os.path.join(path, f"{feature_name}_{metric_name}_step_change.png"))
    plt.clf()


if __name__ == '__main__':
    args = vars(parser.parse_args())

    names = ["trend_str", "trend_slope", "trend_lin", "seasonal_str"]
    feature_name = args["feature_name"]
    i = names.index(feature_name)
    print(f"Predicting and analyzing {feature_name}")

    with open(args["config_path"], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generated_data_path = f"./data/{config['dataset']}/generated/test"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_data, max_val, min_val = load_test_data(config["dataset"],
                                                 config["context_length"] + config["prediction_length"])
    metric_names = ["MAPE", "sMAPE", "MASE", "seasonal_MASE", "MSE"]

    model = get_model(config["model_name"])(**config["model_args"], device=device, path=config["path"])
    model.load_state_dict(torch.load(os.path.join(config["path"], "model.pth")))

    experiment_generated_path = os.path.join(config["path"], "generated_test", args["mode"])
    if not os.path.exists(experiment_generated_path):
        os.makedirs(experiment_generated_path, exist_ok=True)

    time_series = load_generated_test_data(generated_data_path, f"ts_{feature_name}", len(test_data))
    features = load_generated_test_data(generated_data_path, f"feat_{feature_name}", len(test_data))

    # filter time series according to the mode argument
    if args["mode"] == "clip":
        time_series = np.clip(time_series, min_val, max_val)  # leave features unchanged even though it is not correct
    elif args["mode"] == "remove":
        max_values = np.max(time_series, axis=-1)
        min_values = np.min(time_series, axis=-1)

        # we only select time series where the max/min value is smaller/greater than the max/min seen in the entire
        # dataset
        max_mask = max_values <= max_val
        min_mask = min_values >= min_val
        mask = np.logical_and(max_mask, min_mask)
        time_series = time_series[mask]
        features = features[mask]

    metric_path = os.path.join(experiment_generated_path, feature_name)
    if not os.path.exists(metric_path):
        os.mkdir(metric_path)
        test_and_save(model, time_series, feature_name, config, metric_path)

    metrics = load_generated_test_scores(metric_path, feature_name)
    features = np.nan_to_num(features)  # some features might be undefined
    feature_values = features[:, i]
    feature_mean = np.nanmean(features, axis=0)[i]
    feature_std = np.nanstd(features, axis=0)[i]

    for metric, name in zip(metrics, metric_names):
        print(f"Creating plots for {name}")
        plot_average_error_vs_feature_val(metric, feature_values, feature_mean, feature_std, name, feature_name,
                                          metric_path)
        # the plots for trend slope have a lot of outliers so create a plot with a limitation on the x-axis range
        if feature_name == "trend_slope":
            plot_average_error_vs_feature_val(metric, feature_values, feature_mean, feature_std, name, feature_name,
                                              metric_path, limit_x=True)

        nsteps = metric.shape[0] // len(test_data) // 2  # number of steps taken in a single direction for the feature
        original_scores = load_score(config["path"], name)
        inc_comparisons, dec_comparisons = compare_steps_with_baseline(metric, original_scores, nsteps,
                                                                       config["prediction_length"])

        plot_horizon_error_change_per_step(np.nanmean(inc_comparisons, axis=0), np.nanmean(dec_comparisons, axis=0),
                                           feature_name, name, metric_path)
        plot_error_change_per_step(np.nanmean(np.nanmean(inc_comparisons, axis=0), axis=-1),
                                   np.nanmean(np.nanmean(dec_comparisons, axis=0), axis=-1),
                                   feature_name,
                                   name,
                                   metric_path)
