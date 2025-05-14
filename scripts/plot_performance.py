import string

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


def get_ticks(data_min, data_max, n_spaces):
    delta = data_max - data_min
    return [data_min + i * delta / n_spaces for i in range(n_spaces + 1)]


def get_all_ticks(x, y, n_spaces):
    x_min = np.min(x)
    x_max = np.max(x)
    delta_x = x_max - x_min
    x_ticks = get_ticks(x_min, x_max, n_spaces)

    y_min = np.min(y)
    y_max = np.max(y)
    delta_y = y_max - y_min
    y_ticks = get_ticks(y_min, y_max, n_spaces)
    return x_ticks, y_ticks, x_min, x_max, y_min, y_max, delta_x, delta_y


def histogram_overlap_continuous(data1, data2, bins=100, range=None):
    hist1, bin_edges = np.histogram(data1, bins=bins, range=range, density=True)
    hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
    bin_widths = np.diff(bin_edges)
    return np.sum(np.minimum(hist1, hist2) * bin_widths)


def symmetric_histogram_overlap_continuous(data1, data2, bins=100, range=None):
    overlap_1_to_2 = histogram_overlap_continuous(data1, data2, bins=bins, range=None)
    overlap_2_to_1 = histogram_overlap_continuous(data2, data1, bins=bins, range=None)

    return (overlap_1_to_2 + overlap_2_to_1) / 2


def plot(df):
    metric = "mcc"
    fontsize = 18
    fontsize_small = fontsize - 10
    n_spaces = 4
    offset_scale = 0.2
    formatter = ticker.StrMethodFormatter("{x:.2f}")
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )
    filter_width = 25

    # plot 1
    alignment_measure = "full_alignment_pc_fp"

    pc = df[(df.feature_type == "PC")][
        ["dataset", alignment_measure, metric, "test_set_size", "test_set_balance"]
    ]
    fp = df[(df.feature_type == "FP")][["dataset", metric, alignment_measure]]
    pcfp = df[(df.feature_type == "PCFP")][["dataset", metric, alignment_measure]]
    meani = df[(df.feature_type == "MEAN")][["dataset", metric, alignment_measure]]
    maxi = df[(df.feature_type == "MAX")][["dataset", metric, alignment_measure]]

    full_aggregate = (
        pc.merge(fp, on=["dataset", alignment_measure], suffixes=("_PC", "_FP"))
        .merge(pcfp, on=["dataset", alignment_measure], suffixes=("", "_PCFP"))
        .merge(meani, on=["dataset", alignment_measure], suffixes=("", "_MEAN"))
        .merge(maxi, on=["dataset", alignment_measure], suffixes=("", "_MAX"))
    )
    full_aggregate = full_aggregate.rename({"mcc": "mcc_PCFP"}, axis=1)
    test_full_aggregate = full_aggregate
    x = test_full_aggregate[alignment_measure].values
    order = np.argsort(x)
    x = x[order]
    res_y = {}
    n = filter_width
    for col in [col for col in test_full_aggregate.columns if "mcc" in col]:
        y = test_full_aggregate[col].values[order]
        means = np.convolve(y, [1 / n] * n, "valid")
        centered_squared = (y[n // 2 - 1 : -n // 2] - means) ** 2
        variance = np.convolve(centered_squared, [1 / n] * n, "same")
        std = np.sqrt(variance)
        res_y[col] = (means, std)
    alignment_cropped = x[n // 2 - 1 : -n // 2]
    all_x = []
    all_y = []
    for key, value in res_y.items():
        mean = value[0]
        std = value[1]
        all_x.extend(alignment_cropped)
        all_y.extend(mean)
        axs[0].plot(alignment_cropped, mean, label=key)
        axs[0].fill_between(alignment_cropped, mean + std, mean - std, alpha=0.2)
        all_y.extend(mean + std)
        all_y.extend(mean - std)

    x_ticks, y_ticks, x_min, x_max, y_min, y_max, delta_x, delta_y = get_all_ticks(
        all_x,
        all_y,
        n_spaces=n_spaces,
    )

    axs[0].set_xticks(x_ticks)
    axs[0].set_yticks(y_ticks)
    axs[0].set_xticklabels(x_ticks, fontsize=fontsize_small)
    axs[0].set_yticklabels(y_ticks, fontsize=fontsize_small)
    axs[0].set_xlim(x_min - 0 * delta_x / n_spaces, x_max + 0 * delta_x / n_spaces)
    axs[0].set_ylim(
        y_min - offset_scale * delta_y / n_spaces,
        y_max + offset_scale * delta_y / n_spaces,
    )

    axs[0].set_xlabel(r"Align($X_\text{PC}, X_\text{FP}$)", fontsize=fontsize)
    axs[0].set_ylabel(r"MCC", fontsize=fontsize)

    axs[0].xaxis.set_major_formatter(formatter)
    axs[0].yaxis.set_major_formatter(formatter)

    axs[0].legend()

    # plot 2

    x = test_full_aggregate[alignment_measure].values
    order = np.argsort(x)
    x = x[order]
    res_y = {}
    n = filter_width
    bootstraps = []

    for col in [col for col in test_full_aggregate.columns if "mcc" in col]:
        y = test_full_aggregate[col].values[order]
        alignment_cropped = x[n // 2 - 1 : -n // 2]
        means = np.convolve(y, [1 / n] * n, "valid")
        means = y
        alignment_cropped = x
        n_data = len(means)
        for bootstrap in range(1000):
            samples = np.random.choice(range(n_data), size=100, replace=True)
            x_boot = alignment_cropped[samples]
            y_boot = means[samples]
            slope, intercept, r_value, p_value, std_err = linregress(x_boot, y_boot)
            bootstraps.append(
                (bootstrap, col, slope, intercept, r_value, p_value, std_err)
            )

    res_boot = pd.DataFrame(
        bootstraps,
        columns=(
            "bootstrap",
            "col",
            "slope",
            "intercept",
            "r_value",
            "p_value",
            "std_err",
        ),
    )
    methods = res_boot.col.unique()
    heat = np.zeros((len(methods), len(methods)))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            heat[i, j] = symmetric_histogram_overlap_continuous(
                res_boot[res_boot.col == m1].slope,
                res_boot[res_boot.col == m2].slope,
            )
    ticklabels = [r"PC", r"FP", r"PCFP", r"MEAN", r"MAX"]
    sns.heatmap(
        heat,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        ax=axs[1],
        square=True,
        vmin=0,
        vmax=1,
    )

    # general stuff
    for n, ax in enumerate(axs.flat):
        ax
        ax.text(
            -0.15,
            1.015,
            "(" + string.ascii_lowercase[n] + ")",
            transform=ax.transAxes,  #
            size=fontsize,
            weight="bold",
        )

    plt.tight_layout()
    plt.savefig("performance_and_alignment.pdf")
    plt.show()


if __name__ == "__main__":
    df = pd.concat(
        [
            pd.read_csv(
                "/data/shared/exchange/mwelsch/fp_pc_align/results/performance_2025-05-12_15:00:13.csv"
            ),  # adme / hts / tox / tox21,
            pd.read_csv(
                "/data/shared/exchange/mwelsch/fp_pc_align/results/performance_2025-05-12_17:25:45.csv"
            ),  # toxcast / herg
        ]
    ).reset_index(drop=True)
    plot(df)
