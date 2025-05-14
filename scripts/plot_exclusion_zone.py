import string

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import label
from scipy.stats import binomtest


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


def test_at_thresholds(x, y, x_percent, y_percent):
    x_thresh = np.percentile(x, x_percent * 100)
    y_thresh = np.percentile(y, y_percent * 100)
    exclude_zone = np.sum((x > x_thresh) & (y > y_thresh))
    expected_prob = (1 - x_percent) * (1 - y_percent)
    binom_stat = binomtest(exclude_zone, n=len(x), p=expected_prob, alternative="less")
    return x_thresh, y_thresh, binom_stat.pvalue


def plot(df):
    metric = "mcc"
    fontsize = 18
    fontsize_small = fontsize - 10
    n_spaces = 4
    offset_scale = 0.2
    formatter = ticker.StrMethodFormatter("{x:.2f}")
    int_formatter = ticker.FuncFormatter(lambda x, _: f"{int(x)}")
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # plot 1
    pc = df[(df.feature_type == "PC")]
    fp = df[(df.feature_type == "FP")]

    comparision = pc.merge(fp, on="dataset")
    comparision[f"delta_{metric}"] = (
        comparision[f"{metric}_y"] - comparision[f"{metric}_x"]
    )
    comparision[f"delta_{metric}_abs"] = np.abs(comparision[f"delta_{metric}"])
    comparision[f"delta_{metric}_rel"] = comparision[f"delta_{metric}_abs"] / np.abs(
        comparision[f"{metric}_y"]
    )

    comparision["delta_cka_abs"] = np.abs(
        comparision["alignment_pc_fp_x"] - comparision["full_alignment_pc_fp_x"]
    )

    sns.scatterplot(
        comparision,
        y=f"delta_{metric}_abs",
        x="full_alignment_pc_fp_x",
        hue=f"{metric}_y",
        ax=axs[0],
    )
    axs[0].legend(title=r"MCC$_\text{FP}$")

    x_ticks, y_ticks, x_min, x_max, y_min, y_max, delta_x, delta_y = get_all_ticks(
        comparision["full_alignment_pc_fp_x"].values,
        comparision[f"delta_{metric}_abs"].values,
        n_spaces,
    )

    axs[0].set_xticks(x_ticks)
    axs[0].set_yticks(y_ticks)
    axs[0].set_xticklabels(x_ticks, fontsize=fontsize_small)
    axs[0].set_yticklabels(y_ticks, fontsize=fontsize_small)
    axs[0].set_xlim(
        x_min - offset_scale * delta_x / n_spaces,
        x_max + offset_scale * delta_x / n_spaces,
    )
    axs[0].set_ylim(
        y_min - offset_scale * delta_y / n_spaces,
        y_max + offset_scale * delta_y / n_spaces,
    )

    axs[0].set_xlabel(r"Align($X_\text{PC}, X_\text{FP}$)", fontsize=fontsize)
    axs[0].set_ylabel(r"|MCC$_\text{PC}$ - MCC$_\text{FP}$|", fontsize=fontsize)

    axs[0].xaxis.set_major_formatter(formatter)
    axs[0].yaxis.set_major_formatter(formatter)

    # plot 2
    x = comparision["full_alignment_pc_fp_x"].values
    y = comparision[f"delta_{metric}_abs"].values

    original = []
    threshold = 0.05
    x = x[y != 0]
    y = y[y != 0]
    for x_percent in np.arange(0, 1, 0.01):
        for y_percent in np.arange(0, 1, 0.01):
            x_thresh, y_thresh, pvalue = test_at_thresholds(x, y, x_percent, y_percent)
            original.append((x_percent, y_percent, pvalue))
    original = pd.DataFrame(original, columns=["x_percent", "y_percent", "pvalue"])
    heatmap = original.pivot(
        index="x_percent", columns="y_percent", values="pvalue"
    ).values
    signif_mask = heatmap < threshold
    clusters, n_clusters = label(signif_mask)
    cluster_sizes = np.array([np.sum(clusters == i) for i in range(1, n_clusters + 1)])
    print(max(cluster_sizes))

    x_ticks, y_ticks, x_min, x_max, y_min, y_max, delta_x, delta_y = get_all_ticks(
        np.arange(0, 1, 0.01) * 100, np.arange(0, 1, 0.01) * 100, n_spaces
    )
    x_ticks = [np.ceil(tick) for tick in x_ticks]
    y_ticks = [np.ceil(tick) for tick in y_ticks]
    sns.heatmap(
        heatmap.T,
        ax=axs[1],
        xticklabels=x_ticks,
        yticklabels=y_ticks,
        square=True,
        cbar=True,
        vmin=0,
        vmax=1,
    )
    axs[1].invert_yaxis()

    axs[1].set_xticks(x_ticks)
    axs[1].set_yticks(y_ticks)
    axs[1].set_xticklabels(x_ticks, fontsize=fontsize_small)
    axs[1].set_yticklabels(y_ticks, fontsize=fontsize_small)

    axs[1].set_xlabel(r"Quantile Align", fontsize=fontsize)
    axs[1].set_ylabel(r"Quantile MCC", fontsize=fontsize)

    axs[1].xaxis.set_major_formatter(int_formatter)
    axs[1].yaxis.set_major_formatter(int_formatter)

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
    plt.savefig("performance_delta_alignment.pdf")
    plt.show()


if __name__ == "__main__":
    df = pd.concat(
        [
            pd.read_csv(
                "/data/shared/exchange/mwelsch/fp_pc_align/results/performance_2025-05-12_15:00:13.csv"
            ),  # adme / hts / tox / tox21
            pd.read_csv(
                "/data/shared/exchange/mwelsch/fp_pc_align/results/performance_2025-05-12_17:25:45.csv"
            ),  # toxcast / herg
        ]
    ).reset_index(drop=True)
    plot(df)
