import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label
from sklearn.utils import shuffle


def test_at_thresholds(x, y, x_percent, y_percent):
    x_thresh = np.percentile(x, x_percent * 100)
    y_thresh = np.percentile(y, y_percent * 100)
    exclude_zone = np.sum((x > x_thresh) & (y > y_thresh))
    expected_prob = (1 - x_percent) * (1 - y_percent)
    binom_stat = stats.binomtest(
        exclude_zone, n=len(x), p=expected_prob, alternative="less"
    )
    return x_thresh, y_thresh, binom_stat.pvalue


def run_permutation_test(df):
    metric = "mcc"
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

    x = comparision["full_alignment_pc_fp_x"].values
    y = comparision[f"delta_{metric}_abs"].values

    original = []
    threshold = 0.05
    n_simulations = 10000
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
    max_cluster = max(cluster_sizes)

    simuation_result = []
    for sim_nr in range(n_simulations):
        simulation = []
        for x_percent in np.arange(0, 1, 0.01):
            for y_percent in np.arange(0, 1, 0.01):
                y_shuffled = shuffle(y)
                x_thresh, y_thresh, pvalue = test_at_thresholds(
                    x, y_shuffled, x_percent, y_percent
                )
                simulation.append((x_percent, y_percent, pvalue))

        simulation = pd.DataFrame(
            simulation, columns=["x_percent", "y_percent", "pvalue"]
        )
        heatmap_sim = simulation.pivot(
            index="x_percent", columns="y_percent", values="pvalue"
        ).values
        signif_mask_sim = heatmap_sim < threshold
        cluster_sim, n_clusters_sim = label(signif_mask_sim)
        cluster_sizes_sim = np.array(
            [np.sum(cluster_sim == i) for i in range(1, n_clusters_sim + 1)]
        )
        simuation_result.append(max(cluster_sizes_sim))
    print(max_cluster)
    print((np.sum(max_cluster > simuation_result) + 1) / (n_simulations + 1))


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
    run_permutation_test(df)
