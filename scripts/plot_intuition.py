import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from scipy.stats import norm


def main():
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
    # plot 1
    x = np.arange(0,1,0.01)
    y1 = 0.5 * x + 0.5
    y2 = 0.8 * x + 0.3
    df = pd.DataFrame.from_dict({
        'x': np.concatenate([x, x]), 
        'y': np.concatenate([y1, y2]), 
        'type': np.concatenate([['single' for i in range(len(x))], ['ensemble' for i in range(len(x))]])
        }, 'index').T
    sns.lineplot(df, x='x', y='y', hue='type', ax=axs[0])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlabel('Align', fontsize=fontsize)
    axs[0].set_ylabel('Performance', fontsize=fontsize)
    

    # plot 2
    norm2 = norm(0.2, 0.01).rvs(10000)
    norm3 = norm(0.25, 0.01).rvs(10000)
    df = pd.DataFrame.from_dict({
        'x': np.concatenate([norm2, norm3]), 
        'type': np.concatenate([['single' for i in range(len(norm2))], ['ensemble' for i in range(len(norm2))]])
        }, 'index').T
    print(df)
    sns.histplot(df, x='x', hue='type', ax=axs[1], bins=100)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlabel('Slope', fontsize=fontsize)
    axs[1].set_ylabel('Count', fontsize=fontsize)



    plt.tight_layout()
    plt.savefig("performance_and_alignment.pdf")
    plt.show()

if __name__ == '__main__':
    main()