import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from tools.utils import get_root_path, path
# from tools.distribution import kde_simu

"""
def plot_distribution(df, upto=10000):
    # Plot the distribution of the "frequency" column using a histogram
    # remove extreme values
    df_filtered = df[df.freq_m < upto]
    sns.histplot(df_filtered['freq_m'], kde=True)

    # Optional: Add labels and title
    plt.xlabel('Frequency')
    plt.ylabel('Count')
    plt.title('Distribution of Frequency')

    # Show the plot
    plt.show()



# plot frequency and subscribers
def plot_freq_sub(df, sub_upto=60000000, freq_upto=200):
    filtered_df = df[(df['subscribers'] < sub_upto) & (df['freq_m'] < freq_upto)]
    # Create a scatter plot of 'freq_m' vs. 'subscribers' using the filtered DataFrame
    sns.scatterplot(x='freq_m', y='subscribers', data=filtered_df)

    # Optional: Add labels and title
    plt.xlabel('Frequency (freq_m)')
    plt.ylabel('Subscribers')
    plt.title('Relation between Frequency and Subscribers (Filtered)')

    # Show the plot
    plt.show()
    """


# generate simu_process frequency data
def kde_simu(cat, n, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    root_dir = get_root_path()
    data_dir = path(root_dir, f'data/{cat}1000.csv')
    df = pd.read_csv(data_dir)

    raw_freq = df['freq_m']
    kde = gaussian_kde(raw_freq, bw_method=0.2)
    new_freq = kde.resample(size=n)[0]
    a = new_freq
    # min_value = 0
    # max_value = 1000
    # new_freq = np.clip(new_freq, min_value, max_value)
    return new_freq


def gen_simu(cls, n, random_seed=None):
    """
    Generate simulations of frequency of give categories
    @param random_seed:
    @param n: Number of frequencies needed for each category
    @param cls: List of categories to be generated
    @return: save the frequency data to npy files
    """
    for i in cls:
        simu_freq = kde_simu(i, n, random_seed=random_seed)
        np.save(f'freq_simu_{i}.npy', simu_freq)


if __name__ == "__main__":
    categories = ['gaming']
    randon_seed = 41
    gen_simu(categories, 3000, randon_seed)
