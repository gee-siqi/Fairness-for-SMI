from scipy.stats import gaussian_kde
import os
import numpy as np
import pandas as pd
from tools.utils import get_root_path


# generate simu_process frequency data
def kde_simu(cat, n, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    root_dir = get_root_path()
    data_dir = os.path.join(root_dir, f'data/{cat}1000.csv')
    df = pd.read_csv(data_dir)

    raw_freq = df['freq_m']
    kde = gaussian_kde(raw_freq, bw_method=0.2)
    new_freq = kde.resample(size=n)[0]
    a = new_freq
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
        root_dir = get_root_path()
        data_dir = os.path.join(root_dir, f'simu_process/freq_simu_{i}.npy')
        np.save(data_dir, simu_freq)


if __name__ == "__main__":
    categories = ['gaming']
    randon_seed = 41
    gen_simu(categories, 3000, randon_seed)
