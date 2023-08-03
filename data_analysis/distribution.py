import seaborn as sns
import matplotlib.pyplot as plt


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
