import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def plot_cumulative_returns(cum_returns):
    fig, ax = plt.subplots(figsize=(10, 6))  # fixed typo: subplts → subplots
    cum_returns.plot(ax=ax)
    ax.set_title("Cumulative Returns of Stocks")
    ax.set_ylabel("Growth Since Start")
    plt.show()

def plot_correlation_heatmap(corr_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Stock Returns")
    plt.show()