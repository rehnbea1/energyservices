
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
from seaborn import heatmap

def get_plot(days_df):
    #Figure 1
    x = days_df.index
    y = days_df['power_kw']
    y2 = days_df['temp']
    color = 'grey'
    labelx = 'Figure 1: Power (grey) to Temperature (red) relationship'
    labely = 'temperature, C'
    fig, ax1 = plt.subplots(figsize=[30, 10])
    ax1.set_xlabel(labelx)
    ax1.set_ylabel(labely, color=color)
    ax1.plot(x, y, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'red'
    ax2.set_ylabel('Figure 1. Temperature', color=color)  # we already handled the x-label with ax1
    ax2.plot(x,y2.rolling(90).sum(), color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return


def get_plot_2(days_df):
    #Figure 2
    plt.figure(figsize=[20, 10])
    color = 'purple'
    labelx = 'Figure 2: Power in kW (x) to Temperature °C (y) relationship'
    labely = 'temperature, C'
    fig, ax1 = plt.subplots(figsize=[30, 10])
    plt.scatter(days_df['power_kw'], days_df['temp'], color=color)
    ax1.set_xlabel(labelx)
    ax1.set_ylabel(labely)
    plt.show()
    return


def get_plot_3(days_df):
    #Figure 3
    x = days_df.index
    y = days_df['temp']
    color = 'grey'
    labelx = 'Figure 3: Power (grey) to Temperature (red) relationship'
    labely = 'temperature, C'
    fig, ax1 = plt.subplots(figsize=[20, 10])
    ax1.set_xlabel(labelx)
    ax1.set_ylabel(labely, color=color)
    ax1.plot(x, y.rolling(10).sum(), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.show()
    return


def get_missingno(days_df):
    color = int('b20738',16)
    missing_bar = msno.bar(days_df,figsize=(12,7),color="dodgerblue", fontsize=8)
    missing_matrix = msno.matrix(days_df,color=(1, 0.38, 0.27))
    return


def get_corr_matrix(days_df):
    corr_mtx = days_df.corr()
    heatmap(corr_mtx,
            xticklabels=corr_mtx.columns,
            yticklabels=corr_mtx.columns,
            annot=True, fmt='.2f',
            cmap='Blues')
    return

