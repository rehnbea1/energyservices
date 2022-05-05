
from matplotlib import pyplot as plt

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

