# The objective is to develop a model to forecast electricity consumption in the buildings of IST.

# The deliverable of this project are:
# - a python file (py) or a python notebook (ipynb) that uses the raw data files  only.


# randomforrest - check
# regressionmodel - under work

from data_col import read_power_files, read_holiday_file,get_merged_data_frame,read_weather_file
from plots import get_plot, get_plot_2


import missingno as msno
from seaborn import heatmap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame
from sklearn.tree import export_graphviz
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import pydot


def plot_it(x, y, color, labelx, labely):

    fig, ax1 = plt.subplots(figsize=[30, 10])
    ax1.set_xlabel(labelx)
    ax1.set_ylabel(labely, color=color)
    ax1.plot(x, y, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.show()
    return


def plot_it2(x, y, color, labelx, labely):
    plt.figure(figsize=[20, 10])
    plt.scatter(x,y, color=color)
    plt
    plt.show()
    return

    missing = msno.bar(df_all)

    #Plot window setup
    plt.rcParams["figure.figsize"] = [15, 3.5]
    plt.rcParams["figure.autolayout"] = True

    #Print Figure 2
    plt.figure()
    x2 = df_all.index
    y2 = df_all['rain_day']
    plt.ylabel('Figure 2 Rain_day')
    plt.xlabel('df_all-Index')
    plt.plot(x2,y2)
    plt.show()

    return df_all


def analysis(df_all):
    # Create table with only weekends
    df_all['wknd_pwr'] = np.where(df_all['Day_nr'] >= 5, df_all['Power_kW'], None)
    # df_new.index = df_all.index
    df_weekend = pd.DataFrame(data=df_all['wknd_pwr'])
    df_weekend['Day_nr'] = df_all.index.dayofweek
    df_weekend['holiday'] = df_all['holiday']
    df_weekend['saturday'] = np.where(df_weekend['Day_nr'] == 5, df_weekend['wknd_pwr'], None)
    df_weekend['sunday'] = np.where(df_weekend['Day_nr'] == 6, df_weekend['wknd_pwr'], None)

    # create another year 2019
    datelist = pd.date_range('2019-01-01 00:00:00', periods=8760, freq='H', ).tolist()

    dates = {'Y2019': datelist}
    Datelist = pd.DataFrame(data=dates)
    Datelist.set_index('Y2019', inplace=True)
    df_weekend = pd.concat([df_weekend, Datelist])

    #Print figure 3
    plt.figure()
    plt.subplots(figsize=[20, 10])
    x2 = df_weekend.index
    y2 = df_weekend['wknd_pwr']
    plt.ylabel('Figure3_wknd_pwr')
    plt.xlabel('Figure 3, df_weekend_index')
    plt.plot(x2,y2)
    plt.show()

    return df_all, df_weekend

def analyse_data_all(df_all):
    # Simple offset forcasting
    forcast_offset = df_all.shift(periods=1)

    # Correlation matrix

    corr_mtx = df_all.corr()
    heatmap(corr_mtx,
            xticklabels=corr_mtx.columns,
            yticklabels=corr_mtx.columns,
            annot=True, fmt='.2f',
            cmap='Blues')


    random_F_generator(df_all)
    data_regression(df_all)


def data_regression(df_all):
    #This function is for making a linear regression analysis on the power consumption
    #At Alameda campus over 2 years
    df_all = df_all.fillna(0)

    #Initiate regression
    #available vars: temp_C, Power_kW. HR, solarRad_W/m2, rain_mm/h

    x = np.array(df_all['temp_C']).reshape((-1, 1))
    y = np.array(df_all['Power_kW'])
    x1 = np.array(df_all.index)
    #Create linear regression model & calculate the optimal values of the weights 𝑏₀ and 𝑏₁
    model = LinearRegression().fit(x, y)
    reg_score = model.score(x, y)
    print('coefficient of determination (R^2):', reg_score)
    print('intercept, b0 [scalar]:', model.intercept_)
    print('slope, b1 [array]:', model.coef_)

    #Use the model to predict a response
    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')
    y_pred = model.intercept_ + model.coef_ * x

    plot_it(x1,y, 'red', 'date','LR power_kW')
    plot_it(x1,y_pred,'blue','date','LR_prediction_power_kW')
    return


def random_F_generator(df_all):
    # Randomforrestestimate for the next day

    df_all_power_average = df_all['Power_kW'].sum() / len(df_all.index)
    df_all = df_all.fillna(0)

    df_all_backup_data_frame = df_all

    features = pd.get_dummies(df_all['Day_nr'])
    df_all = df_all.join(features)
    df_all = df_all.drop('Day_nr', axis=1)
    target = np.array(df_all['Power_kW'])

    df_all_labels_list = list(df_all.columns)
    df_all = np.array(df_all)

    train_features = df_all[0:10000]
    test_features = df_all
    train_labels = target[0:10000]
    test_labels = target

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # The baseline predictions are the historical averages
    baseline_preds = test_features[:, df_all_labels_list.index('Power_kW')]
    # Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    model = RandomForestRegressor(bootstrap=True,
                                  min_samples_leaf=1,
                                  n_estimators=20,
                                  min_samples_split=15,
                                  max_features='sqrt', max_depth=10)

    # Train the model on training data
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    # Try to print the data
    dataframe_again = pd.DataFrame(df_all)
    x_axis = dataframe_again.index
    y_axis = predictions
    y2_axis = dataframe_again[0]
    fig, ax1 = plt.subplots(figsize=[60, 10])

    color = 'y'
    ax1.set_xlabel('RandomForrestgenerator - Date')
    ax1.set_ylabel('Prediction - Power_kW', color=color)
    ax1.plot(x_axis, y_axis, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'g'
    ax2.set_ylabel('Power_kW', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis, y2_axis, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    print('NIICE')
    return


def main():

    #Read power data 2017 & 2018
    power = read_power_files()

    #Read holiday data
    holidays = read_holiday_file()

    #Merge power_1718 and holidays
    days_df = get_merged_data_frame(power,holidays)

    #Read weather_data
    weather = read_weather_file()

    #Merge to one DataFrame
    days_df = days_df.join(weather)

    #Plot power to temperature
    get_plot(days_df)

    #Plot power to temperature relationship
    get_plot_2(days_df)

    #power to temperature relationship

    #Comment: Chart to show correlation of Power and weather
    fig, ax1 = plt.subplots(figsize=[20, 10])

    color = 'pink'
    ax1.set_xlabel('Main5 - Date')
    ax1.set_ylabel('Power_kW', color=color)
    ax1.plot(x_axis, y_axis, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'grey'
    ax2.set_ylabel('Temp_C', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis, df_all['temp_C'].rolling(90).sum(), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'purple'
    ax2.set_ylabel('Temp_C', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis, df_all['temp_C'].rolling(90).sum(), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # UNCOMMENT ALL THE WAY DOWN HERE ^^^^
    df2 = analyse_data_all(df_all)
    df_list = analysis(df_all)
    df_all = df_list[0]
    df_new = df_list[1]

    print("main run successfully")
    print('END')


if __name__ == '__main__':
    main()

