# The objective is to develop a model to forecast electricity consumption in the buildings of IST.

# The deliverable of this project are:
# - a python file (py) or a python notebook (ipynb) that uses the raw data files  only.


# randomforrest - check
# regressionmodel - under work

from data_col import read_power_files, read_holiday_file,get_merged_data_frame,read_weather_file
from plots import get_plot, get_plot_2,get_plot_3,get_missingno,get_corr_matrix
from cluster import initial_cluster

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







def plot_it2(x, y, color, labelx, labely):
    plt.figure(figsize=[20, 10])
    plt.scatter(x,y, color=color)
    plt
    plt.show()
    return


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
    df_all['wknd_pwr'] = np.where(df_all['Day_nr'] >= 5, df_all['power_kw'], None)
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

def simple_offset_forcasting(days_df):
    # Simple offset forcasting
    forcast_offset = days_df.shift(periods=1)
    return


def data_regression(df_all):
    #This function is for making a linear regression analysis on the power consumption
    #At Alameda campus over 2 years
    df_all = df_all.fillna(0)

    #Initiate regression
    #available vars: temp_C, Power_kW. HR, solarRad_W/m2, rain_mm/h

    x = np.array(df_all['temp']).reshape((-1, 1))
    y = np.array(df_all['power_kw'])
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

    x = x1
    y = y
    y2 = y_pred
    color = 'red'
    labelx = "date"
    labely = 'LR power_kw'

    fig, ax1 = plt.subplots(figsize=[30, 10])
    ax1.set_xlabel(labelx)
    ax1.set_ylabel(labely, color=color)
    ax1.plot(x, y, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'blue'
    ax2.set_ylabel('LR_prediction_power_kw', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()

    return


def random_F_generator(df_all):
    # Randomforrestestimate for the next day

    df_all_power_average = df_all['power_kw'].sum() / len(df_all.index)
    df_all = df_all.fillna(0)

    df_all_backup_data_frame = df_all

    features = pd.get_dummies(df_all['Day_nr'])
    df_all = df_all.join(features)
    df_all = df_all.drop('Day_nr', axis=1)
    target = np.array(df_all['power_kw'])

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
    baseline_preds = test_features[:, df_all_labels_list.index('power_kw')]
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
    fig, ax1 = plt.subplots(figsize=[20, 10])

    color = 'y'
    ax1.set_xlabel('RandomForrestgenerator - Date')
    ax1.set_ylabel('Prediction - power_kw', color=color)
    ax1.plot(x_axis, y_axis, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'g'
    ax2.set_ylabel('power_kw', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis, y2_axis, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    print('randomforrestgenerator run successfully')
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

    #Analyse data with describe
    data_describe = days_df.describe
    data_info = days_df.info
    data_missing = days_df.isna().sum()

    #Missingno analysis
    get_missingno(days_df)

    #Get correlation matrix
    get_corr_matrix(days_df)
    #Figure 1 - power to temperature
    get_plot(days_df)

    #Figure 2 - power to temperature correlation
    get_plot_2(days_df)

    #Figure 3 - weather to index
    get_plot_3(days_df)

    #CLUSTERING

    #Cluster
    initial_cluster(days_df)

    #FORCASTING

    #Run random forrest generator
    random_F_generator(days_df)
    #Run Linear regression
    data_regression(days_df)
    #simple_offset_forcasting
    simple_offset_forcasting(days_df)


    df_list = analysis(days_df)
    df_all = df_list[0]
    df_new = df_list[1]

    print("main run successfully")
if __name__ == '__main__':
    main()

