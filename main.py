# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

#The objective is to develop a model to forecast electricity consumption in the buildings of IST.
#To develop the model, the following data files have been collected:
#- 2017 and 2018 electricity consumption data in 4 buildings (central, civil, south tower and north tower);
#- list of holidays in Lisbon for 2017,2018, and 2019
#- weather data for the weather station in South tower for 2017 and 2018
#The deliverable of this project are:
#- a python file (py) or a python notebook (ipynb) that uses the raw data files  only.
import missingno as msno
from seaborn import heatmap
import numpy as np
import pandas as pd
from datetime import datetime



import matplotlib.pyplot as plt
import myFunctions

def read_file():

    file1 = '/Users/albertrehnberg/Downloads/IST_Central_Pav_2018_Ene_Cons.csv - IST_Central_Pav_2018_Ene_Cons.csv.csv'
    file2 = '/Users/albertrehnberg/Downloads/IST_Central_Pav_2017_Ene_Cons.csv - IST_Central_Pav_2017_Ene_Cons.csv.csv'
    file3 = '/Users/albertrehnberg/Downloads/holiday_17_18_19.csv - holiday_17_18_19.csv.csv'
    file4 = '/Users/albertrehnberg/Downloads/IST_meteo_data_2017_2018_2019.csv - IST_meteo_data_2017_2018_2019.csv.csv'

    #file1 = consumption year 2018
    #file2 = consumption y 2017
    #file3 = Holidays 2017,18,19
    #file4 weather data

    d_parser = lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M')
    e_parser = lambda x: datetime.strptime(x, '%d.%m.%Y')
    f_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


    #Read files and set correct date format
    #'Everything
    df_all = pd.read_csv(file1, parse_dates=["Date_start"], date_parser= d_parser)
    #2017
    power_data_2018 = pd.read_csv(file2, parse_dates=["Date_start"], date_parser=d_parser)

    #Holiday
    df3 = pd.read_csv(file3, parse_dates=['Date'], date_parser = e_parser)
    df4 = pd.read_csv(file4, parse_dates=['yyyy-mm-dd hh:mm:ss'], date_parser = f_parser)


    #merge df1&2 to one file - df
    df_all = pd.concat([df_all, power_data_2018], ignore_index=True)
    df_all.set_index('Date_start', inplace = True)
    df3.set_index('Date', inplace = True)
    df4.set_index('yyyy-mm-dd hh:mm:ss', inplace=True)

    #Resample weather data (df4) to have h-frequency. average is taken for everything except the rain:day, which is as maximum
    #Not sure if this is correct

    df4_resample = df4.resample('H',closed='left', label= 'right')['temp_C', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar',
       'solarRad_W/m2', 'rain_mm/h'].mean()
    df4_resample2 = df4.resample('H',closed='left', label='right')['rain_day'].mean()
    df4_resample = df4_resample.join(df4_resample2)
    df4_resample.sort_index(ascending=True)
    df_all = df_all.join(df4_resample, how= 'left')



    df_all['Day_nr'] = df_all.index.dayofweek

    # add holidays to df_all
    df_all["holiday"] = np.isin(df_all.index.date, df3.index.date)

    missing = msno.bar(df_all)



    #Plot window setup
   # plt.rcParams["figure.figsize"] = [15, 3.5]
   # plt.rcParams["figure.autolayout"] = True

    #Plot figure 1
    #plt.xlabel('Date')
    #plt.ylabel('Power in kW')
    #x_axis = df_all.index
    #y_axis = df_all["Power_kW"]
    #plt.plot(x_axis,y_axis)

    #Print figure 2
    #plt.figure()
    #x2 = df_all.index
    #y2 = df_all['rain_day']
    #plt.plot(x2,y2)
    #plt.show()


    print("nice")
    return (df_all)



def analysis(df_all):

    #Create table with only weekends
    df_all['wknd_pwr'] = np.where(df_all['Day_nr'] >= 5, df_all['Power_kW'],None)
    #df_new.index = df_all.index
    df_weekend = pd.DataFrame(data = df_all['wknd_pwr'])
    df_weekend['Day_nr'] = df_all.index.dayofweek
    df_weekend['holiday'] = df_all['holiday']
    df_weekend['saturday'] = np.where(df_weekend['Day_nr'] == 5, df_weekend['wknd_pwr'], None)
    df_weekend['sunday'] = np.where(df_weekend['Day_nr'] == 6, df_weekend['wknd_pwr'], None)

    #create another year 2019
    datelist = pd.date_range('2019-01-01 00:00:00', periods=8760, freq='H', ).tolist()

    dates = {'Y2019':datelist}
    Datelist = pd.DataFrame(data=dates)
    Datelist.set_index('Y2019', inplace=True)
    df_weekend = pd.concat([df_weekend,Datelist])

    #Print figure 2
    plt.figure()
    x2 = df_weekend.index
    y2 = df_weekend['wknd_pwr']
    plt.plot(x2,y2)
    plt.show()

    # Print figure 2
    plt.figure()
    x2 = df_weekend.index
    y2 = df_weekend['wknd_pwr']
    plt.plot(x2, y2)
    plt.show()






    return df_all, df_weekend


def analyse_data_all(df_all):
    # simple offset forcasting
    forcast_offset = df_all.shift(periods=1)


    #data_corr = abs(df_all.corr(method='pearson', min_periods=1))

    corr_mtx = df_all.corr()
    heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, fmt='.2f', cmap='Blues', )


    print('NIICE')
    return

def main():


    df_all = read_file()
    # Plot figure 1

    #Plot window setup
    x_axis = df_all.index
    y_axis = df_all["Power_kW"]
    y2_axis = df_all['temp_C']

    fig, ax1 = plt.subplots(figsize=[60, 10])

    color = 'y'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Power_kW', color=color)
    ax1.plot(x_axis, y_axis, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'g'
    ax2.set_ylabel('Temp_C', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis,y2_axis, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    #power to temperature relationship
    plt.figure(figsize=[20, 10])
    plt.scatter(df_all['Power_kW'],df_all['temp_C'])
    plt.show()

    plt.figure(figsize=[20, 10])
    plt.scatter(df_all.index,df_all['temp_C'], color = 'purple')
    plt.scatter(df_all.index,df_all['Power_kW'], color = 'r')
    plt.show()

    #Chart to show correlation of Power and weather
    fig, ax1 = plt.subplots(figsize=[20, 10])

    color = 'pink'
    ax1.set_xlabel('Date')
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



    df2 = analyse_data_all(df_all)

    df_list = analysis(df_all)
    df_all = df_list[0]
    df_new = df_list[1]

    # Initiating the class

    #Plot figure 1
    #plt.xlabel('Date')
    #plt.ylabel('Power in kW')
   # x_axis = df_all.index
  #  y_axis = df_all["Power_kW"]
 #   plt.plot(x_axis,y_axis, color = 'r')
#    plt.show()



    var = int(input("select function:"))
    if var==1:
        myFunctions.plot1(file[0])
    elif var == 4:
        myFunctions.test_pandas(file)

    print("ok")



if __name__ == '__main__':
    main()
