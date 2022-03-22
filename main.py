# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

#The objective is to develop a model to forecast electricity consumption in the buildings of IST.
#To develop the model, the following data files have been collected:
#- 2017 and 2018 electricity consumption data in 4 buildings (central, civil, south tower and north tower);
#- list of holidays in Lisbon for 2017,2018, and 2019
#- weather data for the weather station in South tower for 2017 and 2018
#The deliverable of this project are:
#- a python file (py) or a python notebook (ipynb) that uses the raw data files  only.

import numpy as np
import pandas as pd
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

    d_parser = lambda x: pd.datetime.strptime(x, '%d-%m-%Y %H:%M')
    e_parser = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')

    #Read files and set correct date format
    df1 = pd.read_csv(file1, parse_dates=["Date_start"], date_parser= d_parser)
    df2 = pd.read_csv(file2, parse_dates=["Date_start"], date_parser=d_parser)
    df3 = pd.read_csv(file3, parse_dates=['Date'], date_parser = e_parser)
    df4 = pd.read_csv(file4)

    #merge df1&2 to one file - df
    df = pd.concat([df1, df2], ignore_index=True)
    df.set_index('Date_start', inplace = True)
    df3.set_index('Date', inplace = True)

    #add holidays to df
    df['holiday'] = np.where(df.index.to_period('D').astype('datetime64[ns]').isin(df3), True, False)

    #start with df4
    df4


    print("nice")

    return


def main():
    file = read_file()


    print("Selections: 1,2,3 ")

    var = int(input("select function:"))
    if var==1:
        myFunctions.plot1(file[0])
    elif var==2:
        myFunctions.plot2(file[1])
    elif var == 3:
        myFunctions.plot3(file[2])

    elif var == 4:
        myFunctions.test_pandas(file)

    print("ok")


if __name__ == '__main__':
    main()
