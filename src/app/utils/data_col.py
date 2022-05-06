import pandas as pd
import numpy as np
from datetime import datetime



def read_power_files():
    # This file is for reading power data from 2018 and 2017

    file1 = '/Users/albertrehnberg/Desktop/energyservices/src/app/data/IST_Central_Pav_2018_Ene_Cons.csv - IST_Central_Pav_2018_Ene_Cons.csv.csv'
    file2 = '/Users/albertrehnberg/Desktop/energyservices/src/app/data/IST_Central_Pav_2017_Ene_Cons.csv - IST_Central_Pav_2017_Ene_Cons.csv.csv'

    d_parser = lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M')

    # Read files and set correct date format
    consumption18 = pd.read_csv(file1, parse_dates=["Date_start"], date_parser=d_parser)
    consumption17 = pd.read_csv(file2, parse_dates=["Date_start"], date_parser=d_parser)

    power_1718 = pd.concat([consumption18,consumption17])
    power_1718 = power_1718.set_index('Date_start')
    power_1718 = power_1718.rename({'Power_kW':'power_kw'},axis=1)

    return power_1718


def read_holiday_file():
    #This file is for reading holiday data

    file = '/Users/albertrehnberg/Desktop/energyservices/src/app/data/holiday_17_18_19.csv - holiday_17_18_19.csv.csv'

    parser = lambda x: datetime.strptime(x, '%d.%m.%Y')
    holidays = pd.read_csv(file, parse_dates=['Date'], date_parser=parser)

    #Upsample data to be per-hour
    holidays_original = holidays
    holidays = holidays.rename({'Date':'date','Holiday':'holiday'},axis=1)
    holidays = holidays.set_index('date')
    #holidays = holidays.resample("H").ffill()
    #holidays = holidays.set_index('date')

    return holidays


def get_merged_data_frame(power_1718, holidays):
    #This function is used to merge holidays with the power data

    days_df = power_1718.join(holidays, sort=False)
    days_df["holiday"] = np.isin(days_df.index.date, holidays.index.date)
    days_df['Day_nr'] = days_df.index.dayofweek
    #holiday_dummies = pd.get_dummies(days_df['holiday'])
    #days_df = days_df.join(holiday_dummies)
    #days_df = days_df.rename({'False': 'holiday_no','True':'holiday_yes'},axis=1)
    #days_df = days_df.drop('holiday', axis=1)

    return days_df


def read_weather_file():
    #This function is for reading the weather data

    file = '/Users/albertrehnberg/Desktop/energyservices/src/app/data/IST_meteo_data_2017_2018_2019.csv - IST_meteo_data_2017_2018_2019.csv.csv'

    parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    weather = pd.read_csv(file, parse_dates = ['yyyy-mm-dd hh:mm:ss'], date_parser = parser)
    weather = weather.set_index('yyyy-mm-dd hh:mm:ss')
    weather_original = weather
    weather_resample1 = weather.resample('H', closed='left', label='right')['temp_C', 'HR', 'windSpeed_m/s', 'windGust_m/s',
                                                                  'pres_mbar','solarRad_W/m2', 'rain_mm/h'].mean()
    rain_per_hour = weather['rain_mm/h'].groupby(pd.Grouper(freq='H')).mean().dropna(how='all')
    rain_per_day = weather['rain_day'].groupby(pd.Grouper(freq='d')).mean().dropna(how='all')

    weather = weather_resample1.join(rain_per_day)
    weather = weather.join(rain_per_hour,how = 'left', lsuffix = '_left', rsuffix = '_right')
    weather = weather.drop({'rain_mm/h_left'}, axis=1)

    weather = weather.rename({'temp_C':'temp','rain_mm/h_right':'rain_mm_per_hour'},axis=1)
    weather.sort_index(ascending=True)

    return weather