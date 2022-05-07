import pandas as pd
from sklearn.cluster import KMeans
from pandas import DataFrame, concat
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def initial_cluster(days_df):

    days_df_boolean = days_df[['holiday']]
    days_df_numeric = days_df.drop(['holiday'],axis=1)
    days_df_numeric = days_df_numeric.fillna(0)

    min_max_scaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(days_df_numeric)
    tmp_min_max = DataFrame(min_max_scaler.transform(days_df_numeric), index=days_df.index, columns=days_df_numeric.columns)
    df_scaled = concat([tmp_min_max, days_df_boolean], axis=1)

    kmeans = KMeans(n_clusters=3)
    df = DataFrame(df_scaled,columns=['power_kw','temp'])
    kmeans.fit(df)
    print(kmeans.cluster_centers_)
    y_kmeans = kmeans.fit_predict(df)

    df['clusters'] = kmeans.labels_

    plt.scatter(df['power_kw'],df['temp'], c=df['clusters'], s=50, cmap='viridis')
    plt.show()

    return
