import pandas as pd
import matplotlib.pyplot as plt

def plot1(df):
    print("DF1")
    print(df)
    plt.rcParams["figure.figsize"] = [15, 3.5]
    plt.rcParams["figure.autolayout"] = True

    headers = [index, "Power_kW"]

    x_axis = df.index
    y_axis = df["Power_kW"]
    plt.plot(x_axis,y_axis)
    plt.show()


def test_pandas(df1):
    print(df1[0].describe)
    print(df1[1].describe)
    dfa = df1[0]
    dfb = df1[1]
    dfc = pd.concat([dfb,dfa])
    print(dfc)
    dfd = df1[3]
    print("DFD", dfd)
    print("ok")
    print("DFA")
    print(dfa.describe())
    print("DFB")
    print(dfb.describe())
    print("DFC")
    print(dfc.describe())
    print(dfc.describe().head())
    print("DONE")

    dfc["Date_start"] = pd.to_datetime(dfc["Date_start"])
    delta = dfc["Date_start"].max() - dfc["Date_start"].min()
    print(delta)
    print("DONE2")

