import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clusterfock as cf

def get_dissociation(df):
    for col in ["CCD", "QCCD", "VCCD"]:
        df[col] = df[col] + df.FCI

    E0 = df.FCI.iloc[-1]
    df_diss = df.copy()
    
    for col in ["CCD", "QCCD", "VCCD", "FCI"]:
        df_diss[col] = df_diss[col] - E0

    return df_diss, E0

def plot_N2():
    df = pd.read_csv("vanvoorhis_headgordon/n2.txt", sep=" ", skiprows=5, header=0, index_col=False)
    
    df_diss, E0 = get_dissociation(df)
    
    fig, ax = plt.subplots()
    for col in ["CCD", "QCCD", "VCCD", "FCI"]:
        ax.scatter(df_diss.R, df_diss[col], label=col)

    ax.legend()
    plt.show()

if __name__ == '__main__':
    plot_N2()