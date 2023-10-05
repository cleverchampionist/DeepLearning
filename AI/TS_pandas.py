import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_data(input_file, index):
    input_data = np.loadtxt(input_file, delimiter= None)
    dates = pd.date_range('1950-01', periods = input_data.shape[0], freq = 'M')
    output = pd.Series(input_data[:, index], index = dates)
    return output

if __name__=='__main__':
    input_file = "./AO.txt"
    timeseries = read_data(input_file, index = 0)
    plt.figure()
    timeseries.plot()
    plt.show()
    