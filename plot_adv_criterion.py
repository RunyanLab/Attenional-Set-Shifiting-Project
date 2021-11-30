##Extract and convert csv data for %over criterion

import numpy as np
from numpy import save
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv (r'/Users/acbandi/Desktop/behavior.csv')

def total(column2, start, stop): #column is in format 'Unnamed: 5'
    b = df[column2].iloc[start:stop].dropna()
    num_total = b.values
    for i in range (0, len(num_total)):
        num_total[i] = int(float(num_total[i]))
    return num_total 

start = 36
stop = 47
DU_1total = total('DU_1', start, stop)


##plot all variables 

def lineplot(total):
    f, (ax1) = plt.subplots(1, 1, figsize = (5,4))
    ax1.plot(DU_1total)
    ax1.set_title('% of trials over advancement criterion')
    ax1.set_ylabel('% of tirals over advancement criterion')
    ax1.set_xlabel('session #')
    f.tight_layout()

lineplot(total)