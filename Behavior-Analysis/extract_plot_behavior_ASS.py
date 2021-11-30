##Extract and convert csv data for trial accuracy %

import numpy as np
from numpy import save
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv (r'/Users/akhilbandi/Desktop/behavior.csv')

def total(column2, start, stop): #column is in format 'Unnamed: 5'
    b = df[column2].iloc[start:stop].dropna()
    num_total = b.values
    for i in range (0, len(num_total)):
        num_total[i] = int(num_total[i])
    return num_total 

def perc_correct(column1, column2, start, stop): #column is in format 'Unnamed: 5'
    a = df[column1].iloc[start:stop].dropna()
    num_correct = a.values
    b = df[column2].iloc[start:stop].dropna()
    num_total = b.values
    for i in range (0, len(num_correct)):
        num_correct[i] = int(num_correct[i])
    for i in range (0, len(num_total)):
        num_total[i] = int(num_total[i])
    return num_correct/num_total 

def perc_left(column_perc, start, stop): #column is in format 'Unnamed: 5'
    a = df[column_perc].iloc[start:stop].dropna()
    perc_left = a.values
    for i in range (0, len(perc_left)):
            perc_left[i] = float(perc_left[i])
    return perc_left/100

start = 26
stop = 32
EI2_total = total('Unnamed: 5', start, stop)
EI2_corr = perc_correct('Unnamed: 4', 'Unnamed: 5', start, stop)
EI2_left = perc_left('Unnamed: 6', start, stop)

EI3_total = total('Unnamed: 11', start, stop)
EI3_corr = perc_correct('Unnamed: 10', 'Unnamed: 11', start, stop)
EI3_left = perc_left('Unnamed: 12', start, stop)

EI4_total = total('Unnamed: 17', start, stop)
EI4_corr = perc_correct('Unnamed: 16', 'Unnamed: 17', start, stop)
EI4_left = perc_left('Unnamed: 18', start, stop)

##plot all variables 

def lineplot(total, corr, left):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (10,8))
    ax1.plot(total)
    ax1.set_title('# of trials')
    ax1.set_ylabel('# of trials')

    ax2.plot(corr)
    ax2.set_title('% correct')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('% correct')
    ax2.axhline(y = 0.8, color = 'k', linestyle = '--')

    ax3.plot(left)
    ax3.set_title('% left')
    ax3.set_ylim(0, 1)
    ax3.axhline(y = 0.5, color = 'k', linestyle = '--')
    ax3.set_ylabel('% left')
    ax3.set_xlabel('behavior session')
    f.tight_layout()

lineplot(EI3_total, EI3_corr, EI3_left)




