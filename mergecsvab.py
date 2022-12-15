from msilib.schema import Directory
from pandas.core.indexes.range import RangeIndex
import pandas as pd
import numpy as np
import os

PC = 20
width = 100
formatPrint = 'PC{col}.{row}'
columnName = ['class']
for i in range (PC):
    for j in range (width):
        columnName.append(formatPrint.format(col = i+1, row = j+1))
dataframe = pd.DataFrame(columns=columnName)

#directory dataset .csv
directory = 'C:/Users/M S I/csvdaunimprov1/' 
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        df = pd.read_csv(directory + filename, float_precision='round_trip')
        df.pop('Unnamed: 0')
        array = df.transpose().to_numpy().flatten()
        label = os.path.splitext(filename)[0]
        data = [label]
        dd = array.tolist()
        data = data + dd
        dataframe.loc[len(dataframe)] = data

dataframe.to_excel(r'C:/Users/M S I/csvdaunimprov1/mergedaunimprov1.xlsx',index=False)
