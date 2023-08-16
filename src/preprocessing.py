# See https://pypi.org/project/pandas-profiling/ for deprecated pandas_profiling import ProfileReport
#   instead, use from ydata_profiling import ProfileReport" after 1Apr2023
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ydata_profiling import ProfileReport

PATH = os.getcwd()
data_path = os.path.join(PATH, '../data/cp_data_demo.csv')

df = pd.read_csv(data_path)
print(f'Original DataFrame shape: {df.shape}')
#print(df.head(10))
#print(df.describe())

# Jupyter notebook
#profile = ProfileReport(df.copy(), title='Pandas Profiling Report of Cp dataset', html={'style':{'full_width':True}})
#profile.to_widgets()

print(df.columns)
rename_dict = {'FORMULA': 'formula',
               'CONDITION: Temperature (K)': 'T',
               'PROPERTY: Heat Capacity (J/mol K)': 'Cp'}
df = df.rename(columns=rename_dict)
print(df.columns)

df2 = df.copy()
bool_nans_formula = df2['formula'].isnull()
bool_nans_T = df2['T'].isnull()
bool_nans_Cp = df2['Cp'].isnull()

df2 = df2.drop(df2.loc[bool_nans_formula].index, axis=0)
df2 = df2.drop(df2.loc[bool_nans_T].index, axis=0)
df2 = df2.drop(df2.loc[bool_nans_Cp].index, axis=0)

print(f'DataFrame shape before dropping NaNs: {df.shape}')
print(f'DataFrame shape after dropping NaNs: {df2.shape}')

df3 = df.copy()
df3.dropna(axis=0, how='any')
df = df3.copy()

out_path = os.path.join(PATH, '../data/cp_data_cleaned.csv')
df.to_csv(out_path, index=False)