# See https://pypi.org/project/pandas-profiling/ for deprecated pandas_profiling import ProfileReport
#   instead, use from ydata_profiling import ProfileReport" after 1Apr2023
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split

# Set a random seed to ensure reproducibility across runs
RNG_SEED = 42
np.random.seed(seed=RNG_SEED)

PATH = os.getcwd()
data_path = os.path.join(PATH, '../data/cp_data_cleaned.csv')

df = pd.read_csv(data_path)
print(f'Full DataFrame shape: {df.shape}')

X = df[['formula', 'T']]
y = df['Cp']

print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNG_SEED)

print(X_train.shape)
print(X_test.shape)

num_rows = len(X_train)
print(f'There are in total {num_rows} rows in the X_train DataFrame.')

# find the number of unique formulae: unique()
num_unique_formulae = len(X_train['formula'].unique())
print(f'But there are only {num_unique_formulae} unique formulae!\n')

print('Unique formulae and their number of occurances in the X_train DataFrame:')
print(X_train['formula'].value_counts(), '\n')
print('Unique formulae and their number of occurances in the X_test DataFrame:')
print(X_test['formula'].value_counts())

unique_formulae = X['formula'].unique()
print(f'{len(unique_formulae)} unique formulae:\n{unique_formulae}')

# Store a list of all unique formulae
all_formulae = unique_formulae.copy()

# Define the proportional size of the dataset split
val_size = 0.20
test_size = 0.10
train_size = 1 - val_size - test_size

# Calculate the number of samples in each dataset split
num_val_samples = int(round(val_size * len(unique_formulae)))
num_test_samples = int(round(test_size * len(unique_formulae)))
num_train_samples = int(round((1 - val_size - test_size) * len(unique_formulae)))

# Randomly choose the formulate for the validation dataset, and remove those from the unique formulae list
val_formulae = np.random.choice(all_formulae, size=num_val_samples, replace=False)
all_formulae = [f for f in all_formulae if f not in val_formulae]

# Randomly choose the formulate for the test dataset, and remove those from the unique formulae list
test_formulae = np.random.choice(all_formulae, size=num_test_samples, replace=False)
all_formulae = [f for f in all_formulae if f not in test_formulae]

# The remaining formulae will be used for the training dataset
train_formulae = all_formulae.copy()

print('Number of training formulae:', len(train_formulae))
print('Number of validation formulae:', len(val_formulae))
print('Number of testing formulae:', len(test_formulae))