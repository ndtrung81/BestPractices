import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the ML_figures package and the figure-plotting functions
from ML_figures.figures import act_pred
from ML_figures.figures import residual, residual_hist
from ML_figures.figures import loss_curve
from ML_figures.figures import element_prevalence

# Read in example act vs. pred data
df_act_pred = pd.read_csv('ML_figures/example_data/act_pred.csv')
y_act, y_pred = df_act_pred.iloc[:, 1], df_act_pred.iloc[:, 2]

act_pred(y_act, y_pred,
         reg_line=True,
         save_dir='ML_figures/example_figures')

act_pred(y_act, y_pred,
         name='example_no_hist',
         x_hist=False, y_hist=False,
         reg_line=True,
         save_dir='ML_figures/example_figures')

residual(y_act, y_pred,
         save_dir='ML_figures/example_figures')

residual_hist(y_act, y_pred,
              save_dir='ML_figures/example_figures')

# Read in loss curve data
df_lc = pd.read_csv('ML_figures/example_data/training_progress.csv')
epoch = df_lc['epoch']
train_err, val_err = df_lc['mae_train'], df_lc['mae_val']

loss_curve(epoch, train_err, val_err,
           save_dir='ML_figures/example_figures')

# Visualize element prevalence
formula = df_act_pred.iloc[:, 0]

element_prevalence(formula,
                   save_dir='ML_figures/example_figures',
                   log_scale=False)
element_prevalence(formula,
                   save_dir='ML_figures/example_figures',
                   name='example_log',
                   log_scale=True)

plt.rcParams.update({'font.size': 12})
element_prevalence(formula,
                   save_dir='ML_figures/example_figures',
                   ptable_fig=False,
                   log_scale=False)
element_prevalence(formula,
                   save_dir='ML_figures/example_figures',
                   name='example_log',
                   ptable_fig=False,
                   log_scale=True)