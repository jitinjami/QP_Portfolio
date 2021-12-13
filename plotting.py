import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

basic_df = pd.read_csv('basics.csv')
basic_df = basic_df.drop(basic_df.columns[0], axis=1)
short_df = pd.read_csv('short.csv')
short_df = short_df.drop(short_df.columns[0], axis=1)
fixed_df = pd.read_csv('fixed.csv')
fixed_df = fixed_df.drop(fixed_df.columns[0], axis=1)
variable_df = pd.read_csv('variable.csv')
variable_df = variable_df.drop(variable_df.columns[0], axis=1)
plt.figure()
basic_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Comparison of different QP algorithms')
plt.savefig('basics.png')

plt.figure()
short_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Comparison of different QP algorithms')
plt.savefig('short.png')

plt.figure()
fixed_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Comparison of different QP algorithms')
plt.savefig('fixed.png')

plt.figure()
variable_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Comparison of different QP algorithms')
plt.savefig('variable.png')