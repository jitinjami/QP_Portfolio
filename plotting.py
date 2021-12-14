import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

basic_df = pd.read_csv('basics1.csv').drop([0])
basic_df = basic_df.set_index(basic_df.columns[0])
short_df = pd.read_csv('short1.csv').drop([0])
short_df = short_df.set_index(short_df.columns[0])
fixed_df = pd.read_csv('fixed1.csv').drop([0])
fixed_df = fixed_df.set_index(fixed_df.columns[0])
variable_df = pd.read_csv('variable1.csv').drop([0])
variable_df = variable_df.set_index(variable_df.columns[0])
plt.figure()
basic_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Basic Markowitz Problem')
plt.savefig('basics.png')

plt.figure()
short_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Short Sales Constraints')
plt.savefig('short.png')

plt.figure()
fixed_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Fixed Transaction Costs')
plt.savefig('fixed.png')

plt.figure()
variable_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Variable Transactions Costs')
plt.savefig('variable.png')