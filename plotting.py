# TODO: @Jitin: comments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
smoothing_factor = 0.1
pwd = os.path.abspath(os.getcwd())
basic_files = [file for file in sorted(glob.glob(pwd + "/data_sparse/basics*"))]
short_files = [file for file in sorted(glob.glob(pwd + "/data_sparse/short*"))]
fixed_files = [file for file in sorted(glob.glob(pwd + "/data_sparse/fixed*"))]
variable_files = [file for file in sorted(glob.glob(pwd + "/data_sparse/variable*"))]
basic_dfs = []
for i in range(len(basic_files)):
    seed_csv = basic_files[i]
    dummy_df = pd.read_csv(seed_csv).drop([0])
    dummy_df = dummy_df.set_index(dummy_df.columns[0])
    dummy_df = dummy_df.ewm(alpha=smoothing_factor).mean()
    basic_dfs.append(dummy_df)
basic_df = sum(basic_dfs)
plt.figure()
basic_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Basic Markowitz Problem')
plt.savefig('./data_sparse/basic_image_sparse.png')

short_dfs = []
for i in range(len(short_files)):
    seed_csv = short_files[i]
    dummy_df = pd.read_csv(seed_csv).drop([0])
    dummy_df = dummy_df.set_index(dummy_df.columns[0])
    dummy_df = dummy_df.ewm(alpha=smoothing_factor).mean()
    short_dfs.append(dummy_df)
short_df = sum(short_dfs)
plt.figure()
short_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Short Sales Constraints')
plt.savefig('./data_sparse/short_image_sparse.png')

fixed_dfs = []
for i in range(len(fixed_files)):
    seed_csv = fixed_files[i]
    dummy_df = pd.read_csv(seed_csv).drop([0])
    dummy_df = dummy_df.set_index(dummy_df.columns[0])
    dummy_df = dummy_df.ewm(alpha=smoothing_factor).mean()
    fixed_dfs.append(dummy_df)
fixed_df = sum(fixed_dfs)
plt.figure()
fixed_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Fixed Transaction Costs')
plt.savefig('./data_sparse/fixed_image_sparse.png')

variable_dfs = []
for i in range(len(variable_files)):
    seed_csv = variable_files[i]
    dummy_df = pd.read_csv(seed_csv).drop([0])
    dummy_df = dummy_df.set_index(dummy_df.columns[0])
    dummy_df = dummy_df.ewm(alpha=smoothing_factor).mean()
    variable_dfs.append(dummy_df)
variable_df = sum(variable_dfs)
plt.figure()
variable_df.plot()
plt.xlabel('Number of assets')
plt.ylabel('Time')
plt.title('Variable Transactions Costs')
plt.savefig('./data_sparse/variable_image_sparse.png')