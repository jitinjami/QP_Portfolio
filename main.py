"""
TODO: @Jitin: could you please revisit this file agian?
this file is meant to contain the function calls and routienes for runtime measurement
"""
# imports
## library imports
import numpy as np

## own imports
from portfolio import Portfolio
from constraints import Constraints
from problem_formulations import *
from optimizers import *
import pandas as pd
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # define parameters
    list_of_n = [n for n in range(10, 400,10)]
    t = 0.1
    k = 2
    m=10
    data = {'non-sparse': list_of_n, 'sparse': list_of_n}
    basic_df = pd.DataFrame(data=data, index=list_of_n)
    short_df = pd.DataFrame(data=data, index=list_of_n)
    fixed_df = pd.DataFrame(data=data, index=list_of_n)
    variable_df = pd.DataFrame(data=data, index=list_of_n)
    for i,n in enumerate(list_of_n):
        # Basic Markowitz Model
        print(f"n: {n}")
        port = Portfolio(n, 5)
        P, q, con = basic_markowitz_set_up(port, t)
        # print(P)
        runtime = 0
        for j in range(m):
            x,x_buy,x_sell,rt = solve_qp(P,q,con)
            runtime +=rt
        basic_df.iloc[i, 0] = runtime/m

        # Short sales Model
        port = Portfolio(n, 5)
        P, q, con = short_sales_setup(port, t)
        # print(P)
        runtime = 0
        for j in range(m):
            x,x_buy,x_sell,rt = solve_qp(P,q,con)
            runtime +=rt
        short_df.iloc[i, 0] = runtime/m

        # Fixed transaction
        port = Portfolio(n, 5)
        P, q, con = fixed_transaction_cost_setup(port, t)
        runtime = 0
        for j in range(m):
            x,x_buy,x_sell,rt = solve_qp(P,q,con)
            runtime +=rt
        fixed_df.iloc[i, 0] = runtime/m

        # Variable transaction
        port = Portfolio(list_of_n[i], 5)
        P, q, con = variable_transaction_cost_setup(port, t, k)
        runtime = 0
        for j in range(m):
            x,x_buy,x_sell,rt = solve_qp(P,q,con)
            runtime +=rt
        variable_df.iloc[i, 0] = runtime/m
        if runtime > 1 and m>1:
            m-=1
            print(f"m: {m}")


        # Basic Markowitz Model
        port = Portfolio(n, 5)
        P, q, con = basic_markowitz_set_up(port, t)
        # print(P)
        runtime = 0
        for j in range(m):
            x, x_buy, x_sell, rt = solve_sparse_cone_qp(P, q, con)
            runtime += rt
        basic_df.iloc[i, 1] = runtime / m

        # Short sales Model
        port = Portfolio(n, 5)
        P, q, con = short_sales_setup(port, t)
        # print(P)
        runtime = 0
        for j in range(m):
            x, x_buy, x_sell, rt = solve_sparse_cone_qp(P, q, con)
            runtime += rt
        short_df.iloc[i, 1] = runtime / m

        # Fixed transaction
        port = Portfolio(n, 5)
        P, q, con = fixed_transaction_cost_setup(port, t)
        runtime = 0
        for j in range(m):
            x, x_buy, x_sell, rt = solve_sparse_cone_qp(P, q, con)
            runtime += rt
        fixed_df.iloc[i, 1] = runtime / m

        # Variable transaction
        port = Portfolio(list_of_n[i], 5)
        P, q, con = variable_transaction_cost_setup(port, t, k)
        runtime = 0
        for j in range(m):
            x, x_buy, x_sell, rt = solve_sparse_cone_qp(P, q, con)
            runtime += rt
        variable_df.iloc[i, 1] = runtime / m
    basic_df.to_csv('./data_sparse/basics.csv')
    short_df.to_csv('./data_sparse/short.csv')
    fixed_df.to_csv('./data_sparse/fixed.csv')
    variable_df.to_csv('./data_sparse/variable.csv')

    plt.figure()
    basic_df.plot()
    plt.xlabel('Number of assets')
    plt.ylabel('Time')
    plt.title('Basic Markowitz Problem')
    plt.savefig('./data_sparse/basic_image.png')

    plt.figure()
    short_df.plot()
    plt.xlabel('Number of assets')
    plt.ylabel('Time')
    plt.title('Short Sales Constraints')
    plt.savefig('./data_sparse/short_image.png')
    plt.show()
    plt.figure()
    fixed_df.plot()
    plt.xlabel('Number of assets')
    plt.ylabel('Time')
    plt.title('Fixed Transaction Costs')
    plt.savefig('./data_sparse/fixed_image.png')
    plt.show()
    plt.figure()
    variable_df.plot()
    plt.xlabel('Number of assets')
    plt.ylabel('Time')
    plt.title('Variable Transactions Costs')
    plt.savefig('./data_sparse/variable_image.png')

