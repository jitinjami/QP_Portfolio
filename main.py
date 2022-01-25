"""
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
    list_of_n = [n for n in range(10, 400,10)] # number of assets to be tested. If it takes too long reduce upper bound.
    t = 0.1     # risk factor
    k = 2       # number of buckets for wariable transaction cost
    m=10        # average time over m runs ( for short runtimes. for longer runtimes m is reduced)
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

    ### plot results ###
    plt.figure()
    basic_df.plot()
    plt.xlabel('Number of assets')
    plt.ylabel('Time')
    plt.title('Basic Markowitz Problem')
    plt.savefig('./data/basic_image.png')
    plt.show()

    plt.figure()
    short_df.plot()
    plt.xlabel('Number of assets')
    plt.ylabel('Time')
    plt.title('Short Sales Constraints')
    plt.savefig('./data/short_image.png')
    plt.show()

    plt.figure()
    fixed_df.plot()
    plt.xlabel('Number of assets')
    plt.ylabel('Time')
    plt.title('Fixed Transaction Costs')
    plt.savefig('./data/fixed_image.png')
    plt.show()

    plt.figure()
    variable_df.plot()
    plt.xlabel('Number of assets')
    plt.ylabel('Time')
    plt.title('Variable Transactions Costs')
    plt.savefig('./data/variable_image.png')
    plt.show()

