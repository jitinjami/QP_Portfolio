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

if __name__ == '__main__':

    # define parameters
    list_of_n = [n for n in range(10, 500,10)]
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
    basic_df.to_csv('basics.csv')
    short_df.to_csv('short.csv')
    fixed_df.to_csv('fixed.csv')
    variable_df.to_csv('variable.csv')
    pass

