from portfolio import Portfolio
from constraints import Constraints
from problem_formulations import *
import qpsolvers
import numpy as np
from numpy import array, dot
import time
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def prep_basic(P,q,con:Constraints):
    """
    Preparing the necessary matrices of basic formulation such that qpsolvers
    can read it in its respective format.
    """
    #P = np.dot(P.T,P)
    q = q.reshape((n,))
    G = -np.eye(n)
    h = np.zeros(n).reshape((n,))
    A = con.A.reshape((n,))
    b = con.b.reshape((1,))
    return P,q,G,h,A,b

def prep(P,q,con:Constraints):
    """
    Preparing the necessary matrices of short sales, fixed cost, variable cost
    formulation such that qpsolvers library can read it in its respective format.
    """
    #P = np.dot(P.T,P)
    q = q.reshape(len(q),)
    G = con.G
    h = con.h.reshape((len(con.h),))
    A = con.A
    b = con.b.reshape(len(con.b),)
    return P,q,G,h,A,b

n = 100
list_of_n = [n for n in range(2,n)] #List of number of assets
t = 0.1
k = 2
seeds = 10

for seed in range(1,seeds+1):
    solvers = ['cvxopt','quadprog', 'ecos'] #Different solvers available on qpsolvers
    data = {'cvxopt':list_of_n,'quadprog':list_of_n, 'ecos':list_of_n}
    basic_df = pd.DataFrame(data=data, index=list_of_n)
    short_df = pd.DataFrame(data=data,index=list_of_n)
    for i in range(len(list_of_n)):
        for j in range(len(solvers)):
            #Basic Markowitz Model
            n = list_of_n[i]
            solver = solvers[j]
            port = Portfolio(n, 5)
            P,q, con = basic_markowitz_set_up(port,t) #Problem Set up
            P,q,G,h,A,b = prep_basic(P,q, con) #Prepping the matrices for qpsolvers 
            start = time.time()
            x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver) #Solving the problem
            end = time.time()
            basic_df.iloc[i,j] = end - start

            #Short sales Model
            port = Portfolio(n, 5)
            P,q, con = short_sales_setup(port,t) #Problem Set up
            P,q,G,h,A,b = prep(P,q, con) #Prepping the matrices for qpsolvers 
            start = time.time()
            x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver) #Solving the problem
            end = time.time()
            short_df.iloc[i,j] = end - start


    data = {'cvxopt':list_of_n}
    fixed_df = pd.DataFrame(data=data, index=list_of_n)
    variable_df = pd.DataFrame(data=data,index=list_of_n)
    for i in range(len(list_of_n)):
        #Fixed transaction
        n = list_of_n[i]
        solver = solvers[j]
        port = Portfolio(n, 5)
        P,q, con = fixed_transaction_cost_setup(port,t) #Problem Set up
        P,q,G,h,A,b = prep(P,q, con) #Prepping the matrices for qpsolvers
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver='cvxopt') #Solving the problem
        end = time.time()
        fixed_df.iloc[i,:] = end - start

        #Variable transaction
        port = Portfolio(n, 5)
        P,q, con = variable_transaction_cost_setup(port, t, k) #Problem Set up
        P,q,G,h,A,b = prep(P,q, con) #Prepping the matrices for qpsolvers
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver='cvxopt') #Solving the problem
        end = time.time()
        variable_df.iloc[i,:] = end - start

    basic_df.to_csv('./data/basics{}.csv'.format(seed))
    short_df.to_csv('./data/short{}.csv'.format(seed))
    fixed_df.to_csv('./data/fixed{}.csv'.format(seed))
    variable_df.to_csv('./data/variable{}.csv'.format(seed))

#Plotting Section
smoothing_factor = 0.1
pwd = os.path.abspath(os.getcwd())
basic_files = [file for file in sorted(glob.glob(pwd + "/data/basics*"))]
short_files = [file for file in sorted(glob.glob(pwd + "/data/short*"))]
fixed_files = [file for file in sorted(glob.glob(pwd + "/data/fixed*"))]
variable_files = [file for file in sorted(glob.glob(pwd + "/data/variable*"))]
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
plt.savefig('./data/basic_image.png')

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
plt.savefig('./data/short_image.png')

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
plt.savefig('./data/fixed_image.png')

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
plt.savefig('./data/variable_image.png')
