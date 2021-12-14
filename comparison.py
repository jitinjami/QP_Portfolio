from portfolio import Portfolio
from constraints import Constraints
from problem_formulations import *
import qpsolvers
import numpy as np
from numpy import array, dot
import time
import pandas as pd

def prep_basic(P,q,con:Constraints):
    #P = np.dot(P.T,P)
    q = q.reshape((n,))
    G = -np.eye(n)
    h = np.zeros(n).reshape((n,))
    A = con.A.reshape((n,))
    b = con.b.reshape((1,))
    return P,q,G,h,A,b

def prep(P,q,con:Constraints):
    #P = np.dot(P.T,P)
    q = q.reshape(len(q),)
    G = con.G
    h = con.h.reshape((len(con.h),))
    A = con.A
    b = con.b.reshape(len(con.b),)
    return P,q,G,h,A,b

n = 100
list_of_n = [n for n in range(2,n)]
t = 0.1
k = 2
solvers = ['cvxopt', 'ecos', 'osqp', 'quadprog', 'scs']
data = {'cvxopt':list_of_n, 'ecos':list_of_n, 'osqp':list_of_n, 'quadprog':list_of_n,'scs':list_of_n}
basic_df = pd.DataFrame(data=data, index=list_of_n)
short_df = pd.DataFrame(data=data,index=list_of_n)
fixed_df = pd.DataFrame(data=data,index=list_of_n)
variable_df = pd.DataFrame(data=data,index=list_of_n)
for i in range(len(list_of_n)):
    for j in range(len(solvers)):
        #Basic Markowitz Model
        n = list_of_n[i]
        solver = solvers[j]
        port = Portfolio(n, 5)
        P,q, con = basic_markowitz_set_up(port,t)
        P,q,G,h,A,b = prep_basic(P,q, con)
        #print(P)
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver)
        end = time.time()
        basic_df.iloc[i,j] = end - start
        #Short sales Model
        port = Portfolio(n, 5)
        P,q, con = short_sales_setup(port,t)
        P,q,G,h,A,b = prep(P,q, con)
        #print(P)
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver)
        end = time.time()
        short_df.iloc[i,j] = end - start

        #Fixed transaction
        port = Portfolio(n, 5)
        P,q, con = fixed_transaction_cost_setup(port,t)
        P,q,G,h,A,b = prep(P,q, con)
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver)
        end = time.time()
        fixed_df.iloc[i,j] = end - start

        #Variable transaction
        port = Portfolio(list_of_n[i], 5)
        P,q, con = variable_transaction_cost_setup(port, t, k)
        P,q,G,h,A,b = prep(P,q, con)
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver)
        end = time.time()
        variable_df.iloc[i,j] = end - start
#print(basic_df)
basic_df.to_csv('basics10.csv')
short_df.to_csv('short10.csv')
fixed_df.to_csv('fixed10.csv')
variable_df.to_csv('variable10.csv')
