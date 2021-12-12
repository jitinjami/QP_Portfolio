from portfolio import Portfolio
from constraints import Constraints
from problem_formulations import *
import qpsolvers
import numpy as np
from numpy import array, dot
import time
import pandas as pd

def prep_basic(P,q,con:Constraints):
    P = np.dot(P.T,P)
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


list_of_n = [n for n in range(2,99)]
t = 0.1
k = 2
solvers = ['cvxopt','quadprog']

for n in list_of_n:
    for solver in solvers:
        #Basic Markowitz Model
        port = Portfolio(n, 5)
        P,q, con = basic_markowitz_set_up(port,t)
        P,q,G,h,A,b = prep_basic(P,q, con)
        #print(P)
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver)
        end = time.time()

        #Short sales Model
        port = Portfolio(n, 5)
        P,q, con = short_sales_setup(port,t)
        P,q,G,h,A,b = prep(P,q, con)
        #print(P)
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver)
        end = time.time()

        #Fixed transaction
        port = Portfolio(n, 5)
        P,q, con = fixed_transaction_cost_setup(port,t)
        P,q,G,h,A,b = prep(P,q, con)
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver)
        end = time.time()

        #Variable transaction
        port = Portfolio(n, 5)
        P,q, con = variable_transaction_cost_setup(port, t, k)
        P,q,G,h,A,b = prep(P,q, con)
        start = time.time()
        x = qpsolvers.solve_qp(P, q, G, h, A, b ,solver=solver)
        end = time.time()