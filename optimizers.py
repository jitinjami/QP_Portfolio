"""
This file defines the different solvers
"""

# imports
import numpy as np
from cvxopt import matrix, solvers
from portfolio import Portfolio
from constraints import Constraints

# imported optimizers

def solve_qp(C,c,constraints: Constraints):
    C = matrix(C)
    c = matrix(c.T[0])

    if hasattr(constraints,'A') and not hasattr(constraints,'G'):
        A = matrix(constraints.A)
        b = matrix(constraints.b.T[0])
        sol = solvers.qp(C,c, A=A, b=b)
    if hasattr(constraints,'G') and not hasattr(constraints,'A'):
        G = matrix(constraints.G)
        h = matrix(constraints.h.T[0])
        sol = solvers.qp(C,c, G=G, h=h)
    if hasattr(constraints,'A') and hasattr(constraints,'G'):
        A = matrix(constraints.A)
        b = matrix(constraints.b.T[0])
        G = matrix(constraints.G)
        h = matrix(constraints.h.T[0])
        sol = solvers.qp(C,c, A=A, b=b, G=G, h=h)

    # decompose solution
    x_complete = np.asarray(sol['x'])
    x_buy = []
    x_sell = []
    # x no buys and sells
    if len(x_complete) == constraints.n_assets:
        x = x_complete
    # k = 1
    elif len(x_complete) == 3*constraints.n_assets:
        x = x_complete[0:constraints.n_assets]
        x_buy.append(x_complete[constraints.n_assets:constraints.n_assets*2])
        x_sell.append(x_complete[constraints.n_assets*2])
    # k > 1
    elif len(x_complete) == (2*constraints.k+1)*constraints.n_assets:
        x = x_complete[0:constraints.n_assets]
        for i in range(constraints.k):
            x_buy.append(x_complete[constraints.n_assets*(i+1):constraints.n_assets*(i+2)])
            x_sell.append(x_complete[constraints.n_assets*(i+1+constraints.k):constraints.n_assets*(i+2+constraints.k)])

    return x,x_buy,x_sell