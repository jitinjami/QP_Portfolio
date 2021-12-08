"""
This file defines the formulation of different Problems
"""

# imports
import numpy as np
from cvxopt import matrix, solvers
from portfolio import Portfolio
from constraints import Constraints

def basic_markowitz_set_up(p: Portfolio, weight_sum: Constraints, risk: int):
    P = p.covar_matrix
    q = -risk*p.asset_returns
    A = weight_sum.A
    b = weight_sum.b
    return P,q,A,b

def short_sales_setup( p: Portfolio, weight_sum: Constraints, short_sales: Constraints, risk: int):
    P = p.covar_matrix
    q = -risk*p.asset_returns
    A = weight_sum.A
    b = weight_sum.b
    G = short_sales.A
    h = short_sales.b
    return P,q,A,b,G,h


