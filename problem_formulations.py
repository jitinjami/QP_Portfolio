"""
This file defines the formulation of different Problems
"""

# imports
import numpy as np
from cvxopt import matrix, solvers
from portfolio import Portfolio
from constraints import Constraints

def basic_markowitz_set_up(portfolio: Portfolio, t: float):
    assert t < 1, f"t must be less than 1 but is {t}"
    C = portfolio.covar_matrix
    c = t * portfolio.asset_returns.reshape(portfolio.n, 1)
    constraints = Constraints(portfolio)
    constraints.add_weight_summation_constraint()
    return C,c,constraints

def short_sales_setup(portfolio: Portfolio, t: float):
    assert t < 1, f"t must be less than 1 but is {t}"
    C = portfolio.covar_matrix
    c = t * portfolio.asset_returns.reshape(portfolio.n, 1)
    constraints = Constraints(portfolio)
    constraints.add_weight_summation_constraint()
    constraints.add_short_sales_constraint()
    return C,c,constraints

def fixed_transaction_cost_setup(portfolio: Portfolio, t: float):
    assert t < 1, f"t must be less than 1 but is {t}"
    n_x_solution = 3 * portfolio.n
    constraints = Constraints(portfolio, n_res_vec=n_x_solution)
    constraints.add_weight_summation_constraint()
    constraints.add_short_sales_constraint()
    constraints.add_random_fixed_transaction_cost_equality()
    sigma = portfolio.covar_matrix
    mu = portfolio.asset_returns.reshape(portfolio.n, 1)
    p = np.random.uniform(0.005, 0.05, portfolio.n).reshape(portfolio.n, 1) * (-1)
    q = np.random.uniform(0.005, 0.05, portfolio.n).reshape(portfolio.n, 1) * (-1)
    c = np.vstack((mu, p, q)) * (t)
    C = np.eye(n_x_solution)
    C[0:portfolio.n, 0:portfolio.n] = sigma

    return C,c,constraints


def variable_transaction_cost_setup(portfolio: Portfolio, t: float, k: int):
    assert t < 1, f"t must be less than 1 but is {t}"
    assert k >= 1, f"k must be at least 1 but is {k}"
    n_x_solution = (2*k+1) * portfolio.n
    constraints = Constraints(portfolio, n_res_vec=n_x_solution, k=k)
    constraints.add_weight_summation_constraint()
    constraints.add_short_sales_constraint()
    constraints.add_random_var_transaction_cost_equality()
    sigma = portfolio.covar_matrix
    mu = portfolio.asset_returns.reshape(portfolio.n, 1)
    bounds = np.random.uniform(0.01 * (1 / portfolio.n), 0.5 * (1 / portfolio.n), k + 1)
    bounds[0] = 0.01 * (1 / portfolio.n)
    bounds[k] = 0.5 * (1 / portfolio.n)
    bounds.sort()
    p = np.random.uniform(bounds[0], bounds[1], portfolio.n).reshape(portfolio.n, 1)
    q = np.random.uniform(bounds[0], bounds[1], portfolio.n).reshape(portfolio.n, 1)
    for i in range(1,k):
        p_tmp = np.random.uniform(bounds[i], bounds[i + 1], (portfolio.n))
        q_tmp = np.random.uniform(bounds[i], bounds[i + 1], (portfolio.n))
        p = np.vstack((p,p_tmp.reshape(portfolio.n,1)))
        q = np.vstack((q,q_tmp.reshape(portfolio.n,1)))

    c = np.vstack((mu,p*(-1),q*(-1))) * (-t)
    C = np.eye(n_x_solution)
    C[0:portfolio.n, 0:portfolio.n] = sigma

    return C,c,constraints






