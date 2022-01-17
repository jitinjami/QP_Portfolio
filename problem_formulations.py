"""
This file defines the formulation of different Problems.
They mainly differ in their constraints

For a more detailed description of the Simulator check constraints.py and portfolio.py

c and C originate from the standart formulation:
t cT x - 1/2 xT C x | G x <= h, A x = b
"""

# imports
import numpy as np
from cvxopt import matrix, solvers
from portfolio import Portfolio
from constraints import Constraints

def basic_markowitz_set_up(portfolio: Portfolio, t: float):
    """
    Generates the System Matrices and returns them alongside the constraint class which define a Optimization model.
    The matrices belong tho the Problem formulation: t cT x - 1/2 xT C x

    :param portfolio: Portfolio class
    :param t: Risk factor. If set to zero, the only the variance is minimized. Larger t allow for higher influence of
    the maximization of return
    :return: returns C,c,constraints
    """
    assert t < 1, f"t must be less than 1 but is {t}"
    C = portfolio.covar_matrix
    c = t * portfolio.asset_returns.reshape(portfolio.n, 1)
    constraints = Constraints(portfolio)
    constraints.add_weight_summation_constraint()
    return C,c,constraints

def short_sales_setup(portfolio: Portfolio, t: float):
    """
    Generates the System Matrices and returns them alongside the constraint class which define a Optimization model.
    The matrices belong tho the Problem formulation: t cT x - 1/2 xT C x

    :param portfolio: Portfolio class
    :param t: Risk factor. If set to zero, the only the variance is minimized. Larger t allow for higher influence of
    the maximization of return
    :return: returns C,c,constraints
    """
    assert t < 1, f"t must be less than 1 but is {t}"
    C = portfolio.covar_matrix
    c = t * portfolio.asset_returns.reshape(portfolio.n, 1)
    constraints = Constraints(portfolio)
    constraints.add_weight_summation_constraint()
    constraints.add_short_sales_constraint()
    return C,c,constraints

def fixed_transaction_cost_setup(portfolio: Portfolio, t: float):
    """
    Generates the System Matrices and returns them alongside the constraint class which define a Optimization model.
    The matrices belong tho the Problem formulation: t cT x - 1/2 xT C x

    :param portfolio: Portfolio class
    :param t: Risk factor. If set to zero, the only the variance is minimized. Larger t allow for higher influence of
    the maximization of return
    :return: returns C,c,constraints
    """
    assert t < 1, f"t must be less than 1 but is {t}"
    n_x_solution = 3 * portfolio.n
    constraints = Constraints(portfolio, n_res_vec=n_x_solution)
    constraints.add_weight_summation_constraint()
    constraints.add_short_sales_constraint()
    constraints.add_random_fixed_transaction_cost_equality()

    ## construct c
    mu = portfolio.asset_returns.reshape(portfolio.n, 1)
    # draw uniformly random slopes of transaction costs
    p = np.random.uniform(0.005, 0.05, portfolio.n).reshape(portfolio.n, 1) * (-1)
    q = np.random.uniform(0.005, 0.05, portfolio.n).reshape(portfolio.n, 1) * (-1)
    c = np.vstack((mu, p, q)) * (-t)

    ## construct Matrix C
    sigma = portfolio.covar_matrix
    C = np.zeros((n_x_solution,n_x_solution))
    C[0:portfolio.n, 0:portfolio.n] = sigma

    return C,c,constraints


def variable_transaction_cost_setup(portfolio: Portfolio, t: float, k: int):
    """
    Generates the System Matrices and returns them alongside the constraint class which define a Optimization model.
    The matrices belong tho the Problem formulation: t cT x - 1/2 xT C x

    :param portfolio: Portfolio class
    :param t: Risk factor. If set to zero, the only the variance is minimized. Larger t allow for higher influence of
    the maximization of return
    :param k: number of intervals for variable transaction costs
    :return: returns C,c,constraints
    """
    assert t < 1, f"t must be less than 1 but is {t}"
    assert k >= 1, f"k must be at least 1 but is {k}"
    n_x_solution = (2*k+1) * portfolio.n
    constraints = Constraints(portfolio, n_res_vec=n_x_solution, k=k)
    constraints.add_weight_summation_constraint()
    constraints.add_short_sales_constraint()
    constraints.add_random_var_transaction_cost_equality()

    ## construct c
    mu = portfolio.asset_returns.reshape(portfolio.n, 1)
    bounds = np.random.uniform(0.01 * (1 / portfolio.n), 0.5 * (1 / portfolio.n), k + 1)
    bounds[0] = 0.01 * (1 / portfolio.n)
    bounds[k] = 0.5 * (1 / portfolio.n)
    bounds.sort()
    # draw uniformly random slopes of intervals of transaction costs
    p = np.random.uniform(bounds[0], bounds[1], portfolio.n).reshape(portfolio.n, 1)
    q = np.random.uniform(bounds[0], bounds[1], portfolio.n).reshape(portfolio.n, 1)
    for i in range(1,k):
        p_tmp = np.random.uniform(bounds[i], bounds[i + 1], (portfolio.n))
        q_tmp = np.random.uniform(bounds[i], bounds[i + 1], (portfolio.n))
        p = np.vstack((p,p_tmp.reshape(portfolio.n,1)))
        q = np.vstack((q,q_tmp.reshape(portfolio.n,1)))

    c = np.vstack((mu,p*(-1),q*(-1))) * (-t)

    ## construct C
    sigma = portfolio.covar_matrix
    C = np.zeros((n_x_solution,n_x_solution))
    C[0:portfolio.n, 0:portfolio.n] = sigma

    return C,c,constraints






