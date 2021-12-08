"""
This file is to test smaller things.
Please do not delete code snippets. If they hidder execution or increase runtime comment them
"""
# imports
from portfolio import Portfolio
from constraints import Constraints
import numpy as np
from cvxopt import matrix, solvers
from problem_formulations import *
from optimizers import *

# main
if __name__ == '__main__':

    n=20
    k=3
    t = 0.1
    port = Portfolio(n, 5)

    ### to test comment the different problems in or out ###

    #C,c, con = basic_markowitz_set_up(port,t)
    #C,c, con = short_sales_setup(port, t)
    #C,c, con = fixed_transaction_cost_setup(port, t)
    C,c, con = variable_transaction_cost_setup(port, t, k)
    x, x_buy, x_sell = solve_qp(C, c, con)



    """
    #construct complete Constraints
    con = Constraints(port,(2*k+1)*n, k)
    con.add_weight_summation_constraint()
    con.add_short_sales_constraint()
    con.add_random_var_transaction_cost_equality()
    """
    print("done")

