"""
This file is to test smaller things.
Please do not delete code snippets. If they hidder execution or increase runtime comment them
"""
# imports
from portfolio import Portfolio
from constraints import Constraints
import numpy as np
from cvxopt import matrix, solvers

# main
if __name__ == '__main__':
    n=5
    k=2
    port = Portfolio(n, 5)

    #construct complete Constraints
    con = Constraints(port,(2*k+1)*n, k)
    con.add_weight_summation_constraint()
    con.add_short_sales_constraint()
    con.add_random_var_transaction_cost_equality()
    print("done")

