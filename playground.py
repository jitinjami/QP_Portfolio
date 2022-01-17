"""
This file is to test smaller things. Please no not take anything for granted.
Please do not delete code snippets. If they hidder execution or increase runtime comment them
"""
# imports
from portfolio import Portfolio
from constraints import Constraints
from problem_formulations import *
from optimizers import *

# main
if __name__ == '__main__':
    active_set = []
    active_set.append(2)
    active_set.append(3)
    active_set.append(1)
    active_set.sort()

    n=20
    k=2
    t = 0.1
    port = Portfolio(n, 5)

    ### to test comment the different problems in or out ###

    #C,c, con = basic_markowitz_set_up(port,t)
    #C,c, con = short_sales_setup(port, t)


    #FIXME Have to declare an feasible port.asset_weights because we need that as a starting point
    #FIXME Have to declare an p', q', d and e as a numpy array. 
    #We can manually do random initialization of p',q',d and e before calling fixed_transaction_cost_setup instead of using
    #add_random_fixed_transaction_cost_equality function within fixed_transaction_cost_setup
    #C,c, con = fixed_transaction_cost_setup(port, t)


    #FIXME Have to declare an feasible port.asset_weights because we need that as a starting point
    #FIXME Have to declare an p', q', d and e as a numpy array. 
    #We can manually do random initialization of p',q',d and e before calling variable_transaction_cost_setup instead of using
    #add_random_var_transaction_cost_equality function within variable_transaction_cost_setup
    C,c, con = variable_transaction_cost_setup(port, t, k)

    x, x_buy, x_sell = solve_qp(C, c, con)
    print(x)
    print(x_buy)
    print(x_sell)

    """
    #construct complete Constraints
    con = Constraints(port,(2*k+1)*n, k)
    con.add_weight_summation_constraint()
    con.add_short_sales_constraint()
    con.add_random_var_transaction_cost_equality()
    """
    print("done")

