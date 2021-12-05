"""
This file is to test smaller things.
Please do not delete code snippets. If they hidder execution or increase runtime comment them
"""
# imports
from portfolio import Portfolio
import numpy as np

# main
if __name__ == '__main__':
    a  = np.eye(4)
    b = np.asarray([1,2,3,4])
    c = np.asarray([2,2,2,2])
    np.fill_diagonal(a,c+c-b)
    print(a)

    test = Portfolio(20, 5)
    print(np.sum(test.asset_weights))

def basic_markowitz_set_up(Portfolio p, Constraints weight_sum, int risk):
    P = p.covar_matrix
    q = -risk*p.asset_returns
    A = weight_sum.A
    b = weight_sum.b
    return P,q,A,b

def short_sales_setup(Portfolio p, Constraints weight_sum, Constraints short_sales, int risk):
    P = p.covar_matrix
    q = -risk*p.asset_returns
    A = weight_sum.A
    b = weight_sum.b
    G = short_sales.A
    h = short_sales.b
    return P,q,A,b,G,h