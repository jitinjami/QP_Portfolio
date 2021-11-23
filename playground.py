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