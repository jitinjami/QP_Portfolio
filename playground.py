"""
This file is to test smaller things.
Please do not delete code snippets. If they hidder execution or increase runtime comment them
"""
# imports
from portfolio import Portfolio
import numpy as np

# main
if __name__ == '__main__':
    test = Portfolio(20, 5)
    print(np.sum(test.asset_weights))