"""
This file implements a regular Protfolio optimization with naive PQ algorythms
"""
# imports
# libraries
import numpy as np

# own scripts
from portfolio import Portfolio

# helper functions

# main
if __name__ == '__main__':
    #TODO: add argparse for convenient commandline usage

    # init variables
    mu_p = 5 # theoretical expected return of portfolio
    n = 10 # number of assets
    t = 0.5 # risk weiting factor
    l = np.ones(n)

    portfolio = Portfolio(n=n,portfolio_expected_return=mu_p)


    # check and sanitize variables
    assert t >= 0, "t weighs the risk. if set to 0 the lowest tolerance to risk is archieved. " \
                   "Thus negative values for t are not accepted"
    pass