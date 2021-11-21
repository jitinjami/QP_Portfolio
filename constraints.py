"""
This file defines constraints
"""

# imports
import numpy as np

from portfolio import Portfolio

# Equality constraints
"""
all weights have to add up to one. One equality constraint
transaction cost 1: the old value is the new value minus buys and plus sells
"""

# Inequality constraints
"""
short sales: all weights have to be positive. Results in n inequality constraints
transaction cost 2: buys and sells have to be positive but are bound from above. Results in 2n inequality constraints. 
"""

# Define Constraints
class Constraints:
    def __init__(self,
                input_porfolio: Portfolio):        
        """
        Initializes a constraint object which is used as a row in A and b
        in Ax=b
        """

    def weight_summation_constraint(self):
        #All weights must add up to 1
        A = np.ones(self.input_porfolio.n)
        b = 1
        return (A,b)

    def transaction_cost_equality(self,purchase_weight_vector,sale_weight_vector,original_asset_weights):
        #All purchase weights and sale weights must add up to the original asset weights
        assert len(self.input_porfolio.asset_weights) == len(sale_weight_vector), "sale_vector must be same as number of assets"
        assert len(self.input_porfolio.asset_weights) == len(purchase_weight_vector), "purchase_vector must be same as number of assets"
        assert len(self.input_porfolio.asset_weights) == len(original_asset_weights), "original_asset_weights must be same as number of assets"
        n = self.input_porfolio.n
        A = np.zeros(n,n)
        b = np.zeros(1,n)
        for i in range(len(self.input_porfolio.asset_weights)):
            A[i,i] = self.input_porfolio.asset_weights[i] + sale_weight_vector[i] - purchase_weight_vector[i]
            b[1,i] = original_asset_weights[i]
        return (A,b)

    def upper_bound_constraint_per_asset(self, transaction_weight, bound, asset_index):
        #Upper bound on all transaction weights (purchase or sale)
        n = self.input_porfolio.n
        A = np.zeros(n)
        A[asset_index] = transaction_weight
        b = np.array([bound])
        return (A,b)
    
    def lower_bound_constraint_per_asset(self, transaction_weight, asset_index):
        #Upper bound on all transaction weights (purchase or sale)
        n = self.input_porfolio.n
        A = np.zeros(n)
        A[asset_index] = -transaction_weight
        b = np.array([0])
        return (A,b)