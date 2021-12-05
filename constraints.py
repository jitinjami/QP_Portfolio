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

        :param input_porfolio: portfolio the constraints are defined for
        """
        self.input_porfolio = input_porfolio

    def weight_summation_constraint(self):
        """
        Defines the part of Constrain matrix Ax = b which ensures that all weights add up to 1
        :return: A, b
        """

        # All weights must add up to 1
        A = np.identity(self.input_porfolio.n)
        b = np.ones((1,self.input_porfolio.n))
        return A, b
    
    def short_sales_constraint(self):
        #All weights must be greater than zero, this is an inequality constraints
        A = -np.identity(self.input_porfolio.n)
        b = np.zeros((1,self.input_porfolio.n))
        return A, b


    def transaction_cost_equality(self,purchase_weight_vector,sale_weight_vector,original_asset_weights):
        # FIXME: dont we need to define k?
        """
        Defines part of constrainsmatrix A which allows for calculation of the transaction costs. Calculates A, b in Ax=b

        :param purchase_weight_vector: defines the costs of purchases
        :param sale_weight_vector: defines costs of sales
        :param original_asset_weights: original weights of assets. # FIXME: why do we need this? cant we read it from self.input_portfolio?
        #FIXME Answer: I agree
        :return: Tuple (A, b)
        """

        # All purchase weights and sale weights must add up to the original asset weights
        assert len(self.input_porfolio.asset_weights) == len(sale_weight_vector), "sale_vector must be same as number of assets"
        assert len(self.input_porfolio.asset_weights) == len(purchase_weight_vector), "purchase_vector must be same as number of assets"
        assert len(self.input_porfolio.asset_weights) == len(original_asset_weights), "original_asset_weights must be same as number of assets"
        n = self.input_porfolio.n
        A = np.zeros(n, n)
        b = original_asset_weights.copy().T # This implementation is much quicker in python
        np.fill_diagonal(A, self.input_porfolio.asset_weights + sale_weight_vector - purchase_weight_vector)
        return A, b

    def upper_bound_constraint(self, transaction_weights, upper_bounds):
        """
        Calculates part of the Constraint matrix which defines the upper bound constrainst of the transactions

        :param transaction_weights: defines transactions weights
        :param upper_bounds: defines upper bounds per asset
        :return: A, b
        """

        # Upper bound on all transaction weights (purchase or sale)
        n = self.input_porfolio.n
        assert len(transaction_weights) is n, "transaction weights needs to have a length of n"
        assert len(upper_bounds) is n, "the upper bounds vector needs to have a length of n"
        A = np.eye(n)
        np.fill_diagonal(A, transaction_weights) # illy diagonal elements with b. See example in playground
        b = upper_bounds.copy().T
        return A, b
    
    def lower_bound_constraint(self, transaction_weights):
        """
        Defines part of Constraintmatrix Ax = b which sets all lower bound of asset weights to 0

        :param transaction_weights: transaction weights vector
        :return: A, b
        """

        # Upper bound on all transaction weights (purchase or sale)
        n = self.input_porfolio.n
        assert len(transaction_weights) is n, "transaction weights needs to have a length of n"
        A = np.eye(n)
        np.fill_diagonal(A, -transaction_weights)
        b = np.zeros(n).T

        return A, b
