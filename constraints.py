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
NO_COST = 1
FIX_COST = 2
VAR_COST = 3

# Define Constraints
class Constraints:
    def __init__(self,
                input_porfolio: Portfolio,
                n_res_vec: int = None,
                k : int = None):
        """
        Initializes a constraint object which contains the equality sonstrains of form Ax=b and inequality constrains
        defind by Gx<=h.

        x is defined as (x1,...,xn,x+11,...,x+n1,x+12,...,x+nk,x-11,...,x-n1,x-12,...,x-nk).T

        :param input_porfolio: portfolio the constraints are defined for
        :param n_x: defines length of the solution vector
        """
        """
        

        :param input_porfolio: portfolio the constraints are defined for
        """

        self.input_porfolio = input_porfolio
        self.n_assets = input_porfolio.n
        if n_res_vec is not None:
            if k is not None:
                assert n_res_vec == (2 *k+1) *self.n_assets, f"n_res_vec must be 2*n or (2*k+1)*n but is {n_res_vec}"
            else:
                assert n_res_vec==3*self.n_assets,f"n_res_vec must be 2*n or (2*k+1)*n but is {n_res_vec}"

        self.n_res_vec = input_porfolio.n if n_res_vec is None else n_res_vec
        self._mode = NO_COST
        self.k = k
        if n_res_vec == 3 * self.n_assets:
            self._mode = FIX_COST
        if k is not None:
            if n_res_vec == (2 * k + 1) * self.n_assets:
                self._mode = VAR_COST

    def __set_equality_constraints(self, A_new, b_new):
        """
        private helper to add Equality constraints
        :param A_new: matrix A to add
        :param b_new: vector b to add
        :return: None
        """
        if hasattr(self,'A'):
            self.A = np.vstack((self.A,A_new))
        else:
            self.A = A_new

        if hasattr(self,'b'):
            self.b = np.vstack((self.b, b_new))
        else:
            self.b = b_new

    def __set_inequality_constraints(self, G_new, h_new):
        """
        private helper function to set inequality constraints

        :param G_new: matrix G to add
        :param h_new: vector h to add
        :return: None
        """
        if hasattr(self,'G'):
            self.G = np.vstack((self.G,G_new))
        else:
            self.G = G_new

        if hasattr(self,'h'):
            self.h = np.vstack((self.h, h_new))
        else:
            self.h = h_new

    def add_weight_summation_constraint(self):
        """
        Defines the part of Constrain matrix Ax = b which ensures that all weights add up to 1

        """

        # All weights must add up to 1
        A_ones = np.ones((1,self.n_assets))#TODO: sorry, for changing it back. As you did it, almost all rows were plain zero rows, which lead to the constraint 0x=0, which is not wrong but a bit unnecessary.
        A_zeros = np.zeros((1,self.n_res_vec-self.n_assets))
        A_new = np.hstack((A_ones,A_zeros))
        b_new = np.ones((1,1))
        self.__set_equality_constraints(A_new=A_new, b_new=b_new)
    
    def add_short_sales_constraint(self):
        """
        adds short sale constrains. This is that each weight of the assets has to be grater than 0

        :return: None
        """

        G_new = np.zeros((self.n_assets,self.n_res_vec)) #TODO: sorry, for changing it back. As you did it, almost all rows were plain zero rows, which lead to the constraint 0x=0, which is not wrong but a bit unnecessary.
        np.fill_diagonal(G_new[:self.n_assets, :self.n_assets], -1)
        h_new = np.zeros((self.n_assets,1))
        self.__set_inequality_constraints(G_new=G_new, h_new=h_new)

    def add_fixed_transaction_cost_equality(self, purchase_weight_vector, sale_weight_vector):

        # All purchase weights and sale weights must add up to the original asset weights
        assert self._mode == FIX_COST, "to add this constraint, the Constraint object has to be configured to support " \
                                      "the fixed cost model"
        assert len(self.input_porfolio.asset_weights) == len(sale_weight_vector), "sale_vector must be same as " \
                                                                                  "number of assets"
        assert len(self.input_porfolio.asset_weights) == len(purchase_weight_vector), "purchase_vector must be same " \
                                                                                      "as number of assets"

        # add equality constraints
        eye = np.eye(self.n_assets)
        neg_eye = eye.copy()
        np.fill_diagonal(neg_eye,-1)
        A_new = np.hstack((eye,neg_eye,eye))
        b_new = self.input_porfolio.asset_weights.copy().reshape((len(self.input_porfolio.asset_weights),1))
        self.__set_equality_constraints(A_new=A_new, b_new=b_new)

        """
        # add inequality constraints
        G_lower = np.zeros((self.n_res_vec,self.n_res_vec))
        np.fill_diagonal(G_lower[self.n_assets:, self.n_assets:], -1)
        h_lower = np.zeros((self.n_res_vec,1))
        h_lower[self.n_assets:] = np.vstack((purchase_weight_vector.reshape(len(purchase_weight_vector),1),sale_weight_vector.reshape(len(sale_weight_vector),1)))
        G_upper = np.zeros((self.n_res_vec,self.n_res_vec))
        np.fill_diagonal(G_lower[self.n_assets:, self.n_assets:], 1)
        h_upper = np.zeros((self.n_res_vec,1))

        G_new= np.vstack((G_upper, G_lower))
        h_new= np.vstack((h_upper, h_lower))
        
        
        self.__set_inequality_constraints(G_new=G_new, h_new=h_new)
        """

        zeros_block = np.zeros((2*self.n_assets,self.n_assets))
        eye_2n = np.eye(2*self.n_assets)
        neg_eye_2n = np.zeros_like(eye_2n)
        np.fill_diagonal(neg_eye_2n,-1)
        G_lower = np.hstack((zeros_block,neg_eye_2n))
        G_upper = np.hstack((zeros_block, eye_2n))
        G_new = np.vstack((G_lower,G_upper))
        h_lower = np.zeros((2*self.n_assets,1))
        h_new = np.vstack((h_lower,purchase_weight_vector.reshape(len(purchase_weight_vector),1),
                           sale_weight_vector.reshape(len(sale_weight_vector),1)))
        self.__set_inequality_constraints(G_new=G_new,h_new=h_new)

    def add_random_fixed_transaction_cost_equality(self):
        assert self._mode == FIX_COST, "to add this constraint, the Constraint object has to be configured to support " \
                                      "the fixed cost model"
        self.create_random_upper_bounds_fixed()
        self.add_fixed_transaction_cost_equality(self.e, self.d)

    def add_var_transaction_cost_equality(self, purchase_weight_vector_list, sale_weight_vector_list):

        # All purchase weights and sale weights must add up to the original asset weights
        assert self._mode == VAR_COST, "to add this constraint, the Constraint object has to be configured to support " \
                                      "the variable cost model"
        assert len(self.input_porfolio.asset_weights) == len(sale_weight_vector_list[0]), "sale_vector must be same as " \
                                                                                  "number of assets"
        assert len(self.input_porfolio.asset_weights) == len(purchase_weight_vector_list[0]), "purchase_vector must be same " \
                                                                                      "as number of assets"

        # add equality constraints
        eye = np.eye(self.n_assets)
        neg_eye = eye.copy()
        np.fill_diagonal(neg_eye,-1)
        A_new = eye
        for i in range(self.k):
            A_new = np.hstack((A_new,neg_eye))
        for i in range(self.k):
            A_new = np.hstack((A_new,eye))
        b_new = self.input_porfolio.asset_weights.copy().reshape((len(self.input_porfolio.asset_weights),1))
        self.__set_equality_constraints(A_new=A_new, b_new=b_new)

        '''
        # add inequality constraints
        G_lower = np.zeros((self.n_res_vec,self.n_res_vec))
        np.fill_diagonal(G_lower[self.n_assets:, self.n_assets:], -1)
        h_lower = np.zeros((self.n_res_vec,1))
        h_lower[self.n_assets:] = np.vstack((purchase_weight_vector_list.reshape(len(purchase_weight_vector_list),1),sale_weight_vector_list.reshape(len(sale_weight_vector_list),1)))
        G_upper = np.zeros((self.n_res_vec,self.n_res_vec))
        np.fill_diagonal(G_lower[self.n_assets:, self.n_assets:], 1)
        h_upper = np.zeros((self.n_res_vec,1))

        G_new= np.vstack((G_upper, G_lower))
        h_new= np.vstack((h_upper, h_lower))
        
        
        self.__set_inequality_constraints(G_new=G_new, h_new=h_new)
        '''

        zeros_block = np.zeros((2*self.n_assets*self.k,self.n_assets))
        eye_kn = np.eye(2*self.k*self.n_assets)
        neg_eye_kn = np.zeros_like(eye_kn)
        np.fill_diagonal(neg_eye_kn,-1)
        G_lower = np.hstack((zeros_block,neg_eye_kn))
        G_upper = np.hstack((zeros_block, eye_kn))
        G_new = np.vstack((G_lower,G_upper))
        h_lower = np.zeros((2*self.k*self.n_assets,1))
        purchase_bound = purchase_weight_vector_list[0].reshape(len(purchase_weight_vector_list[0]),1)
        sales_bound = sale_weight_vector_list[0].reshape(len(sale_weight_vector_list[0]),1)
        for i in range(1,self.k):
            purchase_bound = np.vstack(
                (purchase_bound,purchase_weight_vector_list[i].reshape(len(purchase_weight_vector_list[i]),1)))
            sales_bound = np.vstack(
                (sales_bound, sale_weight_vector_list[i].reshape(len(sale_weight_vector_list[i]), 1)))
        h_new = np.vstack((h_lower,purchase_bound,sales_bound))#FIXME: check order of e and d
        self.__set_inequality_constraints(G_new=G_new,h_new=h_new)

    def add_random_var_transaction_cost_equality(self):
        #FIXME: Can we make this function output a numpy array with differe d and e values for different assets based on
        #which bucket they fall into? We can perhaps keep a track of which asset falls into which bucket as well. 
        #Based on that tracking, we can also decide the different p and q values for the objective function
        #I modified add_var_transaction_cost_equality to treat the out of this function as a numpy array
        self.create_random_upper_bounds_var()
        self.add_var_transaction_cost_equality(self.e,self.d)

    def create_random_upper_bounds_fixed(self):
        assert self._mode == FIX_COST, "to add this constraint, the Constraint object has to be configured to support " \
                                      "the fixed cost model"
        self.d = np.random.uniform(0.01*(1/self.n_assets),0.5*(1/self.n_assets),(self.n_assets))
        self.e = np.random.uniform(0.01*(1/self.n_assets),0.5*(1/self.n_assets),(self.n_assets))

    def create_random_upper_bounds_var(self):
        assert self._mode == VAR_COST, "to add this constraint, the Constraint object has to be configured to support " \
                                      "the variable cost model"
        self.d = []
        self.e = []
        bounds = np.random.uniform(0.01*(1/self.n_assets),0.5*(1/self.n_assets),self.k+1)
        bounds[0]=0.01*(1/self.n_assets)
        bounds[self.k] = 0.5*(1/self.n_assets)
        bounds.sort()
        for k in range(self.k):
            d_tmp = np.random.uniform(bounds[k],bounds[k+1],(self.n_assets))
            e_tmp = np.random.uniform(bounds[k],bounds[k+1],(self.n_assets))
            self.d.append(d_tmp)
            self.e.append(e_tmp)
