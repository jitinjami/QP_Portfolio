import unittest
from portfolio import Portfolio
from constraints import Constraints
import numpy as np
from cvxopt import matrix, solvers
from problem_formulations import *
from optimizers import *

class testPorfolio(unittest.TestCase):
    def test_PortfolioGeneration(self):
        port = Portfolio(2, 2)
        #To test that the correct number of weights are generated
        self.assertEqual(len(port.asset_weights),2)

class testProblemFormulation(unittest.TestCase):
    def test_basic_markowitz_set_up(self):

        port = Portfolio(2, 2)
        t = 0.1
        C,c, con = basic_markowitz_set_up(port,t)
        self.assertEqual(np.shape(C), (2,2))
        self.assertEqual(np.shape(c), (2,1))
        np.testing.assert_array_equal(con.A, np.array([[1,1]]))
        np.testing.assert_array_equal(con.b, np.array([[1]]))

    def test_short_sales_setup(self):

        port = Portfolio(2, 2)
        t = 0.1
        C,c, con = short_sales_setup(port, t)
        self.assertEqual(np.shape(C), (2,2))
        self.assertEqual(np.shape(c), (2,1))
        np.testing.assert_array_equal(con.A, np.array([[1,1]]))
        np.testing.assert_array_equal(con.b, np.array([[1]]))
        np.testing.assert_array_equal(con.G, -np.identity(2))
        np.testing.assert_array_equal(con.h, np.zeros((2,1)))

    def test_fixed_transaction_cost_setup(self):

        port = Portfolio(2, 2)
        t = 0.1
        C,c, con = fixed_transaction_cost_setup(port, t)
        self.assertEqual(np.shape(C), (6,6))
        self.assertEqual(np.shape(c), (6,1))
        self.assertEqual(np.shape(con.A), (3,6))
        self.assertEqual(np.shape(con.b), (3,1))
        self.assertEqual(np.shape(con.G), (18,6))
        self.assertEqual(np.shape(con.h), (18,1))