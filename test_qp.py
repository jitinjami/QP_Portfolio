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
        n = 20
        port = Portfolio(n, 2)
        t = 0.1
        C,c, con = basic_markowitz_set_up(port,t)
        self.assertEqual(np.shape(C), (n,n))
        self.assertEqual(np.shape(c), (n,1))
        np.testing.assert_array_equal(con.A, np.ones((1,n)))
        np.testing.assert_array_equal(con.b, np.ones((1,1)))

    def test_short_sales_setup(self):
        n=20
        port = Portfolio(n, 2)
        t = 0.1
        C,c, con = short_sales_setup(port, t)
        self.assertEqual(np.shape(C), (n,n))
        self.assertEqual(np.shape(c), (n,1))
        np.testing.assert_array_equal(con.A, np.ones((1,n)))
        np.testing.assert_array_equal(con.b, np.ones((1,1)))
        np.testing.assert_array_equal(con.G, -np.identity(n))
        np.testing.assert_array_equal(con.h, np.zeros((n,1)))

    def test_fixed_transaction_cost_setup(self):
        n = 5
        port = Portfolio(n, 2)
        t = 0.1
        C,c, con = fixed_transaction_cost_setup(port, t)
        len_result_vec = 3*n
        self.assertEqual(np.shape(C), (len_result_vec,len_result_vec))
        self.assertEqual(np.shape(c), (len_result_vec,1))
        self.assertLessEqual(np.sum(c[0:n]),0.0)
        self.assertGreaterEqual(np.sum(c[n:]),0.0)
        self.assertEqual(np.shape(con.A), (n+1,len_result_vec))
        self.assertEqual(np.shape(con.b), (n+1,1))
        self.assertEqual(np.shape(con.G), (5*n,len_result_vec)) #2*n for 0<=x+/-, 2*n for x<=e/d and one for x>=0
        self.assertEqual(np.shape(con.h), (5*n,1))

    def test_var_transaction_cost_setup(self):
        n = 2
        k = 3
        port = Portfolio(n, 2)
        t = 0.1
        C,c, con = variable_transaction_cost_setup(port, t, k)
        len_result_vec = (2*k+1)*n # from paper
        self.assertEqual(np.shape(C), (len_result_vec,len_result_vec))
        self.assertEqual(np.shape(c), (len_result_vec,1))
        self.assertLessEqual(np.sum(c[0:n]),0.0)
        self.assertGreaterEqual(np.sum(c[n:]),0.0)
        self.assertEqual(np.shape(con.A), (n+1,len_result_vec))
        self.assertEqual(np.shape(con.b), (n+1,1))
        self.assertEqual(np.shape(con.G), ((4*k+1)*n,len_result_vec))
        self.assertEqual(np.shape(con.h), ((4*k+1)*n,1))

class test_Solutions(unittest.TestCase):
    def test_solution_cvxopt(self):
        eps = 10e-3
        n = 5
        k = 3
        t = 0.1
        port = Portfolio(n, 5)
        C, c, con = variable_transaction_cost_setup(port, t, k)
        x, x_buy, x_sell = solve_qp(C, c, con)
        sum = np.sum(x)
        self.assertAlmostEqual(sum, 1.0, 8, "The sum of the new weights does not sum up to 1")

        #test if x-x_buy+x_sell = x0
        test_x = x.copy()
        for i in range(k):
            test_x = test_x -x_buy[i]+x_sell[i]
        test_x = test_x.T[0]
        diff = np.sum(test_x - port.asset_weights)
        self.assertAlmostEqual(diff, 0.0, 8, "your buy and sell assets are not equal")

        # test if you dont buy and sell the same stock
        for i in range (k):
            for j in range(n):
                if not (x_buy[i][j,0]<eps):
                    self.assertAlmostEqual(x_sell[i][j,0],0.0,3,f"you are selling and buying the asset {j} of block {i}")

    def test_solution_optimized_qp(self):
        eps = 10e-4
        n = 5
        k = 2
        t = 0.1
        port = Portfolio(n, 5)
        C, c, con = variable_transaction_cost_setup(port, t,k)
        x, x_buy, x_sell = optimized_OP(C, c, con)
        sum = np.sum(x)
        self.assertAlmostEqual(sum, 1.0, 8, "The sum of the new weights does not sum up to 1")

        #test if x-x_buy+x_sell = x0
        test_x = x.copy()
        for i in range(k):
            test_x = test_x -x_buy[i]+x_sell[i]
        test_x = test_x.T[0]
        diff = np.sum(test_x - port.asset_weights)
        self.assertAlmostEqual(diff, 0.0, 8, "your buy and sell assets are not equal")

        # test if you dont buy and sell the same stock
        for i in range (k):
            for j in range(n):
                if(x_buy[i][j,0]<eps):
                    self.assertNotAlmostEqual(x_sell[i][j,0],0.0,4,f"you are selling and buying the asset {j} of block {i}")

