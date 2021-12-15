"""
This file defines the different solvers
"""

# imports
import numpy as np
from cvxopt import matrix, solvers, spmatrix, sparse
from portfolio import Portfolio
from constraints import Constraints
from qpsolvers import solve_qp
import time

# imported optimizers
def solve_qp(C,c,constraints: Constraints):
    n_assets = np.size(c)
    C = matrix(C)
    c = matrix(c.T[0])
    runtime = 0

    if hasattr(constraints,'A') and not hasattr(constraints,'G'):
        A = matrix(constraints.A)
        b = matrix(constraints.b.T[0])
        G = -matrix(np.eye(n_assets))  # negative n x n identity matrix FIXME: I dont think we need them do we?
        h = matrix(np.zeros((n_assets, 1)))
        start = time.time()
        sol = solvers.qp(C,c, A=A, b=b, G=G, h=h)
        runtime = time.time()-start
    if hasattr(constraints,'G') and not hasattr(constraints,'A'):
        G = matrix(constraints.G)
        h = matrix(constraints.h.T[0])
        start = time.time()
        sol = solvers.qp(C,c, G=G, h=h)
        runtime = time.time() - start
    if hasattr(constraints,'A') and hasattr(constraints,'G'):
        A = matrix(constraints.A)
        b = matrix(constraints.b.T[0])
        G = matrix(constraints.G)
        h = matrix(constraints.h.T[0])
        start = time.time()
        sol = solvers.qp(C,c, G, h, A, b)
        runtime = time.time() - start

    # decompose solution
    assert sol['status'] == "optimal", "solver did not find optimal solution"
    x_complete = np.asarray(sol['x'])
    x_buy = []
    x_sell = []
    # x no buys and sells
    if len(x_complete) == constraints.n_assets:
        x = x_complete
    # k = 1
    elif len(x_complete) == 3*constraints.n_assets:
        x = x_complete[0:constraints.n_assets]
        x_buy.append(x_complete[constraints.n_assets:constraints.n_assets*2])
        x_sell.append(x_complete[constraints.n_assets*2])
    # k > 1
    elif len(x_complete) == (2*constraints.k+1)*constraints.n_assets:
        x = x_complete[0:constraints.n_assets]
        for i in range(constraints.k):
            x_buy.append(x_complete[constraints.n_assets*(i+1):constraints.n_assets*(i+2)])
            x_sell.append(x_complete[constraints.n_assets*(i+1+constraints.k):constraints.n_assets*(i+2+constraints.k)])

    return x,x_buy,x_sell,runtime

def solve_sparse_cone_qp(C,c,constraints: Constraints):
    n_assets = np.size(c)
    C = sparse(matrix(C))
    c = matrix(c.T[0])
    runtime = 0

    if hasattr(constraints,'A') and not hasattr(constraints,'G'):
        A = sparse(matrix(constraints.A))
        b = matrix(constraints.b.T[0])
        G = -matrix(np.eye(n_assets))  # negative n x n identity matrix FIXME: I dont think we need them do we?
        h = matrix(np.zeros((n_assets, 1)))
        start = time.time()
        sol = solvers.qp(C,c, A=A, b=b, G=G, h=h)
        runtime = time.time() - start
    if hasattr(constraints,'G') and not hasattr(constraints,'A'):
        G = sparse(matrix(constraints.G))
        h = matrix(constraints.h.T[0])
        start = time.time()
        sol = solvers.qp(C,c, G=G, h=h)
        runtime = time.time() - start
    if hasattr(constraints,'A') and hasattr(constraints,'G'):
        A = sparse(matrix(constraints.A))
        b = matrix(constraints.b.T[0])
        G = sparse(matrix(constraints.G))
        h = matrix(constraints.h.T[0])
        start = time.time()
        sol = solvers.qp(C,c, G, h, A, b)
        runtime = time.time() - start
    # decompose solution
    assert sol['status'] == "optimal", "solver did not find optimal solution"
    x_complete = np.asarray(sol['x'])
    x_buy = []
    x_sell = []
    # x no buys and sells
    if len(x_complete) == constraints.n_assets:
        x = x_complete
    # k = 1
    elif len(x_complete) == 3*constraints.n_assets:
        x = x_complete[0:constraints.n_assets]
        x_buy.append(x_complete[constraints.n_assets:constraints.n_assets*2])
        x_sell.append(x_complete[constraints.n_assets*2])
    # k > 1
    elif len(x_complete) == (2*constraints.k+1)*constraints.n_assets:
        x = x_complete[0:constraints.n_assets]
        for i in range(constraints.k):
            x_buy.append(x_complete[constraints.n_assets*(i+1):constraints.n_assets*(i+2)])
            x_sell.append(x_complete[constraints.n_assets*(i+1+constraints.k):constraints.n_assets*(i+2+constraints.k)])

    return x, x_buy, x_sell, runtime

# own optimizers
def _feasable(x0,A,b,G,h):
    n = len(x0)
    A = A[:,0:n]
    G = G[:,0:n]
    if np.sum(A@x0 != b.T[0]) > 0: return False
    if np.sum(G@x0 > h.T[0]) > 0: return False
    return True

def optimized_OP(C,c, con: Constraints):
    var_costs = (con.k != None)
    n=con.n_assets

    # optimization No 1 -> x0 close to optimal
    x0 = con.input_porfolio.asset_weights.copy()
    if not _feasable(x0,con.A, con.b, con.G, con.h):
        # solve Ax=b subject to d<=x<=e (p.524 in paper)
        A_pre = matrix(con.A)
        b_pre = matrix(con.b.T[0])
        G_pre = matrix(con.G)
        h_pre = matrix(con.h.T[0])
        c_pre = matrix(np.ones(con.n_res_vec))
        sol = solvers.lp(c_pre, G_pre, h_pre, A_pre, b_pre)
        x = np.asarray(sol['x'])
        x = x.T[0,0:len(x0)]
    else:
        x = x0.copy()

    # create set of indeces of basic vars.
    non_basic_idxs = []
    n_curr = n
    for i in range(len(x0)):
        if x0[i]== 0:
            non_basic_idxs.append(i)
    B = con.A[0,0:n][np.newaxis,...]
    b_loc = con.b[0,0][...,np.newaxis, np.newaxis]
    sig_nb = C[:n,:n]
    c_loc = c[:n,0].reshape(n,1)
    u = np.full((len(x0)-len(non_basic_idxs),1),-1)
    while np.min(u) <0:
        while non_basic_idxs:
            idx = non_basic_idxs.pop()
            sig_nb = np.delete(sig_nb,idx,0)
            sig_nb = np.delete(sig_nb, idx, 1)
            B = np.delete(B,idx,1)
            c_loc = np.delete(c_loc,idx,0)
            n_curr -= 1
            b_loc[0,0] -=x[idx]

        zeros = np.zeros((B.shape[0],B.shape[0]))
        mat_upper = np.concatenate((sig_nb,B.T), axis=1)
        mat_lower = np.concatenate((B,zeros),axis=1)
        mat = np.concatenate((mat_upper,mat_lower),axis=0)
        rhs = np.concatenate((c_loc,b_loc),axis=0)
        res = np.linalg.solve(mat, rhs)
        x = res[0:n_curr]
        u_tmp = res[n_curr:0]

        print("test")

    print ("huhu")

    pass