"""
This file defines constrains
"""

# imports
import numpy as np

# Define Constrains

# equality constrains
"""
all weights have to add up to one. One equality constraint
transaction cost 1: the old value is the new value minus buys and plus sells
"""

# inequality constrains
"""
short sales: all weights have to be positive. Results in n inequality constraints
transaction cost 2: buys and sells have to be positive but are bound from above. Results in 2n inequality constraints. 
"""