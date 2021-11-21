"""
This file defines an portfolio Class
"""

# imports
import numpy as np

# Define Assets
class Portfolio:
    def __init__(self,
                 n: int,
                 portfolio_expected_return: float,
                 std_def_means: float = 1,
                 std_def_var : float = 1,
                 mean_of_vars: float = 5,
                 T: int = 100):
        """
        Initializes a random portfolio
        :param n: number of assets
        :param portfolio_expected_return: expected value over all assets
        :param std_def_means: how large do the assets retun value differ
        :param std_def_var: how large does one asset differ over time
        :param mean_of_vars: how large is the difference of the expected values of the assets
        :param T: lengt of time series
        """
        assert n > 1, f"Portfolio needs to contain at leas one asset. n needs to be larger than 1 but is {n}"
        assert T > 1, f"Negative time series not possible T needs to be larger than 1 but is {T}"
        self.n = n
        self._exp = portfolio_expected_return

        # generate sequence of means and vars and
        means = np.random.normal(self._exp, std_def_means, self.n)
        vars = np.random.normal(mean_of_vars, std_def_var, self.n)
        self.asset_returns = np.zeros_like(means)

        # init nxT matrix containing mean free retruns over time.
        S = np.zeros((T, self.n))

        # generate matrix containing returns of n assets over T timesteps
        for i in range(self.n):

            # generate random time sequence
            P = np.random.normal(means[i], vars[i], T)

            # calculate temporal mean
            P_mean = np.mean(P)

            # subtract temporal mean
            P = P-P_mean

            # adjust self.asset_Retunrs
            self.asset_returns[i] = P_mean

            # fill S
            S[:,i]=P

        #start calculating covariance matrix
        self.covar_matrix = S.T @ S
        self.covar_matrix = self.covar_matrix / T

        # generate random weights
        self.asset_weights = np.asarray(np.random.randint(0, 100, self.n), dtype=np.float)
        self.asset_weights = self.asset_weights / np.sum(self.asset_weights)
