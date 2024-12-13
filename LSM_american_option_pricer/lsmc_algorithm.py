import numpy as np
from commctrl import NM_FIRST
from numpy.random import default_rng

from .utils import BasisFunctLaguerre, computeBetaReg

class LSMCAlgorithm:
    def __init__(self, N, M, k=3, reg=1.0, seed=42):
        """
        Parameters:
        - N (int): Number of time steps.
        - M (int): Number of simulated paths for Monte Carlo.
        - k (int): Number of basis functions for regression.
        - reg (float): Regularization parameter for Ridge regression.
        - seed (int): Random seed for reproducibility.
        """
        self.N = N
        self.M = M
        self.k = k
        self.reg = reg
        self.rng = default_rng(seed)

    def simulate_paths(self, T, r, sigma, S0):
        """
        Simulate asset price paths using geometric Brownian motion.

        Parameters:
        - T (float): Time to expiration in years.
        - r (float): Risk-free interest rate.
        - sigma (float): Volatility of the underlying asset.
        - S0 (float): Initial price of the underlying asset.

        Returns:
        - S (numpy.ndarray): Simulated asset price paths.
        - t (numpy.ndarray): Time steps.
        """
        dt = T/ self.N
        t = np.linspace(0, T, self.N + 1)

        # Simulate asset paths with antithetic variates
        z = self.rng.standard_normal((int(self.M / 2), self.N))
        z = np.vstack((z, -z))
        W = np.cumsum(z * np.sqrt(dt), axis=1)
        S = S0 * np.exp((r - 0.5 * sigma**2) * t[1:] + sigma * W)
        S = np.column_stack((S0 * np.ones(self.M), S))

        return S, t



    def LSM_put(self, T, r, sigma, K, S0):
        """
        Price an American Put option using LSMC.

        Parameters:
        - T (float): Time to expiration in years.
        - r (float): Risk-free interest rate.
        - sigma (float): Volatility of the underlying asset.
        - K (float): Strike price.
        - S0 (float): Initial price of the underlying asset.

        Returns:
        - discounted_payoff (float): Estimated option price.
        - exe_bound (list): List of exercise boundaries.
        - opt_prices (list): Option price estimates over time.
        """
        S, t = self.simulate_paths(T, r, sigma, S0)
        P = np.maximum(K - S[:, -1], 0)
        exe_bound = []
        opt_prices = []

        # backward induction
        for i in range(self.N - 1, 0, -1):
            ITMput = np.where(K - S[:, i] > 0)[0]  # extract the row index of paths
            if len(ITMput) == 0:
                break
            X = S[ITMput, i]  # ITM prices
            Y = P[ITMput] * np.exp(-r * (t[1] - t[0]))  # discounted payoffs
            A = BasisFunctLaguerre(X, self.k)  # Generate the basis matrix for regression
            beta = computeBetaReg(A, Y, alpha=self.reg)
            continue_value = A @ beta  # the continuation value is the fitted result of regression

            # exercise decision
            immediate_payoff = K - X
            exercise_paths = ITMput[np.where(immediate_payoff > continue_value)[0]]  # paths where exercise is better
            rest = np.setdiff1d(np.arange(self.M), exercise_paths)  # path to continue holding

            # update payoffs
            P[exercise_paths] = immediate_payoff[np.where(immediate_payoff > continue_value)[0]]
            P[rest] *= np.exp(-r * (t[1] - t[0]))

            exe_bound.append((t[i], P[exercise_paths]))
            opt_prices.append((t[i], np.mean(P)))
        discounted_payoff = np.mean(P) * np.exp(-r * (t[1] - t[0]))  # there is actually N+1 steps in total
        return discounted_payoff, exe_bound, opt_prices


    def LSM_call(self, T, r, sigma, K, S0):
        """
        Price an American Call option using LSMC.

        Parameters:
        - T (float): Time to expiration in years.
        - r (float): Risk-free interest rate.
        - sigma (float): Volatility of the underlying asset.
        - K (float): Strike price.
        - S0 (float): Initial price of the underlying asset.

        Returns:
        - discounted_payoff (float): Estimated option price.
        - exe_bound (list): List of exercise boundaries.
        - opt_prices (list): Option price estimates over time.
        """
        S, t = self.simulate_paths(T, r, sigma, S0)
        P = np.maximum(S[:, -1] - K, 0)
        exe_bound = []
        opt_prices = []

        # backward induction
        for i in range(self.N - 1, 0, -1):
            # perform regression on ITM calls to get the conditional expectation prices
            ITMcall = np.where(S[:, i] - K > 0)[0]  # get the indices for ITM calls
            if len(ITMcall) == 0:
                break
            X = S[ITMcall, i]  # ITM prices
            A = BasisFunctLaguerre(X, self.k)
            Y = P[ITMcall] * np.exp(-r * (t[1] - t[0]))
            beta = computeBetaReg(A, Y, alpha=self.reg)
            continue_value = A @ beta

            # exercise decision
            immediate_payoffs = X - K
            exercise_path = ITMcall[
                np.where(immediate_payoffs > continue_value)[0]]  # path where exercise is a better choice
            rest = np.setdiff1d(np.arange(self.M), exercise_path)  # path to continue holding

            # update payoffs
            P[exercise_path] = immediate_payoffs[np.where(immediate_payoffs > continue_value)[0]]
            P[rest] *= np.exp(-r * (t[1] - t[0]))

            exe_bound.append((t[i], P[exercise_path]))
            opt_prices.append((t[i], np.mean(P)))
        discounted_payoff = np.mean(P) * np.exp(-r * (t[1] - t[0]))
        return discounted_payoff, exe_bound, opt_prices