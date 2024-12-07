import pandas as pd
import numpy as np
import time
from numpy.random import default_rng

from .lsmc_algorithm import LSMCAlgorithm

class AmericanOptionPricer:
    def __init__(self, N=10, M=100, k=3, reg=1, rng_seed=42):
        """
        Initialize the AmericanOptionPricer with default parameters.

        Parameters:
        - N (int): Number of simulations.
        - M (int): Number of time steps.
        - k (int): Number of Laguerre polynomial basis functions.
        - reg (float): Regularization strength for Ridge regression.
        - rng_seed (int): Seed for the random number generator.
        """
        self.N = N
        self.M = M
        self.k = k
        self.reg = reg
        self.rng_seed = rng_seed
        self.lsmc = LSMCAlgorithm(N=self.N, M=self.M, k=self.k, reg=self.reg, seed=self.rng_seed)

    def price_put(self, row):
        """
        Prices an American Put option using LSM_put.

        Parameters:
        - row (pd.Series): A row from the DataFrame containing option data.

        Returns:
        - pd.Series: Estimated option price and computation time.
        """
        try:
            start_time = time.time()
            T = row['dte'] / 365.0  # Convert days to years
            r = row['r']
            sigma = row['p_iv']  # Implied volatility for puts
            K = row['strike']
            S0 = row['underlying_last']

            price, _, _ = self.lsmc.LSM_put(T, r, sigma, K, S0)
            end_time = time.time()
            elapsed_time = end_time - start_time
            return pd.Series({'LSMC_put_price': price, 'Put_time_sec': elapsed_time})
        except Exception as e:
            print(f"Error pricing put option for row {row.name}: {e}")
            return pd.Series({'LSMC_put_price': np.nan, 'Put_time_sec': np.nan})

    def price_call(self, row):
        """
        Prices an American Call option using LSM_call.

        Parameters:
        - row (pd.Series): A row from the DataFrame containing option data.

        Returns:
        - pd.Series: Estimated option price and computation time.
        """
        try:
            start_time = time.time()
            T = row['dte'] / 365.0  # Convert days to years
            r = row['r']
            sigma = row['c_iv']  # Implied volatility for calls
            K = row['strike']
            S0 = row['underlying_last']

            price, _, _ = self.lsmc.LSM_call(T, r, sigma, K, S0)
            end_time = time.time()
            elapsed_time = end_time - start_time
            return pd.Series({'LSMC_call_price': price, 'Call_time_sec': elapsed_time})
        except Exception as e:
            print(f"Error pricing call option for row {row.name}: {e}")
            return pd.Series({'LSMC_call_price': np.nan, 'Call_time_sec': np.nan})
