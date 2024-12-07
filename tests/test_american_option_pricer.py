import pytest
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import os

from modules.american_option_pricer import AmericanOptionPricer

@pytest.fixture
def pricer():
    return AmericanOptionPricer(N=100, M=1000, k=5, reg=0.5, rng_seed=123)

@pytest.fixture
def test_data():
    file_path = r'E:\ML in Fin Lab\group_proj\American-Options-Pricing-NN-Tree-models-LSMC-\LSMC American Option Pricing\data\test_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    df = pd.read_csv(file_path)
    required_columns = ['dte', 'r', 'p_iv', 'c_iv', 'strike', 'underlying_last', 'c_bid', 'c_ask', 'p_bid', 'p_ask']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in test_data.csv: {missing_cols}")
    return df

def test_parallel_pricing_with_csv(pricer, test_data):

    num_cores = 2  # Limit to 2 cores for the test environment

    # Price puts in parallel
    put_results = Parallel(n_jobs=num_cores)(
        delayed(pricer.price_put)(row) for _, row in test_data.iterrows()
    )
    put_results_df = pd.DataFrame(put_results)
    test_data = pd.concat([test_data.reset_index(drop=True), put_results_df], axis=1)
    assert not test_data['LSMC_put_price'].isnull().any()
    assert (test_data['LSMC_put_price'] >= 0).all()

    # Price calls in parallel
    call_results = Parallel(n_jobs=num_cores)(
        delayed(pricer.price_call)(row) for _, row in test_data.iterrows()
    )
    call_results_df = pd.DataFrame(call_results)
    test_data = pd.concat([test_data, call_results_df], axis=1)
    assert not test_data['LSMC_call_price'].isnull().any()
    assert (test_data['LSMC_call_price'] >= 0).all()

    output_file = r'E:\ML in Fin Lab\group_proj\American-Options-Pricing-NN-Tree-models-LSMC-\LSMC American Option Pricing\data\test_data_priced.csv'
    test_data.to_csv(output_file, index=False)
    print(f"Priced data saved to {output_file}")

    # Calculate MSE
    put_mse = np.mean(((test_data['p_bid'] + test_data['p_ask']) / 2 - test_data['LSMC_put_price']) ** 2)
    assert put_mse >= 0  # MSE should be non-negative
    print(f"Put MSE: {put_mse}")

    # Calculate MSE
    call_mse = np.mean(((test_data['c_bid'] + test_data['c_ask']) / 2 - test_data['LSMC_call_price']) ** 2)
    assert call_mse >= 0  # MSE should be non-negative
    print(f"Call MSE: {call_mse}")
    return 0
