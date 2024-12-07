import pytest
from modules.lsmc_algorithm import LSMCAlgorithm

@pytest.fixture # define reusable setups for tests, include the fixture as a parameter
def lsmc():
    """Fixture to initialize the LsmcAlgorithm class for testing."""
    return LSMCAlgorithm(N=100, M=10, k=3, reg=1.0, seed=42)

test_params = [
    (30 / 365.0, 0.01, 0.2, 100, 100),
    (60 / 365.0, 0.01, 0.2, 100, 100)
]
@pytest.mark.parametrize("T, r, sigma, K, S0", test_params)
def test_lsmc_call(lsmc, T, r, sigma, K, S0):
    call_price, call_exe_bound, call_opt_prices = lsmc.LSM_call(T, r, sigma, K, S0)
    print(f"Call Option Results: T={T}, r={r}, sigma={sigma}, K={K}, S0={S0}")
    print(f"Call Price: {call_price}")
    assert call_price > 0, "Call price should be positive"
    assert isinstance(call_exe_bound, list), "Execution boundaries should be a list"
    assert isinstance(call_opt_prices, list), "Option prices should be a list"
    assert len(call_opt_prices) > 0, "Option prices list should not be empty"

@pytest.mark.parametrize("T, r, sigma, K, S0", test_params)
def test_lsmc_put(lsmc, T, r, sigma, K, S0):
    put_price, put_exe_bound, put_opt_prices = lsmc.LSM_put(T, r, sigma, K, S0)
    print(f"Put Option Results: T={T}, r={r}, sigma={sigma}, K={K}, S0={S0}")
    print(f"Put Price: {put_price}")
    assert put_price > 0, "Put price should be positive"
    assert isinstance(put_exe_bound, list), "Execution boundaries should be a list"
    assert isinstance(put_opt_prices, list), "Option prices should be a list"
    assert len(put_opt_prices) > 0, "Option prices list should not be empty"
