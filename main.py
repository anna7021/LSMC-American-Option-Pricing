import os
import pandas as pd
import numpy as np
import logging
from joblib import Parallel, delayed
from modules.data_loader import *
from modules.american_option_pricer import AmericanOptionPricer
import time


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler("pricing_process_log.log", mode="w")  # Log to file
        ]
    )


def log_progress(row_index, total_rows, price_type):
    percentage_complete = (row_index + 1) / total_rows * 100
    if row_index % 1000 == 0 or row_index == total_rows - 1:
        logging.info(f"{price_type} pricing: Processed {row_index + 1}/{total_rows} rows "
                     f"({percentage_complete:.2f}% complete).")


def main():
    setup_logging()
    logging.info("Starting the pricing process...")

    data_loader = DataLoader(r"E:\ML in Fin Lab\group_proj\American-Options-Pricing-NN-Tree-models-LSMC-\LSMC American Option Pricing\data\cleaned_spy_eod_202312.csv")
    data = data_loader.load_data()
    logging.info("Data loaded successfully.")

    required_columns = ['dte', 'r', 'p_iv', 'c_iv', 'strike', 'underlying_last', 'c_bid', 'c_ask', 'p_bid', 'p_ask']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        logging.error(f"Missing required columns in input data: {missing_cols}")
        return

    pricer = AmericanOptionPricer(N=100, M=1000, k=5, reg=0.5, rng_seed=123)
    logging.info("AmericanOptionPricer initialized.")

    start_time = time.time()

    logging.info("Starting put option pricing...")
    num_cores = 10
    total_rows = len(data)
    put_results = Parallel(n_jobs=num_cores)(delayed(pricer.price_put)(row) for _, row in data.iterrows())

    logging.info("Put option pricing completed.")
    put_results_df = pd.DataFrame(put_results)
    data = pd.concat([data.reset_index(drop=True), put_results_df], axis=1)

    logging.info("Starting call option pricing...")
    call_results = Parallel(n_jobs=num_cores)(delayed(pricer.price_call)(row) for _, row in data.iterrows())
    logging.info("Call option pricing completed.")
    call_results_df = pd.DataFrame(call_results)
    data = pd.concat([data, call_results_df], axis=1)

    output_file = r'E:\ML in Fin Lab\group_proj\American-Options-Pricing-NN-Tree-models-LSMC-\LSMC American Option Pricing\data\spy_eod_202312_priced.csv'
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data.to_csv(output_file, index=False)
    logging.info(f"Priced data saved to {output_file}")

    # Calculate MSE
    put_mse = np.mean(((data['p_bid'] + data['p_ask']) / 2 - data['LSMC_put_price']) ** 2)
    call_mse = np.mean(((data['c_bid'] + data['c_ask']) / 2 - data['LSMC_call_price']) ** 2)

    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(data)

    logging.info("\nSummary:")
    logging.info(f"Priced {len(data)} options using LSMC method")
    logging.info(f"Put MSE: {put_mse}")
    logging.info(f"Call MSE: {call_mse}")
    logging.info(f"Total execution time: {total_time:.2f} seconds")
    logging.info(f"Average execution time per option: {avg_time:.2f} seconds")


if __name__ == "__main__":
    main()
