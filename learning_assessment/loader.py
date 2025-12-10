"""
loader.py
Compile assessment data.

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import os
import pandas as pd
import glob


def load(datapath):
    """
    Load and compile assessment data from CSV files in the specified directory.

    Parameters:
    datapath (str): The path to the directory containing the CSV files.

    Returns:
    pd.DataFrame: A DataFrame containing the compiled assessment data.
    """
    all_files = glob.glob(os.path.join(datapath, "*.csv"))
    df_list = []

    for file in all_files:
        df = pd.read_csv(file)
        date = os.path.basename(file).replace(".csv", "").split(" ")[-1]
        assignment = os.path.basename(file).replace(".csv", "").strip()
        # Catch NaNs (missing assignment)
        df = df.fillna(0.0)
        df["Date"] = pd.to_datetime(date)
        df["Assignment"] = assignment
        df_list.append(df)

    compiled_df = pd.concat(df_list, ignore_index=True)
    return compiled_df
