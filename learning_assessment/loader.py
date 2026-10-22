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

ID_COLUMNS = ["id", "name"]
LONG_COLUMNS = ID_COLUMNS + ["Date", "Assignment", "Objective", "Score"]


def load(datapath):
    """
    Load assessment CSV files into an observation table.

    Each output row represents one student, objective, and assessment. An
    objective that is not part of an assessment does not generate any rows for
    that assessment. A missing student submission remains an explicit missing
    value in ``Score`` so that the model can decide how to handle it.

    Parameters:
    datapath (str): The path to the directory containing the CSV files.

    Returns:
    pd.DataFrame: A ataFrame with columns ``id``, ``name``,
        ``Date``, ``Assignment``, ``Objective``, and ``Score``.
    """
    all_files = sorted(glob.glob(os.path.join(datapath, "*.csv")))
    df_list = []

    for file in all_files:
        df = pd.read_csv(file)
        missing_columns = set(ID_COLUMNS) - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Assessment file {file} is missing columns: {missing}")

        date = os.path.basename(file).replace(".csv", "").split(" ")[-1]
        assignment = os.path.basename(file).replace(".csv", "").strip()

        objective_columns = [
            column for column in df.columns if column not in ID_COLUMNS
        ]
        if not objective_columns:
            raise ValueError(f"Assessment file {file} has no objective columns")

        long_df = df.melt(
            id_vars=ID_COLUMNS,
            value_vars=objective_columns,
            var_name="Objective",
            value_name="Score",
        )
        long_df["Date"] = pd.to_datetime(date)
        long_df["Assignment"] = assignment
        df_list.append(long_df[LONG_COLUMNS])

    if not df_list:
        raise ValueError(f"No assessment CSV files found in {datapath}")

    compiled_df = pd.concat(df_list, ignore_index=True)
    return compiled_df.sort_values(
        ["Date", "Assignment", "Objective", "id"], ignore_index=True
    )
