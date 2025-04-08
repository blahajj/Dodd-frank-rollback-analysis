#!/usr/bin/env python3

"""
ubpr_utils.py

Utility function to retrieve a time series for a specific bank and UBPR metric
from a pre-combined UBPR CSV file.
"""

import pandas as pd

def get_ubpr_timeseries(
    csv_path: str,
    bank_id: int,
    ubpr_code: str,
    date_format: str = "%m/%d/%Y %I:%M:%S %p"
) -> pd.Series:
    """
    Reads a combined UBPR CSV file, filters by the given bank ID and UBPR code,
    and returns a time series (index is date).

    :param csv_path: Path to the combined UBPR CSV file
    :param bank_id: The bank's RSSD ID to filter on
    :param ubpr_code: The UBPR metric code (e.g., "UBPR7204") to filter on
    :param date_format: Format for parsing the date index (optional override)
    :return: A Pandas Series containing the requested UBPR metric over time
    """

    # 1) Read the CSV with a multi-level header
    #    index_col=0 means use the first column as the index
    #    header=[0,1] means the first two rows in CSV contain column headers
    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])

    # 2) Convert the index to datetime, if it matches the given date_format
    #    If it fails, we just leave it as-is
    try:
        df.index = pd.to_datetime(df.index, format=date_format).date
    except ValueError:
        # If there's a mismatch in date format or the index isn't a date, ignore
        pass

    # 3) Drop the second level of the header, which typically contains descriptions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # 4) Filter for rows matching the bank’s RSSD ID
    #    "ID RSSD" is one of the columns
    if "ID RSSD" not in df.columns:
        raise KeyError("Could not find 'ID RSSD' column in the DataFrame.")

    mask = (df["ID RSSD"] == bank_id)
    filtered_df = df.loc[mask]

    # 5) Select the UBPR code column (e.g. "UBPR7204")
    if ubpr_code not in filtered_df.columns:
        raise KeyError(f"Could not find UBPR code '{ubpr_code}' in the DataFrame.")

    series = filtered_df[ubpr_code].copy()

    # 6) Sort by the index (date)
    series.sort_index(inplace=True)

    return series
