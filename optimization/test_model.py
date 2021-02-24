"""Python script used for testing optimization model.

This python script performs only three "simple" tasks:

    * Loads input files for lot and non lot reports (stored at input_data folder inside this package);

    * Calls main function that performs the optimization process;

    * Saves results to our results folder.

If necessary, deleting this script from codebase will not result in errors.

"""
import os

import pandas as pd

from optimization import input_data
from optimization.model.main import main
from optimization.opt_tools.load_data import load_data

if __name__ == '__main__':

    lot_fname = 'Input_Lot_Inventory_Report.xlsx'
    nonlot_fname = 'Input_Non_Lot_Inventory_Report.xlsx'

    input_dir = os.path.dirname(os.path.abspath(input_data.__file__))

    try:
        # Loading lot and nonlot inventory. We're using input_data folder as base fpath.
        lot_df = load_data(os.path.join(input_dir, lot_fname))
        nonlot_df = load_data(os.path.join(input_dir, nonlot_fname))
        # Making sure process returned with 2 pandas dataframes as it should
        assert type(lot_df) == pd.DataFrame
        assert type(nonlot_df) == pd.DataFrame

    except AssertionError:
        raise AssertionError(
            "Something went wrong while loading lot and nonlot reports.")

    # sku_qty = 0 --> run optimization for all sku. Otherwise run for some specific qty of SKU's
    main(lot_df, nonlot_df, optimize_what='both', save_res=True, sku_qty=0)
