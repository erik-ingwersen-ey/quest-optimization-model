"""Pipelines used for adjusting input inventory reports to be passed to optimization model.
"""
from datetime import datetime as dt

import pandas as pd

from optimization.constants import Columns
from optimization.datatools.dataprep import bu_qty_on_hand
from optimization.datatools.dataprep import clean_names
from optimization.datatools.dataprep import columns_error_handling
from optimization.datatools.dataprep import columns_needed
from optimization.datatools.dataprep import combine_lot_non_lot
from optimization.datatools.dataprep import create_column
from optimization.datatools.dataprep import cumulative_days_of_inventory
from optimization.datatools.dataprep import cumulative_stock
from optimization.datatools.dataprep import date_column
from optimization.datatools.dataprep import days_of_inventory
from optimization.datatools.dataprep import days_to_expire
from optimization.datatools.dataprep import delta_days_of_inventory
from optimization.datatools.dataprep import how_many_bus_have_item
from optimization.datatools.dataprep import items_to_expire
from optimization.datatools.dataprep import last_transformations
from optimization.datatools.dataprep import remove_expired_items
from optimization.datatools.dataprep import reset_index
from optimization.datatools.dataprep import substitute_flag


##  Module Functions  ######################################################

def load_data_pipeline(x_df: pd.DataFrame, columns_list: list, non_lot: bool = True) -> pd.DataFrame:
    """Pipeline used for loading input files and selection necessary columns.

    This function is called by ``data_pipeline`` and the following steps are performed:

    - **Step 1:** ``clean_names`` Clean column names. This step makes sure no inconsistencies on column \
     names will affect the optimization model.

    - **Step 2:** ``columns_error_handling`` If one of the necessary columns needed by the model is not found, \
    this step tries to find the column based on other possible names for that given column. \
    These lists of possible alternative column names are stored inside the method and were populated \
    with names seen on different inventory reports throughout development phase.

    - **Step 3:** ``columns_needed`` After making sure all columns were found, select all necessary columns. \
    These columns are required either on the **final output report or by the model**.

    - **Step 4 (For Non Lot Only):** Populates column ``lot_id`` with **NONLOT**. Column is passed \
     with null values and this can impact some processes, since later on both lot and non lot reports \
      are joined together.

    - **Step 5 (For Non Lot Only):** Populate column ``lot_qty_on_hand`` with same value as ``bu_qty_on_hand`` for \
    the same reasons as **Step 5**.

    - **Step 6 (For Non Lot Only):** Populate column ``expire_date`` with value that in general terms \
    can be considered infinite for the same reasons as **Step 5** and **Step 6**.

    Parameters
    ----------
    x_df : pd.DataFrame Dataframe with lot or non-lot input input data.
    columns_list : list
        List of columns used throughout the model.
    non_lot : bool, optional Flag if input data being processed is for lot
        or non lot inventory. If **True** then model assumes **non lot** inventory, else **lot**, by default True.

    Returns
    -------
    pd.DataFrame
        Dataframe with lot or non-lot normalized data.

    Note
    ----
    During all the described processes, all changes to input data are either informed for user by raising and error
    or by adding it to the ``log file``.

    """
    load_df = (x_df
               .pipe(clean_names)
               .pipe(columns_error_handling, columns_list)
               .pipe(columns_needed, columns_list)
               .pipe(create_column, Columns.flag_lot, non_lot)
               .pipe(reset_index))

    # Extra non-lot steps
    if non_lot:
        load_df.loc[:, Columns.lot_id] = 'NONLOT'
        load_df.loc[:, Columns.lot_qty_on_hand] = load_df[Columns.bu_qty_on_hand]
        load_df.loc[:, Columns.expire_date] = dt(2100, 1, 1)

    return load_df


def clean_data(x_df: pd.DataFrame) -> pd.DataFrame:
    """Filter input data so we only consider data-points that can be optimized and that we're removing possible data
    with errors

    The filters that are applied in this function take into consideration
    not only the data types that the columns have (or that they should have).
    It also considers aspects related to what that column really means.

    Example
    -------
    The column ``bu_qty_on_hand`` represents the quantity of items a given BU has.
    It can be expected that this column should have only numerical values that are integers
    (we can't have half an item) but also that they should be, at least theoretically bigger
    or equal to zero.

    **Steps**

    - **Step 1:** Replaces columns can_transfer_inventory and can_receive_inventory values from [Y, N] to [1, 0]

    - **Step 2:** Changes columns with rows that have dates from str to real dates dtype

    - **Step 3:** Fill blank values tha can impact in the model

    - **Step 4:** Filter out rows with price value equal to zero.

    - **Step 5:** Make sure BU can at least receive or transfer inventory

    - **Step 6:** Filter out SKUs that have only one BU that uses them

    - **Step 7:** Removes expired items. This is not the same as deleting the entire row. \
    We just consider that given Lot to have 0 items on hand.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe with inventory data.

    Returns
    -------
    pd.DataFrame : Cleaned dataframe

    """
    return (x_df
            .reset_index()
            .pipe(substitute_flag, Columns.can_transfer_inventory)
            .pipe(substitute_flag, Columns.can_receive_inventory)
            .pipe(date_column, [Columns.report_date, Columns.expire_date, Columns.date_lot_added_to_bu_inv])
            .fillna({Columns.lot_id: "NULL_LOT",
                     Columns.bu_qty_on_hand: 0,
                     Columns.lot_qty_on_hand: 0,
                     Columns.average_item_daily_use: 0,
                     Columns.price: 0,
                     Columns.can_transfer_inventory: 0,
                     Columns.can_receive_inventory: 0,
                     Columns.expire_date: dt(2199, 12, 1)})
            .query(Columns.price + '> 0')
            .query(Columns.can_receive_inventory + "+" + Columns.can_receive_inventory + ">=1")
            .pipe(how_many_bus_have_item)
            .query(Columns.how_many_bu + "> 1")
            .pipe(remove_expired_items))


def doi_balance_pipeline(x_df: pd.DataFrame) -> pd.DataFrame:
    """ Pipeline for calculating DOI (days of inventory) balance.

    Parameters
    ----------
    x_df : pd.DataFrame

    """
    return (x_df
            .pipe(days_of_inventory)
            .pipe(cumulative_days_of_inventory)
            .pipe(cumulative_stock)
            .pipe(delta_days_of_inventory))


def calculations_pipeline(x_df: pd.DataFrame) -> pd.DataFrame:
    """Based on our input reports, calculate necessary columns that will we used by the optimization model.

    **Added Calculated Columns:**

    * ``Inventory Balance``: How far from the target number of days of inventory each item at each business unit are.

    * ``Days to Expire``: Total number of days that each Lot item has before expiring.

    * ``Items to Expire``: Based on the business unit consumption rate for that item, and the quantity it has on hand,\
        how many of those items will expire before the business unit has a chance to consume them.

    Parameters
    ----------
    x_df : pd.DataFrame
        Combined Inventory report for lot and non-lot items.

    Returns
    -------
    pd.DataFrame : pd.DataFrame
        Table with additional columns for ``inventory balance`` and ``items to expire``

    Note
    ----
    If you want to add new calculations to the report, we recommend adding the calculations to this pipeline.
    That's because there are other steps that are performed before that clean our input data and performing the calculations
    before might give you bad results.
    """
    return (x_df
            .assign(lot_id=lambda x_df: x_df.lot_id.astype('str'))
            .pipe(days_to_expire)
            .pipe(items_to_expire)
            .pipe(bu_qty_on_hand)
            .pipe(doi_balance_pipeline)
            .pipe(last_transformations))


def data_pipeline(lot_df: pd.DataFrame, nonlot_df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline with data transformations.

    Parameters
    ----------
    lot_df : pd.DataFrame
        Inventory report for lot items.
    nonlot_df : pd.DataFrame
        Inventory report for non-lot items.

    Returns
    -------
    pd.DataFrame: Complete inventory report for current month
    """
    df_non_lot = load_data_pipeline(nonlot_df, columns_list=Columns.cols_needed, non_lot=True)
    df_lot = load_data_pipeline(lot_df, columns_list=Columns.cols_needed, non_lot=False)

    return (df_non_lot
            .pipe(combine_lot_non_lot, df_lot)
            .pipe(reset_index)
            .pipe(clean_data)
            .pipe(bu_qty_on_hand)
            .pipe(calculations_pipeline)
            .pipe(how_many_bus_have_item)
            .query(Columns.how_many_bu + " > 1"))
