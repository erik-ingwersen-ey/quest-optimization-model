"""Functions created for data transformation or validation processes

Before optimizing inventory, we use the functions defined at this module for validating and manipulating
input data. Here we defined all necessary building blocks for calculating our necessary columns and also to
make sure that there are no potential inconsistencies in the input data that could impact model recommendation constants.
These methods are used in the form of pipelines.

"""
import logging
from datetime import datetime as dt
from typing import Union

import numpy as np
import pandas as pd

from optimization import constants
from optimization.constants import Columns
from optimization.constants import MIN_SHIPMENT_VALUE

pd.options.mode.use_inf_as_na = True

##  Data_prep Module Variables  ###################################

negative_non_lot_value = 0
negative_lot_value = 0

##  Data_prep Module Functions  #####################################

def column_exists(x_df: pd.DataFrame, column_name: str, condition: int = 0):
    """Assert column exists in dataframe.

    Parameters
    ----------
    x_df : pd.DataFrame
    column_name : str
        Name of the column we want to know if already exists.
    condition : int
        If we want to assert that column exists or not. Defaults to 0.

        - **if condition = 0:** assert that column exist.

        - **if condition = 1:** assert column doesn't exist.

    Raises
    ------
    ValueError: Error message to be displayed if assertion fails.
        **If test fail, this function will stop the model from running.**
    """
    if condition == 0:
        try:
            assert column_name in x_df.columns
        except AssertionError:
            raise ValueError(
                f"Column with name {column_name} not found in dataframe. Some method asked to assert that column " +
                f"with name {column_name} existed.")
    else:
        try:
            assert column_name not in x_df.columns

        except AssertionError:
            raise ValueError(
                "Found column in dataframe. Some method asked to assert that column " +
                f"with name {column_name} didn't exist.")


def columns_needed(x_df: pd.DataFrame, column_list: list):
    """
    Select need columns.

    The method selects only the columns needed based on list of columns needed
    that is passed as an argument.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe with many unused columns.
    column_list : list
        List of columns to be mantained.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with only the columns specified by the column_list.

    """
    try:
        return x_df[column_list]

    except KeyError:
        cols_not_found = np.setdiff1d(column_list, x_df.columns)
        raise KeyError(
            "The following columns are not in the dataframe: {}".format(cols_not_found))


def clean_names(x_df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe column names.

    The method cleans column names by:

        1. Making everything lower case
        2. Adding '_' instead of ' '
        3. Removing special character '$'
        4. removing parenthesis '(' or ')'

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe with columns to be cleaned.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with cleaned columns.

    """
    x_df.columns = (
        x_df.columns.str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('(', '')
            .str.replace(')', '')
            .str.replace('$', '')
            .str.replace('/', '')
            .str.replace('?', '')
            .str.replace('+', '_')
            .str.replace('__', '_'))
    return x_df


def date_column(x_df: pd.DataFrame, column_list: list):
    """
    Convert columns from string to datetime.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe with column dates to be converted.
    column_list : list
        List of column names that need to be converted from string to datetime.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe converted columns.

    """
    for column_name in column_list:
        x_df[column_name] = pd.to_datetime(x_df[column_name], errors='coerce', dayfirst=False)
    return x_df


def fill_na(x_df: pd.DataFrame, column_name: str, value):
    """Fill null rows from column based on specified value.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.
    column_name : str
        Name of the column with null values.
    value : object
        Value to be used to fill null values.

        Can be:

            * A float or int.

            * 'mean': will fill with the mean value of the column.

            * str: will search for a column with this string name at x_df columns.

    Returns
    -------
    x_df : pd.DataFrame
        Result dataframe.

    """
    count = x_df[x_df[column_name].isna()][column_name].count()
    if constants.LOG_MODE:
        logging.warning(
            "Found {} NUll Values. For Column {}. Replacing with: {}".format(count, column_name, value))

    if isinstance(value, str):
        value = get_mean(x_df, column_name) if value == 'mean' else x_df[value]

    x_df[column_name].fillna(value, inplace=True)

    return x_df


def get_mean(x_df: pd.DataFrame, column_name: str):
    """Mean value from column.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.
    column_name : str
        Name of the column to calculate mean value.

    Returns
    -------
    mean_value: object
        Mean of column analyzed.

    """
    return x_df[column_name].mean()


def create_column(x_df: pd.DataFrame, column_name: str, default_value: None):
    """
    Create new column based on passed name and fill with default_value.

    The new column will be filled with values specified by the ``default_value``
    variable. These values can range from **numbers** to **strings** and **even other column
    values** from ``x_df``.

    If ``default_value`` is equal to a column name from x_df, the new column
    gets filled with values from that other column.

    If ``default_value`` is not equal to a column name from ``x_df``, and is of type ``string``,
    ``int`` or ``float``, the new column gets filled with that value.

    Attention
    ---------
    If a column with the same name as ``column_name`` already exists at ``x_df`` this
    function will return an error and inform at ``log file`` about its occurrence.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to place new column
    column_name : str
        Name of new column to be created.
    default_value : object
        Value to be used as default value.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with newly created column.

    """
    x_df = reset_index(x_df)
    column_exists(x_df, column_name=column_name, condition=1)
    if (isinstance(default_value, str)) and (default_value in x_df.columns):
        x_df.loc[:, column_name] = x_df[default_value]
    else:
        x_df.loc[:, column_name] = default_value

    return x_df


def replace_values(x_df: pd.DataFrame, column_name: str, value: None, column_transformed: str, new_value: None):
    """
    Replace Values on column based on condition.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.
    column_name : str
        Name of the column where condition applies.
    value : float
        Value to be used in condition.
    column_transformed : str
        Column with values to modify.
    new_value : float
        New value to be added based on condition.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with new column values.

    """
    column_exists(x_df, column_name=column_name, condition=0)
    x_df.loc[(x_df[column_name] == value), column_transformed] = new_value

    return x_df


def reset_index(x_df: pd.DataFrame):
    """Reset the index to a new dataframe.

    Parameters
    ----------
    x_df : pd.DataFrame

    """
    return x_df.reset_index(drop=True)


def concat(results_list):
    """Concatenates list of results.

    Parameters
    ----------
    results_list : list
        List with optimization results.

    Returns
    -------
        Optimization results : pd.DataFrame:

    """
    assert isinstance(results_list, list), "results_list is not an list with optimization results: %r" % results_list
    return pd.concat(results_list)


def combine_lot_non_lot(*args):
    """Combine lot and non-lot inventories.

    Returns
    -------
        object: Combined inventory.

    """
    inventory_list = [arg for arg in args]
    return concat(inventory_list)


def days_to_expire(x_df: pd.DataFrame):
    """Create new column with days to expire.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with the new column added.

    """
    x_df[Columns.days_to_expire] = x_df[Columns.expire_date] - x_df[Columns.report_date]
    x_df[Columns.days_to_expire] = x_df[Columns.days_to_expire].map(lambda x: np.nan if pd.isnull(x) else x.days)
    x_df[Columns.days_to_expire] = x_df[Columns.days_to_expire].fillna(0)

    # For logging reasons
    error_values = x_df[x_df[Columns.days_to_expire] < 0]
    error_qty = error_values[Columns.days_to_expire].count()
    if error_qty > 0:
        if constants.LOG_MODE:
            logging.warning(
                "Found {} rows with negative quantities of days to expire.."
                " In these situations, right now we're considering them as zero.".format(error_qty),
                extra=error_values)

        x_df.loc[x_df[Columns.days_to_expire] < 0, Columns.days_to_expire] = 0

    return x_df


def remove_expired_items(x_df: pd.DataFrame):
    """
    Removes items that have expired.

    Function verifies if ``expire_date`` is older than ``report_date``.
    If so, consider that **Lot as having zero items**.

    Attention
    ---------
    We don't remove that record from our table, since the business unit
    might have an ``average_daily_consumption_rate`` bigger than **zero**.
    If so, we might still want to transfer her items.

    Parameters
    ----------
    x_df : pd.DataFrame
        Input table to be filtered.

    Returns
    -------
    pd.DataFrame
        Table without expired items.
    """

    x_df.loc[x_df[Columns.expire_date] < x_df[Columns.report_date], Columns.lot_qty_on_hand] = 0
    return x_df


def how_many_bus_have_item(x_df: pd.DataFrame):
    """Calculate number of unique BU's that have a given item.

    This function returns new column that gives us the number of
    unique BU's that have a given item. We use this information to
    filter out Items that have only one BU.

    Parameters
    ----------
    x_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame: Dataframe with new column that has number of BU's that have that a given Item.

    """
    x_df[Columns.how_many_bu] = x_df.groupby([Columns.item_id])[Columns.inv_bu].transform(pd.Series.nunique)
    return x_df


def bu_qty_on_hand(x_df: pd.DataFrame):
    """Recalculates the quantity each BU has on hand.

    This calculation is necessary, since many BU qty on hand are actually wrong
    and in some cases, considering items that have already expired.

    Parameters
    ----------
    x_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame: Dataframe with updated value for BU Qty on Hand.

    """
    x_df[Columns.bu_qty_on_hand] = x_df.groupby([Columns.item_id, Columns.inv_bu])[Columns.lot_qty_on_hand].transform(
        pd.Series.sum)
    return x_df


def cumulative_days_of_inventory(x_df: pd.DataFrame):
    """Calculate the cumulative days to expire for every SKU at every BU.

    This method works like a window SQL function. It does some calculation
    (in this case, the cumulative sum) based on a subgroup from our dataset.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with the new column added.

    See Also
    --------
    `towardsdatascience.com/sql-window-functions-in-python <https://towardsdatascience.com/sql-window-functions-in-python-pandas-data-science-dc7c7a63cbb4>`_

    """
    x_df[Columns.cum_doi] = x_df.sort_values(by=[Columns.days_to_expire], ascending=True) \
        .groupby([Columns.inv_bu, Columns.item_id])[Columns.days_of_inventory].cumsum()

    return x_df


def cumulative_stock(x_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the cumulative days to expire for every SKU at every BU.

    This method works like a window SQL function. It does some calculation
    (in this case, the cumulative sum) based on a subgroup from our dataset.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with added column added.

    """
    x_df[Columns.cum_stock] = x_df.sort_values(by=[Columns.days_to_expire], ascending=True) \
        .groupby([Columns.inv_bu, Columns.item_id])[Columns.lot_qty_on_hand].cumsum()

    return x_df


def items_to_expire(x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new column with items to expire.

    Items to expire calculation should be calculated
    at SKU/Business Unit granularity. This means that for
    Business Units with many different Lots, we need to take
    into consideration that those items from those Lots are going
    to be consumed in a chain-wise manner.

    Example
    -------
    The example bellow demonstrates how we need to consider the relationship between
    different Lots at the same BU, with different expiration dates.

    +--------+----------+---------+---------------------+----------------------------------+
    | Inv BU |  Item ID |  Lot ID |  Lot Days to Expire |   BU Cumulative Days to Expire   |
    +========+==========+=========+=====================+==================================+
    |  BU A  | SKU 1000 | Lot H10 | Days to Expire (10) |  cumulative Days to Expire (10)  |
    +--------+----------+---------+---------------------+----------------------------------+
    |  BU A  | SKU 1000 | Lot H15 | Days to Expire (18) |  cumulative Days to Expire (28)  |
    +--------+----------+---------+---------------------+----------------------------------+
    |  BU A  | SKU 1000 | Lot X05 | Days to Expire (25) |  cumulative Days to Expire (53)  |
    +--------+----------+---------+---------------------+----------------------------------+
    |  BU A  | SKU 1000 | Lot O90 | Days to Expire (30) |  cumulative Days to Expire (83)  |
    +--------+----------+---------+---------------------+----------------------------------+
    |  BU A  | SKU 1000 | Lot L07 | Days to Expire (50) | cumulative Days to Expire (133)  |
    +--------+----------+---------+---------------------+----------------------------------+

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with days to expire calculated.

    Note
    ----
    Days to expire should be in ascending order.

    """
    # cumulative items to expire is equal to the number of items in stock,
    # subtracted the amount of items that the target DOI accounts for.
    # Since the average daily consumption rate vary from 0 to large positive numbers
    # we have that the average consumption rate times DOI target ranges from 0 to large positive.
    x_df = x_df.pipe(cumulative_stock)
    x_df = x_df.sort_values(by=[Columns.inv_bu, Columns.item_id, Columns.expire_date, Columns.cum_stock],
                            ascending=True)

    x_df[Columns.cum_items_to_expire] = x_df[Columns.cum_stock] - (
            x_df[Columns.average_item_daily_use] * x_df[Columns.days_to_expire])

    # If value is negative, it means that there are no items to expire.
    x_df.loc[x_df[Columns.cum_items_to_expire] < 0, Columns.cum_items_to_expire] = 0
    # Since non-lot items have no expire date, even if their average daily use is zero,
    # that doesn't mean that they are going to expire.
    x_df.loc[x_df[Columns.flag_lot] == True, Columns.cum_items_to_expire] = 0

    # Grouping values by BU ID and Item ID, and sorting values by days to expire in ascending order.
    y_df = x_df.sort_values(by=[Columns.inv_bu, Columns.item_id, Columns.expire_date], ascending=True)

    # last_bu and last_item are going to be used as control variables.
    # When looping through our inventory, we want to now when we arrive at
    # new combinations of Item ID and BU IDs.
    last_bu = y_df[Columns.inv_bu].iloc[0]
    last_item = y_df[Columns.item_id].iloc[0]

    expiring_items = [y_df[Columns.cum_items_to_expire].iloc[0]]
    for idx in range(len(y_df)):

        if idx != 0:  # skip first row that has already been initialized
            # If condition is true, then we are still at the same SKU/BU combination.
            if (y_df[Columns.inv_bu].iloc[idx] == last_bu) and (y_df[Columns.item_id].iloc[idx] == last_item) and (
                    y_df[Columns.lot_id].iloc[idx] != 'NONLOT'):

                expire_diff = y_df[Columns.cum_items_to_expire].iloc[idx] - y_df[Columns.cum_items_to_expire].iloc[
                    idx - 1]
                expiring_items.append(expire_diff)

            else:  # Pointer arrived at new BU/SKU combination.
                # Update our "pointers" to new values.
                last_bu = y_df[Columns.inv_bu].iloc[idx]
                last_item = y_df[Columns.item_id].iloc[idx]
                expiring_items.append(y_df[Columns.cum_items_to_expire].iloc[idx])

    y_df[Columns.items_to_expire] = expiring_items

    # Finally, we need to make sure there are no values of items to expire
    # that are smaller than zero. There might be negative values due to the fact that
    # not every item from certain Lot will expire before being consumed.
    y_df.loc[y_df[Columns.items_to_expire] < 0, Columns.items_to_expire] = 0
    y_df.loc[y_df[Columns.days_to_expire] == 0, Columns.items_to_expire] = y_df[Columns.lot_qty_on_hand]
    y_df.loc[y_df[Columns.lot_id] == 'NONLOT', Columns.items_to_expire] = 0

    return y_df


def days_of_inv_with_transit(row: pd.Series) -> pd.Series:
    """Calculates the days of the given row.

    Parameters
    ----------
    row : pd.Series

    Returns
    -------
    Row based on conditional statement applied : pd.Series
    """
    if row[Columns.delta_doi] < 0:
        return row[Columns.bu_oh_plus_transit]

    return row[Columns.bu_oh]


def days_of_inventory(x_df: pd.DataFrame, consider_transit: bool = False):
    """Create new column with days of inventory.

    Days of inventory is a calculation based on the number of items in stock
    and the business unit average consumption rate for that item.

    Note
    ----
    If we have an average consumption rate of zero, we should have an
    infinite quantity of days of inventory. In reality, this means that
    no item is being consumed at that business unit and so, we should send all
    of those items to somewhere else.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.
    consider_transit: bool
        If True, consider items in transit, when calculating days of inventory, defaults to False.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with the new column added.

    """
    if consider_transit:
        x_df[Columns.days_of_inventory] = x_df.apply(lambda row: days_of_inv_with_transit(row), axis=1)
    else:
        x_df[Columns.days_of_inventory] = x_df[Columns.bu_qty_on_hand] / x_df[Columns.average_item_daily_use]

    x_df.loc[x_df[Columns.average_item_daily_use] == 0, Columns.days_of_inventory] = 0

    return x_df


def delta_days_of_inventory(x_df: pd.DataFrame):
    """
    Column with delta days of inventory.

    Create new column with the difference in days between
    the number of days of inventory and the days of inventory target for every
    item/business unit.

    Days of inventory is a calculation based on the number of items in stock
    and the business unit average consumption rate for that item.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with the new column added.

    """
    x_df[Columns.bu_doi_balance] = x_df[Columns.bu_qty_on_hand] - (x_df[Columns.doi_target] * x_df[Columns.average_item_daily_use])
    x_df[Columns.doi_balance] = x_df[Columns.bu_doi_balance] * x_df[Columns.lot_qty_on_hand]/x_df[Columns.bu_qty_on_hand]

    return x_df.fillna({Columns.doi_balance: 0})


def group_by(x_df: pd.DataFrame, group_cols: Union[str, list], agg_dict: dict) -> pd.DataFrame:
    """Perform group by operation on a given DataFrame.

    Parameters
    ----------
    x_df : pd.DataFrame
        Inventory report to be used at the ``groupby``
    group_cols : Union[str, list]
        column or list of columns to be used as **groupings**
    agg_dict : dict
        Dictionary with columns to be aggregated and their respective kind of aggregation.

    Returns
    -------
    Grouped dataframe: pd.DataFrame

    """
    return x_df.groupby(group_cols, as_index=False).agg(agg_dict)


def join(x_df: pd.DataFrame, y_df: pd.DataFrame, on: Union[str, list], how: str = 'left') -> pd.DataFrame:
    """Join dataframe.

    Used to combine two DataFrames based on common set of keys and a specified type of join.

    Parameters
    ----------
    x_df : pd.DataFrame
        Left side Dataframe
    y_df : pd.DataFrame
        Right side Dataframe
    on : Union[str, list]
        Column or list of columns to be used as keys.
    how : str
        How the join should be performed, by default 'left'.

    Returns
    -------
    Merged Dataframe : pd.DataFrame
    """
    return x_df.merge(y_df, how=how, on=on)


def last_transformations(x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Last column transformations.

    Makes sure that if business unit has average
    item daily use equal to zero, and items on
    hand that these number of items is used as
    doi balance.


    Applies last columns transformations.

    Parameters
    ----------
    x_df : pd.DataFrame
        Dataframe to be used.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe with last transformed columns.
    """
    x_df.loc[(x_df[Columns.average_item_daily_use] == 0)
             & (x_df[Columns.lot_qty_on_hand] > 0), Columns.doi_balance] = x_df[Columns.lot_qty_on_hand]

    x_df.loc[((x_df[Columns.average_item_daily_use] == 0)
              & (x_df[Columns.lot_qty_on_hand] > 0))
             | (x_df[Columns.average_item_daily_use].isna()), Columns.average_item_daily_use] = 0.0

    # If Business unit has lot quantity and these items are lot items (with expiring dates)
    # we consider all items as going to expire.
    x_df.loc[(x_df[Columns.average_item_daily_use] == 0)
             & (x_df[Columns.lot_qty_on_hand] > 0)
             & (x_df[Columns.flag_lot] is False), Columns.items_to_expire] = x_df[Columns.lot_qty_on_hand]

    x_df.loc[(x_df[Columns.average_item_daily_use] == 0)
             & (x_df[Columns.lot_qty_on_hand] > 0)
             & (x_df[Columns.flag_lot] is True), Columns.items_to_expire] = 0

    return x_df


def get_bu_granularity(x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get aggregated inventory report by business unit and item id granularity.

    Parameters
    ----------
    x_df : pd.DataFrame

    Returns
    -------
    Dataframe with business unit/item granularity : pd.DataFrame

    """
    agg_dict = {
        Columns.bu_qty_on_hand: 'mean',
        Columns.average_item_daily_use: 'mean',
        Columns.min_shipment_value: 'mean',
        Columns.price: 'mean',
        Columns.can_transfer_inventory: 'min',
        Columns.can_receive_inventory: 'min',
        Columns.doi_target: 'mean',
        Columns.flag_lot: 'max'
    }

    group_by_list = [Columns.inv_bu, Columns.item_id]

    return x_df.pipe(group_by, group_by_list, agg_dict)


def substitute_flag(x_df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Substitute a flag in a column.

    Changes column flags from ``Yes`` and ``No`` to ``1`` and ``0``, respectively.

    Parameters
    ----------
    x_df : pd.DataFrame
        Inventory report.
    column_name : str
        Column with flag to be substituted.
    Returns
    -------
    Dataframe with new flag type : pd.DataFrame

    """
    x_df.loc[(x_df[column_name] == 'Yes'), column_name] = 1
    x_df.loc[(x_df[column_name] == 'No'), column_name] = 0

    return x_df


def columns_error_handling(x_df: pd.DataFrame, column_list: list) -> pd.DataFrame:
    """
    Handle column name errors.

    For column names that were not found \
    in the original dataframe, we try to find If \
    the column is present but with a different name.

    We also handle errors if columns are not found for specific \
    column names that usually are not present in the non-lot dataframe.

    Parameters
    ----------
    x_df : pd.DataFrame
        Inventory report
    column_list : list
        List of columns that the method needs to find.

    Returns
    -------
        Pandas dataframe with all error handling transformations : pd.DataFrame

    """
    column_names = np.setdiff1d(column_list, x_df.columns)
    if constants.LOG_MODE:
        logging.warning("Some columns were not found at Dataframe: {}".format(column_names))
        logging.info("Trying to fix the problem. Please see the logs afterwards.")

    return (x_df
            .pipe(handle_name_matching_errors, column_names)
            .pipe(lambda x_df: x_df.pipe(create_column, Columns.expire_date,
                                         dt(2199, 12, 1)) if Columns.expire_date in column_names else x_df)
            .pipe(lambda x_df: x_df.pipe(create_column, Columns.min_shipment_value,
                                         MIN_SHIPMENT_VALUE) if Columns.min_shipment_value not in x_df.columns else x_df)
            .pipe(lambda x_df: (x_df
                                .pipe(create_column, Columns.lot_id, "NONLOT")
                                .pipe(create_column, Columns.lot_qty_on_hand,
                                      Columns.bu_qty_on_hand)) if Columns.lot_id in column_names else x_df))


def handle_name_matching_errors(x_df: pd.DataFrame, passed_vars: dict) -> pd.DataFrame:
    """Handles the process of trying to find the column name that corresponds to the one we're looking for.

    Parameters
    ----------
    x_df : pd.DataFrame
        Table with column names.
    passed_vars: dict
        Dictionary with column names that were not found on dataframe

    Returns
    -------
    Table with renamed columns if match was found : pd.DataFrame

    """
    # Possible names for columns
    possible_bu_names = ['business_unit', 'business_unit_id', 'bu_id']
    possible_price_names = ['std_price']
    possible_min_ship_names = ['min_transfer_value']
    possible_item_names = ['inv_item_id', 'sku_id']
    possible_report_names = ['query_date', 'month']
    possible_cons_names = ['item_consume', 'item_consumption', 'consume_rate']
    possible_avg_daily_use_names = ['average_daily_use',
                                    'avg_daily_use',
                                    'avg_day_use',
                                    'avg_day_usage',
                                    'avg_daily_usage',
                                    'average_daily_usage',
                                    'average_item_daily_usage',
                                    'avg_item_daily_usage',
                                    'avg_item_daily_use',
                                    'bu_item_avg_daily_usage']

    return (x_df
            .pipe(
        lambda x_df: x_df.pipe(try_possible_names, Columns.inv_bu, possible_bu_names) if Columns.inv_bu in passed_vars
        else x_df)
            .pipe(lambda x_df: x_df.pipe(try_possible_names, Columns.item_id,
                                         possible_item_names) if Columns.item_id in passed_vars
    else x_df)
            .pipe(lambda x_df: x_df.pipe(try_possible_names, Columns.consume,
                                         possible_cons_names) if Columns.consume in passed_vars
    else x_df)
            .pipe(lambda x_df: x_df.pipe(try_possible_names, Columns.report_date,
                                         possible_report_names) if Columns.report_date in passed_vars
    else x_df)
            .pipe(lambda x_df: x_df.pipe(try_possible_names, Columns.average_item_daily_use,
                                         possible_avg_daily_use_names) if Columns.average_item_daily_use in passed_vars
    else x_df)
            .pipe(
        lambda x_df: x_df.pipe(try_possible_names, Columns.price, possible_price_names) if Columns.price in passed_vars
        else x_df)
            .pipe(lambda x_df: x_df.pipe(try_possible_names, Columns.min_shipment_value,
                                         possible_min_ship_names) if Columns.min_shipment_value in passed_vars
    else x_df))


def try_possible_names(x_df: pd.DataFrame, column_name: str, possible_names: list) -> pd.DataFrame:
    """Rename column if found in list of possible names.

    The method tries to find the column name in list of
    specified possible names for that given column. If matching
    value is found it renames the column using the name
    specified at the column_name variable.

    When this method is called by the columns_error_handling
    procedure, we don't specify that this method needs to find the
    column. If no match is found, we simlpy pass as a result the x_df
    itself. I didn't add any exception if match is not found because not
    necessarily that imposes an error, but chances are, the model will
    return with error.

    Example
    -------
    - Column report_date was not found at x_df.columns.

    - So we loop through all the columns and try find if any of x_df columns has one of the following names:

        possible_names = ['query_date', ...]

    - Found that name inside x_df columns so we rename the column from  query_date to column_name (= report_date)

    Parameters
    ----------
    x_df : pd.DataFrame
    column_name : str
        New column name.
    possible_names : list
        List of possible column names.

    Returns
    -------
    Inventory report after trying to find the column amongst possible column names list : pd.DataFrame

    """
    for possible_name in possible_names:
        if possible_name in x_df.columns:
            x_df.rename(columns={possible_name: column_name}, inplace=True)

    return x_df


def rank_column(x_df: pd.DataFrame, new_column_name: str,
                group_column: Union[str, list], col_rank: str,
                rank_type: str = "dense", ascending: bool = True) -> pd.DataFrame:
    """Ranks column values by specified groups.

    Function divides inventory report into different column based groups.
    It then determines the specified column's rank relative to its
    group—the function returns the initial input table with an additional
    column with each row's ranking.

    Numerical data ranks go from 1 through n.

    Attention
    ---------
    This function is not used by the model anymore. It was used during development to simulate
    impact of transfer recommendations through time.

    Parameters
    ----------
    x_df : pd.DataFrame
        Table with columns to be ranked.
    new_column_name : str
        Name of the column that stores rank values.
    group_column : Union[str, list]
        Used to determine the groups for the groups in which to rank by.
    col_rank : str
        Column with values to be ranked.
    rank_type : str
        Defaults to ``dense``. How to rank the group of records that have the same value (i.e. tries):

            * **average:** average rank of the group

            * **min:** lowest rank in the group

            * **max:** highest rank in the group

            * **first:** ranks assigned in order they appear in the array

            * **dense:** like ``min``, but rank always increases by 1 between groups.
    ascending : bool, optional
        Whether or not the elements should be ranked in ascending order, defaults to ``True.``

    Returns
    -------
    pd.DataFrame : Returns Pandas ``Series`` or ``DataFrame`` with data ranks in the specified column.

    See Also
    --------
    `pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html <http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html>`_

    """
    # Making sure that the column we're trying to store ranking values into doesn't already exist.
    column_exists(x_df, column_name=new_column_name, condition=1)

    x_df[new_column_name] = x_df.groupby(group_column, as_index=False)[col_rank].rank(rank_type, ascending=ascending)

    return x_df


def delete_column(x_df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """delete a column from x_df

    Parameters
    ----------
    x_df : pd.DataFrame
    column_name : str

    Returns
    -------
    pd.DataFrame : Returns Pandas ``DataFrame`` without the deleted column

    """
    return x_df.drop([column_name], axis=1)


def fix_last_results_join(x_df: pd.DataFrame) -> pd.DataFrame:
    """Join the last results of the last query.

    Parameters
    ----------
    x_df : pd.DataFrame
        Removes ``null`` values that appear after combining transfer recommendations back to the inventory report.

    Returns
    -------
    pd.DataFrame : Dataframe with fixed join

    """
    return (x_df
            .pipe(fill_na, Columns.price, 'price_backup')
            .pipe(fill_na, Columns.average_item_daily_use, Columns.real_avg_consumption)
            .pipe(fill_na, Columns.can_transfer_inventory, 'can_transfer_inventory_backup')
            .pipe(fill_na, Columns.can_receive_inventory, 'can_receive_inventory_backup')
            .pipe(fill_na, Columns.doi_target, 'doi_target_backup')
            )


def rename_columns(x_df: pd.DataFrame, old_column_name: str, new_column_name: str) -> pd.DataFrame:
    """
    Rename one or more columns

    Parameters
    ----------
    x_df : pd.DataFrame
    old_column_name : str
    new_column_name : str

    Returns
    -------
    pd.DataFrame : dataframe with renamed columns

    """
    column_exists(x_df, old_column_name)  # Making old column exist
    column_exists(x_df, new_column_name, 1)  # Making sure new column is not used

    return x_df.rename({old_column_name: new_column_name}, axis=1)


def item_value_importance(x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate value of item based on all inventory.

    Calculate Decile for item total value
    (total dollar value of given item if you sum the quantity from every BU).

    For doing so, we create **3 new columns**:

    * ``item_value``: Total dollar value for every lot.

    * ``total_item_value``: Total dollar value for every item.

    * ``DecileRank``: Decile of ``total_item_value``.

    Parameters
    ----------
    x_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame : dataframe with new column ``normalized_item_value``

    Note
    ----
    If configuration for ``USE_DYNAMIC_TIME`` is enabled, this function
    allows the optimizer to determine the maximum amount of time that it can spend solving
    a single **item id** based on the monetary importance of that item in relation to other Item IDs.

    To enable or disable it, go to ``optimization.constants``.

        """
    # Calculate Decile for item total value (total dollar value of given item if you sum the quantity from every BU)
    x_df["item_value"] = x_df[Columns.price] * x_df[Columns.lot_qty_on_hand]
    x_df["total_item_value"] = x_df.groupby([Columns.item_id])["item_value"].transform(pd.Series.sum)
    x_df["normalized_item_value"] = (x_df["total_item_value"] - x_df["total_item_value"].min()) \
                                    / (x_df["total_item_value"].max() - x_df["total_item_value"].min())
    return x_df
