"""Constants used throughout optimization model scripts.
"""

LOG_MODE = True  # enables event logging
"""Save runtime logs
"""

SAVE_MODEL = False  # Saves optimization model
"""Save optimization model, by default is set to False, since in production environment optimization results are
passed back to alteryx that handles the results saving process
"""

USE_DYNAMIC_TIME = False
"""Determine maximum amount of time optimization model can spend optimizing a single item
based on the decile of that item dollar value. If set to False, main.py use the default time as limit to the model.
"""

MAX_TIME = 500  # Decile 10 will be given MAX_TIME value (if USE_DYNAMIC_TIME=True)
"""Maximum amount of time in seconds that solver can spend on single item id. This value is used only
when USE_DYNAMIC_TIME is set to True
"""

DEFAULT_TIME = 30
"""Default value in seconds that solver can spend on single item id
"""

## * FOR ALTERYX PURPOSES  #####################################
# If you specify Lot and Non Lot input files to python using Alteryx
# You can call these connections and reference them inside Python Alteryx API
# to the variables below. If you do so, the model will automatically recognize that
# There is no need for it to search the input files at the @param INPUT_FOLDER.
# ? How to do that?
#  (1) Connect both Input Data connections to python Client
#  (2) Inside python client, add to your script:

# > from ayx import Alteryx
# > from optimization import constants
# > constants.LOT_DF = Alteryx.read("#2") # Assuming second connection is Lot input file
# > constants.NONLOT_DF = Alteryx.read("#1")
# > ...
LOT_DF = None
NONLOT_DF = None

## * COLUMNS ERROR HANDLING *  #####################################
# If no column is found for minimum shipment value, then input_data pipeline
# creates new column and adds default value that can be set by this variable
# ! * OBSERVATION:* Consider removing it from production environment and instead returning
# ! error if no values are found.
MIN_SHIPMENT_VALUE = 50
"""Default value for minimum shipment in dollars if values are not found on the inventory report
"""

class Columns:
    """Specify inventory input report column names.

    Below you can find **all names of columns** that are either **created or used** by the model.

    Attributes
    ----------
    inv_bu : str
        Name of column from inventory report that stores the ID of each business unit
    item_id : str
        Name of column from inventory report with ID from every item
    lot_id : str
        Name of column from inventory report with ID from every lot
    report_date : str
        Name of column from inventory report with date the report was generated
    expire_date : str
        Name of column from inventory report with lot expiration date
    average_item_daily_use : str
        Name of column from inventory report that stores the average item daily usage
    bu_qty_on_hand : str
        Name of column from inventory report with total quantity of items at each business unit
    lot_qty_on_hand : str
        Name of column from inventory report with quantity of items from a specific lot at each business unit
    doi_target : str
        Name of column from inventory report that tells us the target quantity of days each of inventory each item
        at every business unit should have
    days_of_inventory : str
        Name of column to store the calculated field values with quantity of days of inventory for
        for every item at each business unit
    cum_doi : str
        Name of column to store the calculated field values with cumulative quantity of days of inventory for
        for every lot on a business unit
    cum_doi_balance : str
        Name of column to store the calculated field values with cumulative balance of days of inventory for
        for every lot on a business unit
    doi_balance : str
        Name of column to store calculated values with balance of days of inventory each business unit has
        for every item
    bu_doi_balance : str
        Name of column to store calculated values with balance of days of inventory for each item from business units
    delta_days_of_inventory : str
        Name of column from inventory report that
    days_to_expire : str
        Name of column to store calculated values for quantity of days left before a given lot at each business unit
        expires
    items_to_expire : str
        Name of column to store calculated values of quantity of items to expire based on current consumption rate and
        days left before the item expiration date
    can_transfer_inventory : str
        Name of column from inventory report that specifies if a given business unit can transfer certain item
    can_receive_inventory : str
        Name of column from inventory report that specifies if a given business unit can receive certain item
    min_shipment_value : str
        Name of column from inventory report that gives us the minimum dollar value that each transfer needs to be
    price : str
        Name of column from inventory report with the unit price in dollars for each item
    cum_stock : str
        Name of column to store the calculated field values with cumulative quantity of items a given Item
        at each business unit
    cum_items_to_expire : str
        Name of column to store the calculated field values with cumulative quantity of items to expire for
        a given Item at each business unit
    receiver_bu : str
        Name of column to store the id for receiver business units on the transfer recommendation report
    provider_bu : str
        Name of column to store the id for provider business units on the transfer recommendation report
    optimization_transfer : str
        Name of column from inventory report that
    default_shipment_days : str
        Name of column from inventory report with the default shipment days to consider when recommending transfers
    how_many_bu : str
        Name of column used to store calculated value for quantity of business units that have a given item
    delta_doi : str
        Name of column used to store how far from target days of inventory for each item id,
        the business unit is currently
    bu_oh_plus_transit : str
        Name of column from inventory report that
    bu_oh : str
        Name of column from inventory report that
    bu_item_qty_in_transf : str
        Name of column from inventory report that informs the quantity of items currently in transit
    bu_description : str
        Name of column from inventory report with business unit description (long name)
    bu_region : str
        Name of column from inventory report with business unit region
    contact_email : str
        Name of column from inventory report with contact email from each business unit supervisor
    on_site_email : str
        Name of column from inventory report with business unit contact email
    item_description : str
        Name of column from inventory report with item description
    supplier_name : str
        Name of column from inventory report with name of the item supplier
    std_uom : str
        Name of column from inventory report with std_uom
    chart_of_accounts : str
        Name of column from inventory report with chart of accounts information
    bu_address : str
        Name of column from inventory report with business unit address
    minimum_days_of_inventory_for_lot : str
        Name of column from inventory report that defines the minimum days of inventory threshold required for each item
         at each business unit
    date_lot_added_to_bu_inv : str
        Name of column from inventory report with date each lot was added to each business unit
    bu_item_last_lot_depleted : str
        Name of column from inventory report that stores information about which lot is current lot in use
    item_stats : str
        Name of column from inventory report that stores the item status for each item ID at each business unit

    Hint
    ----
    If the **names of the columns used by the model change, there is no need to go through all modules and change manually**.
    Just **add their new name to their respective field.**

    Tip
    ---
    If new columns need to be specified to the model, place them in this script as the other columns.
    Place for specifying column names used on Inventory Dataframe.

    """
    inv_bu = 'inv_bu'
    item_id = 'item_id'
    lot_id = 'lot_id'
    report_date = 'report_date'
    expire_date = 'expire_date'
    average_item_daily_use = 'average_item_daily_use'
    temp_avg_daily_use = 'temp_avg_daily_use'
    flag_avg_daily_use = 'flag_avg_daily_use'
    bu_qty_on_hand = 'bu_qty_on_hand'
    lot_qty_on_hand = 'lot_qty_on_hand'
    doi_target = 'doi_target'
    days_of_inventory = 'days_of_inventory'
    cum_doi = 'cumulative_days_of_inventory'
    cum_doi_balance = 'cumulative_doi_balance'
    doi_balance = 'doi_balance'
    bu_doi_balance = 'bu_doi_balance'
    delta_days_of_inventory = 'delta_days_of_inventory'
    days_to_expire = 'days_to_expire'
    items_to_expire = 'items_to_expire'
    can_transfer_inventory = 'can_transfer_inventory'
    can_receive_inventory = 'can_receive_inventory'
    min_shipment_value = 'min_shipment_value'
    price = 'price'
    cum_stock = 'cumulative_item_qty'
    cum_items_to_expire = 'cumulative_items_to_expire'
    flag_lot = 'flag_non_lot'
    consume = 'consume'
    period = 'period'
    receiver_bu = 'receiver_bu'
    provider_bu = 'provider_bu'
    bu_qty_end_month = 'bu_qty_end_month'
    optimization_transfer = 'optimization_transfer'
    real_avg_consumption = 'real_avg_consumption'
    default_shipment_days = 'default_shipment_days'

    #### Calculated Columns
    how_many_bu = 'how_many_bu'

    #### New Columns
    delta_doi = 'delta_doi'
    bu_oh_plus_transit = 'bu_oh_transit_days_supply'
    bu_oh = 'bu_oh_days_supply'

    #### Output Columns
    bu_item_qty_in_transf = 'bu_item_qty_in_transfer'
    bu_description = 'bu_descrip'
    bu_region = 'bu_region'
    contact_email = 'approver_contact_email'
    on_site_email = 'onsite_contact_email'
    item_description = 'item_descrip'
    supplier_name = 'supplier_name'
    std_uom = 'std_uom'
    chart_of_accounts = 'chart_of_accounts'
    bu_address = 'bu_address'
    minimum_days_of_inventory_for_lot = 'minimum_days_of_inventory_for_lot'
    date_lot_added_to_bu_inv = 'date_lot_added_to_bu_inv'
    bu_item_last_lot_depleted = 'bu_item_last_lot_depleted'

    #### Columns Inventory report filtering
    item_stats = "bu_item_status"

    # ADD COLUMNS TO BE USED BY THE MODEL ON THE LIST THAT FOLLOWS
    cols_needed = [
        inv_bu, item_id, lot_id, report_date, expire_date, price,
        average_item_daily_use, bu_qty_on_hand, lot_qty_on_hand, doi_target,
        min_shipment_value, can_transfer_inventory, can_receive_inventory,
        delta_doi, bu_oh_plus_transit, bu_oh, item_description,
        bu_region, contact_email, supplier_name, on_site_email,
        std_uom, chart_of_accounts, bu_address, bu_description,
        item_stats, bu_item_qty_in_transf, minimum_days_of_inventory_for_lot,
        date_lot_added_to_bu_inv, default_shipment_days, bu_item_last_lot_depleted,]
    """List of columns needed, either to calculate necessary information used by the optimization model or
    columns that are required to be added to final report
    """
