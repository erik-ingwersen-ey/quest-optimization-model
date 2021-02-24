"""Main module used for defining the scripts needed for running the optimization model.
"""
__version__ = '0.8'
__author__ = 'EY'

import logging
import os

import pandas as pd

##  Optimization Module Imports  #####################################
from optimization.constants import Columns
from optimization.constants import DEFAULT_TIME
from optimization.constants import LOG_MODE
from optimization.constants import MAX_TIME
from optimization.constants import USE_DYNAMIC_TIME
from optimization.datatools.extra_output import extra_output
from optimization.datatools.extra_output import qty_to_optimize
from optimization.datatools.extra_output import transfer_reason
from optimization.datatools.pipelines import data_pipeline
from optimization.logconfig import logconfiguration
from optimization.model.optimizer import OptimizationModel
from optimization.opt_tools.aux_funcs import save_results
from optimization.solspace import SolutionSpace

try:
    from tqdm import tqdm
    USE_TQDM = True

except ImportError:
    USE_TQDM = False

if LOG_MODE:
    logconfiguration()


##  Module Functions and Classes  ######################################################
class ModelOptimization:
    """Implements :meth:`optimization.model.optimizer.OptimizationModel<optimization.model.optimizer.OptimizationModel>` **class** for every ``item ID``.

    This class defines the optimization problem for every item on the inventory report and
    returns a combined DataFrame with all transfer recommendations.

    Returns
    -------
    pd.DataFrame
        Optimization Model transfer recommendations.

    Raises
    ------
    TypeError
        Raises error if ``df_inventory`` is not of type ``pd.DataFrame``.

    """
    transfer_value = "$ Value of Transfer"
    qty_to_optimize = {"Item ID": [], "Inventory Balance": [], "Items to Expire": [], transfer_value: [], }

    def __init__(self, df_inventory: pd.DataFrame, optimize_what: str, sku_qty: int = 0):

        self.df_inventory = df_inventory
        self.optimize_what = optimize_what
        self.sku_qty = sku_qty
        self._get_sku()

    def _get_solver_time(self, sku: object) -> int:
        """Get maximum amount of time optimizer can use optimizing a single ``Item ID``.

        Parameters
        ----------
        sku : object
            ``Item ID`` of SKU we want to optimize.

        Returns
        -------
        int : Total time the solver has to optimize each ``Item ID`` (row).
        """

        # If USE_DYNAMIC_TIME = True, determine maximum amount of time based on item dollar value.
        if USE_DYNAMIC_TIME:
            try:  # This option is not vital to process.
                # Therefore if it fails for some reason we tell the model to use the default time.
                self.df_inventory = self.item_value_decile()
                return self.dynamic_max_solver_time(sku)
            except ValueError:
                if LOG_MODE:
                    logging.info(
                        "Error trying to set maximum solver time dynamically. Using default time of {} s instead."
                            .format(DEFAULT_TIME))

        # If USE_DYNAMIC_TIME = False, we tell the model to use default value
        # for maximum time spent on a single item ID
        return DEFAULT_TIME

    def _get_sku(self):
        """Get SKU List from Inventory.

        Returns
        -------
        list : List of unique SKUs.

        """
        if not isinstance(self.df_inventory, pd.DataFrame):
            raise TypeError(
                "df_inventory should be pandas dataframe: %r" % self.df_inventory)

        self.sku_list = list(self.df_inventory[Columns.item_id].unique())

    def dynamic_max_solver_time(self, sku: object):
        """Determine max time to solve optimization dynamically.

        If ``USE_DYNAMIC_TIME`` is enabled, this function returns
        the time limit to spend at a single item ID.

        Parameters
        ----------
        sku : object
            ``Item ID`` that we're going to optimize.

        Returns
        -------
        int
            Value that ranges from ``MAX_TIME`` to **10%** of ``MAX_TIME``

        Tip
        ---
        To use this method to determine maximum amount of time the optimization model can send at each BU, you need to set \
        the attribute ``USE_DYNAMIC_TIME`` inside of :meth:`optimization.constants<optimization.constants>` to ``True``.

        """
        return MAX_TIME * (self.df_inventory[self.df_inventory[Columns.item_id] == sku]["DecileRank"].iloc[0] + 1) / 10

    def optimize(self, sku: object):
        """
        Run optimization for specific BU.

        When running for entire inventory, this method is called
        in a loop for every unique Item ID.

        This method determines maximum time the optimization model has
        to solve the problem, creates the solution space matrix and calls the
        the optimization model solver that returns as output table with the
        transfer recommendations.

        Parameters
        ----------
        sku : object
            ``Item ID`` that we're going to optimize.

        Returns
        -------
        opt_df : pd.DataFrame
            Pandas dataframe with optimization results. If none could be \
            obtained, then function returns empty.

        """
        solver_time = self._get_solver_time(sku)
        # Generating that item's solution matrix.
        # If no feasible solution space is found, the SolutionSpace doesn't return us any value.
        #   Example of unfeasible solution spaces:
        #       - Item IDs that exist only on one BU.
        #       - Item IDs that Don't have any item to expire and surplus in all inventories (or shortage).
        #       - Item IDs that all have and average daily consumption equal to zero....
        smatrix = SolutionSpace(self.df_inventory[self.df_inventory[Columns.item_id] == sku], sku).sol_matrix()

        # If SolutionSpace couldn't form the solution matrix, we skip to next item ID.
        if smatrix is not None:
            inv_balance, items_to_expire = qty_to_optimize(self.df_inventory, sku)

            # Trying to optimize the item by transferring among BU's
            optimization = OptimizationModel(smatrix, sku, self.optimize_what, solver_time)
            opt_df = optimization.solve()

            if opt_df is not None:
                self.qty_to_optimize["Item ID"].append(sku)
                self.qty_to_optimize["Inventory Balance"].append(inv_balance)
                self.qty_to_optimize["Items to Expire"].append(items_to_expire)
                self.qty_to_optimize[self.transfer_value].append(opt_df[self.transfer_value].sum())

                logging.info(
                    "Item ID: {}    Inventory Balance: ${:20,.2f}     Items to Expire: ${:20,.2f}      Optimized "
                    "Inventory: ${:20,.2f}  Objective: {} "
                        .format(
                        sku, inv_balance, items_to_expire, opt_df[self.transfer_value].sum(), optimization.opt_problem))
                return opt_df

    def run_all(self):
        """Run optimization model for entire inventory.

        Attributes
        ----------
        USE_TQDM : bool
            When set to True, adds progression bar at console running python.

        Returns
        -------
        optimization_list : list
            List of all optimization results.

        Tip
        ---
        ``TQDM`` will only be enabled if found in the Python environment.
        Else this attribute is set to ``False`` automatically.

        """

        if USE_TQDM:
            sku_list = tqdm(self.sku_list[:self.sku_qty]) if self.sku_qty != 0 else tqdm(self.sku_list)
        else:
            sku_list = self.sku_list[:self.sku_qty] if self.sku_qty != 0 else self.sku_list

        return [self.optimize(sku) for sku in sku_list]


def concat(results_list: list) -> pd.DataFrame:
    """Concatenates list of optimization results.

    Since the optimization model runs separately for every unique Item ID
    and outputs separate recommendation reports for every SKU, this function is used inside
    ``main`` to join all recommendations together into the final results.

    Parameters
    ----------
    results_list : (list)
        List with optimization results obtained.

    Returns
    -------
    pd.DataFrame
        Combined optimization model results.

    Raises
    ------
    TypeError
        Gives error if the function input ``results_list`` is not of type ``list``

    """
    if type(results_list) != list:
        raise TypeError(
            "results_list is not an list with optimization results: %r" % results_list)

    return pd.concat(results_list)


def main(lot_df: pd.DataFrame, nonlot_df: pd.DataFrame, optimize_what: str, save_res: bool = True, sku_qty: int = 0):
    """Combines all necessary steps for running the optimization model.

    This function calls all necessary procedures that are needed to perform the optimization
    This is the function that **Alteryx Server** uses to generate the optimization results.

    Parameters
    ----------
    lot_df : pd.DataFrame
        Input Inventory report for **Lot** Items.
    nonlot_df : pd.DataFrame
        Input Inventory report for **Non-Lot** Items.
    optimize_what : str
        Type of problem we want to optimize. Can be one of:

            * ``expire``: Optimize inventory for reducing **only** quantity of items to expire.

            * ``surplus``: Optimize inventory for reducing **only** quantity of surplus items.

            * ``both``: Optimize inventory for reducing **both** quantity of surplus items and items to expire at the same time.

            * ``experimental``: Experimental objective function that tries to formulate the bi-objective function in a way that is
                more complex but has potential to yield better results.

    save_res : bool, optional
        Used if user wants to save optimization results at the folder ``optimization/results``, by default True
    sku_qty : int, optional
        **Mainly used for debugging purposes**. Inform the quantity of unique item IDs the model should perform the optimization.
            If set to 0 runs the optimization for all Item IDs, by default 0

    Returns
    -------
    pd.DataFrame
        List with all ``transfer recommendations``. In production environment this Table is then passed on to Alteryx and saved as ``.xlsx`` file.

    Warn
    ----
    Do not use the experimental option on production environment. Experimental objective function is not 100% validated and might
    give incorrect recommendation transfers in some specific cases.

    """
    # ================== Loading Inventory Datasets ==================
    df_inventory = data_pipeline(lot_df, nonlot_df)

    # ================== Running Optimization ==================
    model = ModelOptimization(df_inventory, optimize_what, sku_qty)
    optimization_list = model.run_all()

    # ================== Combining Optimization Results ==================
    result = concat(optimization_list)
    if LOG_MODE:
        logging.info("Optimization Total Inventory Transfer Value: ${:20,.2f}"
                     .format(result["$ Value of Transfer"].sum()))

    # ================== Format Output ==================
    result = transfer_reason(df_inventory, result)

    # ================== Change dtype of some columns ==================
    result["Sender BU ID"] = result["Sender BU ID"].astype(int).astype(str)
    result["Item ID"] = result["Item ID"].astype(int).astype(str)
    result["Receiver BU ID"] = result["Receiver BU ID"].astype(int).astype(str)

    # ================== Saving Results ==================
    if save_res:
        from optimization import constants
        current_folder = os.path.dirname(os.path.abspath(constants.__file__))
        results_dir = os.path.join(current_folder, "Results")
        save_results(result, results_dir, 'optimization_results')
        save_results(pd.DataFrame(model.qty_to_optimize), results_dir, 'optimizable_inventory')

    extra_sheet = extra_output(df_inventory)
    """Extra sheet required for generating dashboard"""

    return result, extra_sheet
