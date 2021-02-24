"""Scripts used for generating the **solution space.**

Python module used to generate the solution space that is then used by ``optimizer`` module to generate the
optimization transfer recommendations.


In simple terms this module contains one class named :obj:`optimization.solspace.SolutionSpace` that is used to:

    (1) Filter inventory dataframe for the SKU that we're analyzing;

    (2) Make one aggregation at a BU granularity for determining providing BU's and another at Lot ID granularity for determining receiving BU's/Lot ID's

    (3) Query both grouping results, filtering for BU's that don't have items to expire (those will be the receiver BU's) and for those BU's that have either items to expire or surplus inventory consider them as providers.

    (4) Create the solution space matrix adding column-wise, receiving BU's and row-wise providing  BU's/Lot ID's'.

    (5) Add extra columns to the solution matrix that will be used on the optimization model constraints or objective function definitions.

"""

from typing import Union

import numpy as np
import pandas as pd

from optimization.constants import Columns
from optimization.datatools.dataprep import group_by
from optimization.opt_tools.aux_funcs import round_values

PROVIDING_COLUMNS = {
    Columns.inv_bu: 0,
    Columns.doi_balance: 1,
    Columns.min_shipment_value: 2,
    Columns.price: 3,
    Columns.average_item_daily_use: 4,
    Columns.items_to_expire: 5,
    Columns.days_to_expire: 6,
    Columns.doi_target: 7,
    Columns.lot_qty_on_hand: 8,
    Columns.lot_id: 9,
    Columns.bu_doi_balance: 10,
    Columns.delta_doi: 11,
    Columns.bu_oh_plus_transit: 12,
    Columns.bu_oh: 13,
    Columns.item_description: 14,
    Columns.bu_region: 15,
    Columns.contact_email: 16,
    Columns.supplier_name: 17,
    Columns.std_uom: 18,
    Columns.chart_of_accounts: 19,
    Columns.bu_address: 20,
    Columns.bu_description: 21,
    Columns.on_site_email: 22,
    Columns.expire_date: 23,
    Columns.bu_qty_on_hand: 24,
    Columns.bu_item_qty_in_transf: 25,
    Columns.default_shipment_days: 26,
    Columns.can_transfer_inventory: 27,  # DELETE AFTER (FOR TESTS ONLY)
    Columns.can_receive_inventory: 28,  # DELETE AFTER (FOR TESTS ONLY)
    Columns.item_stats: 29,  # DELETE AFTER (FOR TESTS ONLY)
}

RECEIVING_COLUMNS = {
    Columns.inv_bu: 0,
    Columns.bu_doi_balance: 1,
    Columns.min_shipment_value: 2,
    Columns.price: 3,
    Columns.average_item_daily_use: 4,
    Columns.doi_target: 5,
    Columns.bu_qty_on_hand: 6,
    Columns.delta_doi: 7,
    Columns.bu_oh_plus_transit: 8,
    Columns.bu_oh: 9,
    Columns.item_description: 10,
    Columns.bu_region: 11,
    Columns.contact_email: 12,
    Columns.supplier_name: 13,
    Columns.std_uom: 14,
    Columns.chart_of_accounts: 15,
    Columns.bu_address: 16,
    Columns.bu_description: 17,
    Columns.on_site_email: 18,
    Columns.items_to_expire: 19,
    Columns.bu_item_qty_in_transf: 20,
    Columns.default_shipment_days: 21,
    Columns.can_transfer_inventory: 22,
    Columns.can_receive_inventory: 23,
    Columns.item_stats: 24,
}


class SolutionSpace:
    """Creates a basic solution space for the given inventory.

    This class uses the module constants ``PROVIDING_COLUMNS`` and ``RECEIVING_COLUMNS``
    for mapping additional values to the solution space matrix rows and columns
    that might be used by the optimization model in the next step.

    **Receivers**

    - *Granularity:* ``BU ID``

    - *Conditions:*

        - BU can't have items to expire. It might have surplus though.

        - Adding BU's with surplus at the receiving list should \
        not affect the model, because of the **elastic constraints**. Transferring \
        items to BU's that already have surplus (essentially making that BU surplus even higher) \
        penalize the objective function, therefore the model will only do so if it minimizes items to expire.

    **Providers**

        - *Granularity:* ``Lot ID``

        - *Conditions:*

            - Needs to have items to expire (Lot-wise) or the BU has surplus.

            - The total monetary value of possible items that can be transferred \
            needs to be greater than minimum shipment value.

    Attributes
    ----------
    EXTRA_COLUMNS: int
        Quantity of extra columns that store information about ``Providing
        BU's`` that are either used to define the **optimization problem** or are used in the **output file.**

    EXTRA_ROWS: int
        Quantity of extra columns that store information about ``Receiving BU's`` that are either used to define the
         **optimization problem** or are used in the **output file.**

    Note
    -----
    For minimum shipment value we're considering total monetary value of the Lot
    and not the BU. Since items from different Lots but from from the same BU can
    be transferred, we theoretically should perform this filter using BU total values.
    We're not doing so at the moment for 2 reasons: first there aren't so many items
    with price bellow :math:`$50.00` and even if the model filters out those values the total
    amount that we're not considering in the optimization is very small in comparisson
    to the rest of the SKUs.

    See Also
    --------

    `coin-or.github.io/pulp/guides/how_to_elastic_constraints.html <http://coin-or.github.io/pulp/guides/how_to_elastic_constraints.html>`_

    `royalsocietypublishing.org/doi/10.1098/rsta.2007.2122 <http://royalsocietypublishing.org/doi/10.1098/rsta.2007.2122>`_

    `en.wikipedia.org/wiki/Constrained_optimization <http://en.wikipedia.org/wiki/Constrained_optimization>`_

    """
    EXTRA_COLUMNS = len(PROVIDING_COLUMNS)
    EXTRA_ROWS = len(RECEIVING_COLUMNS)

    PROVIDING_GROUPBY = [Columns.inv_bu, Columns.lot_id]
    RECEIVING_GROUPBY = [Columns.inv_bu]

    RECEIVING_AGG = {
        Columns.bu_doi_balance: 'mean',
        Columns.min_shipment_value: 'mean',
        Columns.price: 'mean',
        Columns.doi_target: 'mean',
        Columns.bu_qty_on_hand: 'mean',
        Columns.items_to_expire: 'sum',
        Columns.average_item_daily_use: 'mean',
        Columns.can_transfer_inventory: 'min',
        Columns.can_receive_inventory: 'min',
        Columns.delta_doi: 'mean',
        Columns.bu_oh_plus_transit: 'mean',
        Columns.bu_oh: 'mean',
        Columns.item_description: 'first',
        Columns.bu_region: 'first',
        Columns.contact_email: 'first',
        Columns.supplier_name: 'first',
        Columns.std_uom: 'first',
        Columns.chart_of_accounts: 'first',
        Columns.bu_address: 'first',
        Columns.bu_description: 'first',
        Columns.on_site_email: 'first',
        Columns.bu_item_qty_in_transf: 'mean',
        Columns.default_shipment_days: 'mean',
        Columns.item_stats: 'first',
    }

    PROVIDING_AGG = {
        Columns.doi_balance: 'sum',
        Columns.bu_doi_balance: 'mean',
        Columns.min_shipment_value: 'mean',
        Columns.price: 'mean',
        Columns.doi_target: 'mean',
        Columns.bu_qty_on_hand: 'mean',
        Columns.lot_qty_on_hand: 'sum',
        Columns.items_to_expire: 'sum',
        Columns.days_to_expire: 'mean',
        Columns.average_item_daily_use: 'mean',
        Columns.can_transfer_inventory: 'min',
        Columns.can_receive_inventory: 'min',
        Columns.delta_doi: 'mean',
        Columns.bu_oh_plus_transit: 'mean',
        Columns.bu_oh: 'mean',
        Columns.expire_date: 'first',
        Columns.item_description: 'first',
        Columns.bu_region: 'first',
        Columns.contact_email: 'first',
        Columns.supplier_name: 'first',
        Columns.std_uom: 'first',
        Columns.chart_of_accounts: 'first',
        Columns.bu_address: 'first',
        Columns.bu_description: 'first',
        Columns.on_site_email: 'first',
        Columns.bu_item_qty_in_transf: 'mean',
        Columns.default_shipment_days: 'mean',
        Columns.item_stats: 'first',
    }

    def __init__(self, df_inventory: pd.DataFrame, item_id: object):
        """SolutionSpace class initialization

        Parameters
        ----------
        df_inventory : pd.DataFrame
            Inventory table
        item_id : obj
            Item ID being analyzed

        """

        self.df_inventory = df_inventory
        self.item_id = item_id

    def _create_matrix(self, providing_list: pd.DataFrame, receiving_list: pd.DataFrame) -> np.ndarray:
        """This method is used to create the solution space matrix.

        Parameters
        ----------
        providing_list : pd.DataFrame
            List with providing BU ID's.
        receiving_list : pd.DataFrame
            List with receiving BU and Lot ID's.

        Returns
        -------
        sol_space : np.ndarray
            Numpy array to be used in the optimization model.

        """
        return np.zeros((len(providing_list) + self.EXTRA_ROWS,
                         len(receiving_list) + self.EXTRA_COLUMNS), dtype=object)

    def _get_bu_list(self, group_by_list: list, agg_dict: dict) -> pd.DataFrame:
        """
        Filter for BU's that have the item that we're trying to optimize and
        groups values according to the specified ``group_by_list`` and ``agg_dict``.

        Parameters
        ----------
        group_by_list : list
            List of columns to be used at the groupby.
        agg_dict : dict
            Dictionary with columns and aggregation type to be used at the groupby.

        Returns
        -------
        pd.DataFrame : pd.DataFrame
            DataFrame with filtered and grouped values.

        """
        return (self.df_inventory
                .query("item_id == " + str(self.item_id))
                .pipe(group_by, group_by_list, agg_dict))

    @staticmethod
    def _rounding_columns(x_df: pd.DataFrame) -> pd.DataFrame:
        """Round the values of the columns to the nearest 2 - byte order.

        Parameters
        ----------
        x_df : pd.DataFrame
            Dataframe to be used for performing the pipe transformations.

        Returns
        -------
        pd.DataFrame: pd.DataFrame
            Transformed dataframe.

        """
        return (x_df
                .pipe(round_values, Columns.bu_doi_balance, 2)
                .pipe(round_values, Columns.average_item_daily_use, 7))

    def _get_receiving(self, sku_rec: pd.DataFrame) -> pd.DataFrame:
        """
        Get list of receiving BU's.

        Parameters
        ----------
        sku_rec : pd.DataFrame
            Pandas dataframe with possible receiving BU's.

        Returns
        -------
        pd.DataFrame: pd.DataFrame
            BU's to be considered as receivers.

        """
        return (sku_rec
                .pipe(self._rounding_columns)
                .query("bu_item_status == 'Active'"))

    def _get_providing(self, sku_prov: pd.DataFrame) -> pd.DataFrame:
        """Get list of providing BU's by Lot.

        Parameters
        ----------
        sku_prov : pd.DataFrame
            Pandas dataframe with possible providing BU's.

        Returns
        -------
        pd.DataFrame: pd.DataFrame
            BU's to be considered as providers.

        """
        return (sku_prov
                .pipe(self._rounding_columns)
                .pipe(lambda sku_rec: sku_rec.query("can_transfer_inventory > 0")))

    def _populate_solspace(self, sol_space: np.matrix, bu_prov: pd.DataFrame, bu_rec: pd.DataFrame) -> pd.DataFrame:
        """Populate the sol_space 2-D array with the given data for provider and receiver BU's.

        Parameters
        ----------
        sol_space : np.matrix
            Final solution space to be used in the next step
            by the optimization model
        bu_prov : pd.DataFrame
            BU's to be considered as providers.
        bu_rec : pd.DataFrame
            BU's to be considered as receivers.

        Returns
        -------
        pd.DataFrame : pd.DataFrame
            Solution space with added extra columns.

        See Also
        --------
        ``PROVIDING_COLUMNS`` for information of what extra columns for providing BU's
        are being added to the solution matrix.

        ``RECEIVING_COLUMNS`` for information of what extra columns for receiving BU's
        are being added to the solution matrix.

        If you want to add new columns to the solution matrix, change both dictionaries accordingly
        and the rest of the code will adapt automatically.

        """
        # Extra providing BU columns
        for idx in range(len(bu_prov)):
            for key, value in PROVIDING_COLUMNS.items():
                sol_space[idx + self.EXTRA_ROWS, value] = bu_prov[key].iloc[idx]

        # Extra receiving BU columns.
        for idx in range(len(bu_rec)):
            for key, value in RECEIVING_COLUMNS.items():
                sol_space[value, idx + self.EXTRA_COLUMNS] = bu_rec[key].iloc[idx]

        return sol_space

    def sol_matrix(self) -> pd.DataFrame:
        """
        Create the solution space "matrix".

        Based on the item ID of our inventory database, it
        creates the matrix that contains in the first column
        all the BU's that can provide items and at the first
        row all BU's that can receive items.

        Returns
        -------
        sol_space : pd.DataFrame
            Numpy matrix to be used in the optimization model.

        """
        sku_prov = self._get_bu_list(self.PROVIDING_GROUPBY, self.PROVIDING_AGG)
        bu_prov = self._get_providing(sku_prov)

        sku_rec = self._get_bu_list(self.RECEIVING_GROUPBY, self.RECEIVING_AGG)
        bu_rec = self._get_receiving(sku_rec)

        if (SolutionSpace.filter_solspace(bu_prov, bu_rec)) and (len(bu_rec) >= 1 and len(bu_prov) >= 1):
            # Creating the base matrix to store the solution matrix.
            sol_space = self._create_matrix(bu_prov, bu_rec)
            sol_space = self._populate_solspace(sol_space, bu_prov, bu_rec)

            return sol_space

    @staticmethod
    def filter_solspace(provider: pd.DataFrame, receiver: pd.DataFrame) -> Union[None, bool]:
        """Apply check to solution matrix to check if Item ID can be optimized.

        This filter runs after we obtain ``bu_prov`` and ``bu_rec`` to check if there
        are any BU's that can be optimized. This check is done by verifying the following conditions:

        - ``receiver[Columns.average_item_daily_use].sum() != 0``: If all receiver BU's have an average \
        consumption rate of zero, the are no possible BU's that will reduce inventory balance or items to expire

        Parameters
        ----------
        provider : pd.DataFrame
            DataFrame with information about all providing lots from all business units.
        receiver : pd.DataFrame
            DataFrame with information about all receiving business units.

        Returns
        -------
        Union[None, bool]
            If all tests were passed, returns ``True`` else returns nothing

        """
        rec_avg = receiver[Columns.average_item_daily_use].sum()
        prov_bu_qty = provider[Columns.bu_qty_on_hand].sum()

        if (rec_avg != 0) or (prov_bu_qty != 0):
            return True
