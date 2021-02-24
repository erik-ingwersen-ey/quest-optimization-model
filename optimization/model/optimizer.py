"""Main script for generating transfer recommendations.

Module used to define the optimization problem that in general is composed by a combination of constraints
and objective function.

"""
import logging
import os
import time
from typing import Union

import numpy as np
import pandas as pd
import pulp as plp
from pulp.apis.core import PulpSolverError

from optimization import constants
from optimization import solspace
from optimization.opt_tools.load_data import get_file_extension

RECEIVING_COLUMNS = solspace.RECEIVING_COLUMNS
"""Receiver BU columns
"""

PROVIDING_COLUMNS = solspace.PROVIDING_COLUMNS
"""Sender/Provider BU columns
"""

COLUMNS_TO_EXPORT = [
    "Sender BU ID",
    "Sender BU Description",
    "Sender BU Region",
    "Sender BU Approver Contact Email",
    "Sender BU Onsite Contact Email",
    "Sender BU's SKU Current DOI",
    "Sender BU's SKU Target DOI",
    "Sender Average Weekly Use",
    "Sender BU On Hand Quantity",
    "Sender BU in Transit Quantity",
    "Sender BU Approves transfer? (YES/NO)",
    "Item ID",
    "Item Description",
    "Supplier Name",
    "Source Lot ID",
    "Item-Lot Expiry Date",
    "Item Quantity to transfer",
    "STD UOM",
    "$ Value of Transfer",
    "Transfer Recommendation Reason",
    "PeopleSoft Transfer ID",
    "Receiver BU ID",
    "Receiver BU Description",
    "Receiver BU Region",
    "Receiver BU Approver Contact Email",
    "Receiver BU Onsite Contact Email",
    "Receiver BU's SKU Current DOI",
    "Receiver BU's SKU Target DOI",
    "Receiver Average Weekly Use",
    "Receiver BU On Hand Quantity",
    "Receiver BU in Transit Quantity",
    "Receiver BU Approves transfer? (YES/NO)",
    "Chart of Accounts",
    "Receiver BU Address",
    "Sender Lot Items to Expire",]
"""
Columns required at the final output report
"""

COLUMNS_RENAME = {
    "bu_provide": "Sender BU ID",
    "bu_doi_balance": "Sender BU Inventory Balance",
    "max_provide": "Sender Lot Inventory Balance",
    "avg_cons_prov": "Sender Average Daily Use",
    "max_expire": "Sender Lot Items to Expire",
    "days_expire": "Days to Expire",
    "target_provide": "Sender BU's SKU Target DOI",
    "inv_provide": "Sender Lot Qty on Hand",
    "provide_delta_doi": "Sender Delta DOI",
    "provide_bu_oh_plus_transit": "Sender BU OH + Transit Days Supply",
    "provide_days_of_inventory": "Sender BU's SKU Current DOI",
    "item_id": "Item ID",
    "lot_id": "Source Lot ID",
    "transfer_value": "$ Value of Transfer",
    "solution_value": "Item Quantity to transfer",
    "price_prov": "Price",
    "min_ship_prov": "Minimum Shipment Value",
    "bu_receive": "Receiver BU ID",
    "max_receive": "Receiver BU Inventory Balance",
    "avg_cons": "Receiver Average Daily Use",
    "target_receive": "Receiver BU's SKU Target DOI",
    "inv_receive": "Receiver BU On Hand Quantity",
    "receive_delta_doi": "Receiver Delta DOI",
    "receive_bu_oh_plus_transit": "Receiver BU OH + Transit Days Supply",
    "receive_days_of_inventory": "Receiver BU's SKU Current DOI",
    "provide_item_descrip": "Item Description",
    "provide_bu_descrip": "Sender BU Description",
    "provide_bu_region": "Sender BU Region",
    "provide_contact_email": "Sender BU Approver Contact Email",
    "provide_supplier_name": "Supplier Name",
    "provide_std_uom": "STD UOM",
    "receive_chart_of_accounts": "Chart of Accounts",
    "receive_bu_descrip": "Receiver BU Description",
    "receive_bu_region": "Receiver BU Region",
    "receive_contact_email": "Receiver BU Approver Contact Email",
    "receive_bu_address": "Receiver BU Address",
    "receive_on_site_email": "Receiver BU Onsite Contact Email",
    "provide_on_site_email": "Sender BU Onsite Contact Email",
    "weekly_avg_cons_prov": "Sender Average Weekly Use",
    "weekly_avg_cons": "Receiver Average Weekly Use",
    "provide_expire_date": "Item-Lot Expiry Date",
    "provide_bu_qty_on_hand" : "Sender BU On Hand Quantity",
    "provide_bu_item_qty_in_transf": "Sender BU in Transit Quantity",
    "receive_bu_item_qty_in_transf": "Receiver BU in Transit Quantity",}
"""Rename columns with their names to be added to generated output report
"""


class ObjectiveFunction:
    """Class used for defining the objective functions"""

    def __init__(self):
        pass  # no initialization argument is currently being used

    def hasattr(self, attr):
        """Looks for a given attribute ``attr`` inside ``__dict__``

        Parameters
        ----------
        attr : object
            Attribute we're trying to find.

        Returns
        -------
        attr : object
            Method returns the Attribute itself if found.

        Note
        ----
        :meth:`ObjectiveFunction<optimization.model.optimizer.ObjectiveFunction>` is the ``base class`` of \
        :meth:`OptimizationModel<optimization.model.optimizer.OptimizationModel>`. We created this method because \
        the ``base class`` inherits attributes and methods from its child class and we want to know, specifically \
        if inside the base class, there is an attribute named ``opt_problem``. If there is, we automatically set \
        the **objective function** to our optimization problem without having to call any method \
        (right at the instantiation of this ``base class``)

        """
        if attr in self.__dict__.keys():
            return attr

    def __setattr__(self, attr, val):

        opt_problem = self.hasattr("opt_problem")
        if opt_problem and (attr == "_objective_function"):
            self._validate(opt_problem)
            self.__dict__[attr] = self.set_objective()
        else:
            self.__dict__[attr] = val

    def __getattr__(self, attr):

        if attr == '_objective_function':
            self.__dict__[attr] = self.set_objective()

        return self.__dict__[attr]

    @staticmethod
    def _validate(opt_problem: str):
        """Assert opt_problem is one of possible choices.

        Parameters
        ----------
        opt_problem : str
            Type of objective we want to optimize. Can be:

                * **expire:** minimize items to expire
                * **surplus:** minimize surplus
                * **both:** minimize both surplus and items to expire
                * **experimental:** bi-objective function that uses different configurable \
                weights for inventory balance and items to expire.

        Raises
        ------
        AttributeError
            We don't have an objective function for the type of
            problem that was specified or someone mistyped its name.

        """
        if opt_problem not in ['expire', 'surplus', 'both', 'experimental']:
            raise AttributeError(
                "opt_problem must be equal to 'expire' or 'surplus' or 'both. {} was passed".format(opt_problem))

    def _surplus_objective(self):
        """Define surplus minimization objective.

        We want it to transfer the maximum number of items as possible, given the applied constraints.

        .. math::
            min\\,\\bigg ( \\,\\sum_{j=1}^{n2}\\,Inv. Balance(\\,j\\,)\\,+\\,(\\,\\sum_{i=1}^{n1}\\,\\,  (\\,x_i\\,) \\bigg )

        """
        objective = 0

        for row_idx in self.set_i:

            trans_val = plp.lpSum(self.x_vars[row_idx, col_idx] for col_idx in self.set_j)
            after_transfer = self.max_provide[row_idx] - trans_val

            if self.max_provide[row_idx] < 0:
                after_transfer = -1 * after_transfer

            objective += after_transfer

        for col_idx in self.set_j:

            trans_val = plp.lpSum(self.x_vars[row_idx, col_idx] for row_idx in self.set_i)
            after_transfer = self.max_receive[col_idx] + trans_val

            if self.max_receive[col_idx] < 0:
                after_transfer = -1 * after_transfer

            objective += after_transfer

        # We want to minimize the difference between transferred items and items to expire or surplus
        self.prob.sense = plp.LpMinimize
        self.prob.setObjective(objective)

    def _expire_objective(self):
        """Define items to expire minimization objective.

        .. math::
            min\\, \\bigg ( \\,\\sum_{j=1}^{n2}\\,ITE(\\,j\\,)\\,-\\,(\\,\\sum_{i=1}^{n1}\\,\\,  (\\,x_i\\,) \\bigg  )

        """
        objective = plp.lpSum(
            self.max_expire[row_idx]
            - plp.lpSum(self.x_vars[row_idx, col_idx]
                        for col_idx in self.set_j)
            for row_idx in self.set_i)

        self.prob.sense = plp.LpMinimize
        self.prob.setObjective(objective)

    def _combined_objective(self):
        """Combine both objectives (minimize items to expire and surplus).

        If Provider Lot has items to expire and, its BU surplus, we determine what is more
        representative: surplus or items to expire and try to minimize that. The other possibility
        would be that the BU has shortage but items to expire and so we consider minimizing items
        to expire in our objective function.

        .. math::
            min \\bigg [ \\, \\text{expire_weight} \\, \\cdot \\, \\bigg (\\, \\sum_{j=1}^{row}\\,ITE(\\,j\\,)\\,-\\,(\\,\\sum_{i=1}^{col}\\,\\, (\\,x_i\\,) \\bigg )

            + \\, \\text{surplus_weight} \\cdot \\, \\bigg ( \\, \\sum_{j=1}^{col}\\,Inv. Balance(\\,j\\,)\\,+ \\,(\\,\\sum_{i=1}^{row}\\,\\,  (\\,x_i\\,) \\bigg ) \\bigg ]

        """
        objective = 0
        expire_weight = 10
        surplus_weight = 1

        for row_idx in self.set_i:
            if self.max_expire[row_idx] > 0:
                expired_items_after = plp.lpSum(self.max_expire[row_idx]
                                                - plp.lpSum(self.x_vars[row_idx, col_idx]
                                                            for col_idx in self.set_j)
                                                if plp.lpSum(self.x_vars[row_idx, col_idx]
                                                            for col_idx in self.set_j)
                                                <= self.max_expire[row_idx] else 0)
                objective += expired_items_after * expire_weight

        for col_idx in self.set_j:
            colsum = plp.lpSum(self.x_vars[row_idx, col_idx] for row_idx in self.set_i)
            transferred = plp.lpSum(self.max_receive[col_idx] * (-1) - colsum)
            if self.max_receive[col_idx] <=0:
                objective += transferred * surplus_weight

            else:
                objective += self.max_receive[col_idx]

        # We want to minimize the difference between transferred items and items to expire or surplus
        self.prob.sense = plp.LpMinimize
        self.prob.setObjective(objective)

    def _experimental_objective(self):
        """Experimental objective function.

        This function is used to test new objective functions during development phase.

        """
        objective = 0

        for row_idx in self.set_i:

            trans_val = plp.lpSum(self.z_vars[row_idx, col_idx] for col_idx in self.set_j)
            after_transfer = self.max_provide[row_idx] - trans_val

            if self.max_provide[row_idx] < 0:
                after_transfer = -1 * after_transfer

            objective += after_transfer

        for col_idx in self.set_j:
            trans_val = plp.lpSum(self.z_vars[row_idx, col_idx] for row_idx in self.set_i)
            after_transfer = self.max_receive[col_idx] + trans_val

            if self.max_receive[col_idx] < 0:
                after_transfer = -1 * after_transfer

            objective += after_transfer

        # We want to minimize the difference between transferred items and items to expire or surplus
        self.prob.sense = plp.LpMinimize
        self.prob.setObjective(objective)

    def _get_total_transfers(self):
        """Capture quantity of different transfers between BU's were made.

        Returns
        -------
        int
            Total quantity of distinct transfers between BU's. Can range from 0
            to quantity of different combinations of BU's possible.

        """
        total_transfers = 0
        for row_idx in self.set_i:
            for col_idx in self.set_j:
                if self.x_vars[row_idx, col_idx] >= 1:
                    total_transfers += 1

        return total_transfers

    def set_objective(self):
        """Used to define which objective function to call.

        The argument used by this method is passed to the class ``OptimizationModel`` as parameter.

        **Can be either:**

        * ``expire``: Optimize inventory for reducing **only** quantity of items to expire.

        * ``surplus``: Optimize inventory for reducing **only** quantity of surplus items.

        * ``both``: Optimize inventory for reducing **both** quantity of surplus items and items to expire at the same time.

        * ``experimental``: Experimental objective function that tries to formulate the bi-objective function in a way that is \
        more complex but has potential to yield better results.

        Warning
        -------
        Do not use the experimental option on production environment. Experimental objective function is not 100% validated and might
        give incorrect recommendation transfers in some specific cases.

        """
        if self.opt_problem == 'expire':
            self._expire_objective()
        elif self.opt_problem == 'surplus':
            self._surplus_objective()
        elif self.opt_problem == 'experimental':
            self._experimental_objective()
        else: # both models
            self._combined_objective()

class OptimizationModel(ObjectiveFunction):
    """Main class for defining the optimization problem.

    The definition of the optimization problem can be divided into two components/activities:

    - Define problem **constraints**

    - Create the **objective function**.

    Attributes
    ----------
    EXTRA_COLUMNS : int
        Extra columns attached to the solution space. These represent information about the provider business units
        and will be used to define model constraints and objective function
    EXTRA_ROWS : int
        Extra rows attached to the solution space. These represent information about the receiver business units
        and will be used to define model constraints and objective function
    PROVIDING_COLUMNS : dict
        Dictionary with index value for each "extra" column of provider business units

    RECEIVING_COLUMNS : dict
        Dictionary with index value for each "extra" column of receiver business units
    """

    # The extra columns attached to Solution space create a blank space on top left corner of the solution matrix.
    # This blank space has the same size as the extra columns that were added to the providing BU's row-wise
    # and the extra columns added to the receiving BU's column-wise.
    # These extra quantities are stored in the constants EXTRA_COLUMNS, EXTRA_ROWS and used to
    # and added to PROVIDING_COLUMNS and RECEIVING_COLUMNS index to ignore these extra lines and columns.
    EXTRA_COLUMNS = len(PROVIDING_COLUMNS)  # Extra columns attached to the solution space.
    EXTRA_ROWS = len(RECEIVING_COLUMNS)     # Extra rows attached to the solution space.

# ==================================== Example of the Solution Matrix "Blank Space" ====================================

#    +-----------------------------------------------------------------------+-----------------+----------+--------+
#    |                                                                       |                 |          |        |
#    |                                                                       |     Inv BU      |  10000   | 27500  |
#    |                                                                       +=================+==========+========+
#    |                                                                     A |                 |          |        |
#    |                                                                     T |  Inv. Balance   | -1262.6  |   0    |
#    |                                                                     T +-----------------+----------+--------+
#    |                                                                     A |                 |          |        |
#    |                                                                     C |      Price      |  2.401   | 2.401  |
#    |                                                                     H +-----------------+----------+--------+
#    |                                                                     E |                 |          |        |
#    |                          Blank Space Generated                      D |    Avg. Cons    |  134.65  |   0    |
#    |                                                                      +-----------------+----------+--------+
#    |                                                                     R |                 |          |        |
#    |                                                                     O |        …        |    …     |   …    |
#    |                                                                     W +-----------------+----------+--------+
#    |                                                                     S |                 |          |        |
#    |                                                                       | BU Qty on Hand  |   1969   |   0    |
#    |                                                                       +-----------------+----------+--------+
#    |                                                                       |                 |          |        |
#    |                  ATTACHED COLUMNS                                     |        0        |    0     |   0    |
#    +---------+---------------+--------+-------------+----+-----------------+-----------------+----------+--------+
#    |         |               |        |             |    |                 |                 |          |        |
#    | Inv BU  | Inv. Balance  | Price  | Avg. Cons.  | …  | BU Qty on Hand  |        0        |    0     |   0    |
#    +---------+---------------+--------+-------------+----+-----------------+-----------------+----------+--------+
#    |         |               |        |             |    |                 |                 |          |        |
#    |  10000  |    -1262.6    | 2.401  |   134.65    | …  |      1969       |        0        |    0     |   0    |
#    +---------+---------------+--------+-------------+----+-----------------+-----------------+----------+--------+
#    |         |               |        |             |    |                 |                 |          |        |
#    |  27500  |       0       | 2.401  |      0      | …  |        0        |        0        |    0     |   0    |
#    +---------+---------------+--------+-------------+----+-----------------+-----------------+----------+--------+
#    |         |               |        |             |    |                 |                 |          |        |
#    |    …    |       …       |   …    |      …      | …  |        …        |        …        |    …     |   …    |
#    +---------+---------------+--------+-------------+----+-----------------+-----------------+----------+--------+


    PROVIDING_COLUMNS = {
        'bu_provide': 0,
        'max_provide': 1,
        'min_ship_prov': 2,
        'price_prov': 3,
        'avg_cons_prov': 4,
        'max_expire': 5,
        'days_expire': 6,
        'target_provide': 7,
        'inv_provide': 8,
        'lot_id': 9,
        'bu_doi_balance': 10,
        'provide_delta_doi': 11,
        'provide_bu_oh_plus_transit': 12,
        'provide_days_of_inventory': 13,
        'provide_item_descrip': 14,
        'provide_bu_region': 15,
        'provide_contact_email': 16,
        'provide_supplier_name': 17,
        'provide_std_uom': 18,
        'provide_chart_of_accounts': 19,
        'provide_bu_address': 20,
        'provide_bu_descrip': 21,
        'provide_on_site_email': 22,
        'provide_expire_date': 23,
        'provide_bu_qty_on_hand' : 24,
        'provide_bu_item_qty_in_transf': 25,
        'provide_default_shipment_days': 26,
        'provide_can_transfer_inventory': 27,
        'provide_can_receive_inventory': 28,
        'provide_item_stats':29,
        }
    """Dictionary with index value for each "extra" column of provider business units"""

    RECEIVING_COLUMNS = {
        'bu_receive': 0,
        'max_receive': 1,
        'min_ship': 2,
        'price': 3,
        'avg_cons': 4,
        'target_receive': 5,
        'inv_receive': 6,
        'receive_delta_doi': 7,
        'receive_bu_oh_plus_transit': 8,
        'receive_days_of_inventory': 9,
        'receive_bu_region': 11,
        'receive_contact_email': 12,
        'receive_std_uom': 14,
        'receive_chart_of_accounts': 15,
        'receive_bu_address': 16,
        'receive_bu_descrip': 17,
        'receive_on_site_email': 18,
        'receive_items_to_expire': 19,
        'receive_bu_item_qty_in_transf': 20,
        'receive_can_transfer_inventory': 22,
        'receive_can_receive_inventory': 23,
        'receive_item_stats':24,
        }
    """Dictionary with index value for each "extra" column of receiver business units"""

    def __init__(self, smatrix: np.ndarray, item_id: object, opt_problem: str, solver_time: int):
        """
        Arguments used at the optimization model.

        Parameters
        ----------
        smatrix : np.ndarray
            Matrix of size (n x m) with data containing the amounts
            of items from a given SKU that can be sent and received from one BU to
            another. Size n represents all lots from a given item, available on all BUs and m the number o receiving BUs.
        item_id : int
            ID of the item that we're trying to optimize.
        opt_problem: str
            Type of objective to be added to the model. ``expire``, ``surplus``, ``both``.
        solver_time : int
            Maximum time the solver has to find the optimal solution to a single item ID.

        Note
        ----
        The method :meth:`_check_optimization_objective()<optimization.model.optimizer.OptimizationModel._check_optimization_objective>` \
            will override the ``opt_problem`` parameter that was passed if it identifies that there is no need to optimize both ``surplus`` and ``items to expire``.

        """
        self.smatrix_df = pd.DataFrame(smatrix)
        self.item_id = item_id
        self.solver_time = solver_time
        self.opt_problem = opt_problem
        self._specify_extra_constants()

        self._check_optimization_objective()
        self.prob = plp.LpProblem("Inventory_Optimization")
        super(OptimizationModel, self).__init__()

        self._create_decision_variables()
        self._objective_function
        self._create_main_constraints()

    def _check_optimization_objective(self):
        """
        Used to verify if problem needs to be bi-objective or not.

        If we have NONLOT items or if the Lot items have no quantity to expire,
        we don't need to use the multi-objective function. We just need to
        minimize surplus and shortage.

        On the other hand, if we items to expire and in more quantity than there is
        BUs with shortage, then we don't need to optimize surplus and shortage. All
        shortage will be already most likely consumed by the expiring items and some BUs
        will end up with a surplus in order to minimize those expiring items.

        Finally, if we have items to expire but in less quantity than shortage, then
        we need the bi-objective function to try to arrange the transfers the best way possible.

        **In other words this function will:**

        1. Analyze input data and determine if new objective needs to be set.

        2. Set new objective function and override the one that was passed if it identifies that the objective needs to be changed.

        """
        max_expire = sum(self.max_expire[row_idx] for row_idx in self.set_i)
        max_shortage = sum(
            -1 * self.max_receive[col_idx] if self.max_receive[col_idx] <= 0 else 0
            for col_idx in self.set_j
        )

        if (self.lot_id[0] == 'NONLOT') or (max_expire<=0):
            self.opt_problem = "surplus"
        elif max_shortage <= max_expire:
            self.opt_problem = "expire"

    # ================== Decision variables ==================
    def _specify_extra_constants(self):
        """Extra constants used to define model constraints and objective function.

        """
        # ================== Range X,Y ==================
        # Size of the solution space matrix
        # Used to create the variables of the model.
        self.set_i = range(len(self.smatrix_df.index) - self.EXTRA_ROWS)
        self.set_j = range(len(self.smatrix_df.columns) - self.EXTRA_COLUMNS)
        # ================== Receiving/Providing BU Constant ==================
        self._create_constants()

        # ================== Min Shipment Flag ==================
        # Big M gives us a way of adding logical constraints to the model.
        # In our case here, we use it to tell the model not to transfer items
        # from one BU to another, if their total value doesn't exceed the
        # minimum shipment value.
        self.BIG_M = max(sum(self.max_receive), sum(self.max_provide), sum(self.inv_provide)) * 11

    def _create_constants(self):
        """Generate model constants.

        These constants are used by the model output or to create the boundaries for the model constraints.

        """
        const_dict = {
            'providing': self.PROVIDING_COLUMNS,
            'receiving': self.RECEIVING_COLUMNS}

        for key, value in const_dict.items():
            self._specify_constants(value, key)

    def _specify_constants(self, mapping_list: list, what: str='receiving'):
        """Specify the mapping of constants with their respective names and indexes.

        Parameters
        ----------
        mapping_list : list
            List os constants and their index in the ``solution matrix``.
        what : str, optional
            If constants being defined are for ``receivers`` or ``providers``., by default 'receiving'

        """
        for key, value in mapping_list.items():
            try:
                exec("self.{} = {}".format(key, self._specify_how(what, value)))
            except NameError:
                pass

    def _specify_how(self, what: str, value: int):
        """
        Used to get lists with information from receiving and providing BUs.

        Parameters
        ----------
        what : str
            Flag used to specify if we're trying to obtain information from ``Providers`` or ``Receivers`` business units.
        value : int
            Index inside of the solution matrix where that given information is located.

        Returns
        -------
        attr: list
            Attribute of receiving or providing BU.

        Note
        ----
        The indexes for ``Receivers`` and ``Providers`` is located inside this module and are named ``RECEIVING_COLUMNS`` and ``PROVIDING_COLUMNS``. \
        If you want to add additional columns to be passed on to the model or to the output table, you first need to add them to \
        :meth:`optimization.datatools.pipelines.data_pipeline()<optimization.datatools.pipelines.data_pipeline>` method, then at \
        :meth:`optimization.solspace.SolutionSpace<optimization.solspace.SolutionSpace>` ``Class`` you need to add the new column to ``PROVIDING_AGG`` \
        and (or) to ``RECEIVING_AGG``, specifying an aggregation type (``max``, ``first``, ``sum``...). After adding the new column there, add the new column \
        to the end of ``PROVIDING_COLUMNS`` and (or) ``RECEIVING_COLUMNS``. Finally, if you added the new column to both ``PROVIDING_COLUMNS`` and ``RECEIVING_COLUMNS``, \
        you need to add new names to differentiate the column at receiving part and providing part. You can perform this last step specifying the new names at \
        ``COLUMNS_RENAME`` inside this module.

        """
        if what == 'providing':
            return list(self.smatrix_df.iloc[self.EXTRA_ROWS:, value])

        return list(self.smatrix_df.iloc[value, self.EXTRA_COLUMNS:])

    def _create_decision_variables(self):
        """Define optimizable variables (it is a matrix of size n x n).
        """
        # ================== Transfer Recommendations ==================
        # X variables are of type integer
        # These will be the transfer recommendations for item of lot A (some lot) between BU i and j

        self.x_vars = {(i, j): plp.LpVariable(
            cat='Integer',
            lowBound=0,
            upBound=max(self.max_provide[i],
                        self.max_expire[i]),
            name="x_{}_{}".format(i, j)) for i in self.set_i for j in self.set_j}
        """plp.LpVariable models an LP (linear programming) variable with the specified associated parameters
        
        - name (Required): Name of the variable (each variable requires unique name)
        
        - lowBound: The lower bound on this variable’s range. Default is negative infinity
        
        - upBound: The upper bound on this variable’s range. Default is positive infinity
        
        - cat: The category this variable is in, Integer, Binary or Continuous(default)
        
        - e: Used for column based modelling. Relates to the variable’s existence
            in the objective function and constraints
        
        See Also
        --------
        `https://www.coin-or.org/PuLP/pulp.html#pulp.LpVariable <https://www.coin-or.org/PuLP/pulp.html#pulp.LpVariable>`_
        """

        # Used for filtering out x_vars if their total value don't exceed the minimum threshold of the shipment value.
        self.y_vars = {(i, j): plp.LpVariable(cat='Binary',name="y_{}_{}".format(i, j)) for i in self.set_i for j in self.set_j}

    # ================== Constraints ==================
    def _create_elastic_constraints(self, col_idx: int):
        """Generates model elastic constraints.

        These constraints, are soft constraints and when the model reaches their boundary
        the objective function gets penalized. Right now, we apply these constraint to penalize
        when the model transfers more items than the receivers maximum shortage.

        Parameters
        ----------
        col_idx : int
            Column we're applying the elastic constraint.

        """
        vars_rowsum = plp.lpSum([self.x_vars[row_idx, col_idx] for row_idx in self.set_i])
        shortage = sum(-1 * self.max_receive[col] for col in self.set_j if self.max_receive[col] < 0)
        items_to_expire = sum(self.max_expire)

        if items_to_expire == 0:
            penalty = 100000
            lbound = 0.0
            rbound = 0.0
        else:
            penalty = max(items_to_expire - shortage, shortage - items_to_expire)
            lbound = 0.05
            rbound = 0.05

        c6 = plp.LpConstraint(e=vars_rowsum, sense=-1,
                            name='elastics_constr_receiver_' + str(self.bu_receive[col_idx]),
                            rhs=abs(self.max_receive[col_idx]))
        c6_elastic = c6.makeElasticSubProblem(penalty = penalty, proportionFreeBoundList = [lbound,rbound])
        self.prob.extend(c6_elastic)

    def _row_constraints(self):
        """Define constraints that are passed to every row of our solution space matrix.

        In general row-wise constraints are related to providing BU's.
        This function contains constraints that are applied depending on the objective

        """
        for row_idx in self.set_i:

            vars_colsum = plp.lpSum(self.x_vars[row_idx, col_idx] for col_idx in self.set_j)
            max_trans_name = "MaxTransf_{}".format(row_idx)

            if self.opt_problem == 'surplus':
                self.prob += vars_colsum <= max(self.max_provide[row_idx], 0), max_trans_name
            elif self.opt_problem == 'expire':
                self._restrict_sender_shortage(row_idx)
            else: # both objectives.
                self._restrict_sender_shortage(row_idx)

    def _column_constraints(self):
        """Define constraints that are passed to every column of our solution space matrix.

        In general column-wise constraints are related to receiving BU's.

        """
        for col_idx in self.set_j:

            # If receiving business unit has zero consumption rate, then we shouldn't transfer her items.
            self._receiver_avg_cons_constraint(col_idx)
            # self._create_elastic_constraints(col_idx)

            # ================== Consume Before Expire Constraint ==================
            self._consume_before_expire_constraint(col_idx)

            if self.opt_problem != 'expire':
                self._restrict_receiver_surplus(col_idx)

            # ================== Third Constraint ==================
            for row_idx in self.set_i:

                # If we don't have any item expiring, shortage at the providing end and surplus at receiving we shouldn't send
                # any item. That's because you're only going to increase someone's surplus to minimize others shortage.
                # In other words, this doesn't bring any benefit.
                if self.bu_provide[row_idx] == self.bu_receive[col_idx]:
                    self.prob += self.x_vars[row_idx, col_idx] == 0, "SameBU{}_{}".format(row_idx, col_idx)

                # Constraint used to restraint transfer recommendations to be bigger than min transfer value.
                self._min_shipment_value_constraint(row_idx, col_idx)

    def _restrict_receiver_surplus(self, col_idx: int):
        """Constraint to restrict receiver surplus.

        In case we don't have more **items to expire** than we have **shortage**, \
        we shouldn't transfer more items than what receiver business units with \
        shortage can accommodate. This method is used in such cases to let the optimization model \
        know it cannot transfer more items to receiver business units than what they have of shortage.

            .. math::
                max(\\text{ITE}) \\, = \\,\\sum_{i}^{row} \\, ITE_{i}

                max(\\text{Shortage}) \\, = \\,\\sum_{i}^{col} \\,  Shortage_{i}

                \\text{if} \\, max(\\text{Shortage}) \\, \\geq \\, max(\\text{ITE}^{max}) \\, \\text{:}

                \\sum_{i}^{col} \\, ( Shortage_{i} \\, + \\, \\sum_{i}^{row} \\, x(i, j) ) \\, \\leq \\, 0

        Parameters
        ----------
        col_idx : int
            Column index of receiver business unit we're applying the constraint to.

        """

        vars_rowsum = plp.lpSum(self.x_vars[row_idx, col_idx] for row_idx in self.set_i)

        if self.max_receive[col_idx] <= 0:
            self.prob += vars_rowsum <= -1 * (self.max_receive[col_idx]), "AvoidReceiverSurplus{}".format(col_idx)
        else:
            self.prob += vars_rowsum == 0, "AvoidReceiverSurplus{}".format(col_idx)

    def _restrict_sender_shortage(self, row_idx):
        """Restriction blocks sender from going from surplus to shortage and is applied when the
        objective is to minimize only sender BU's surplus (instead of items to expire and surplus)
        """

        vars_colsum = plp.lpSum(self.x_vars[row_idx, col_idx] for col_idx in self.set_j)

        if max(self.max_provide[row_idx], self.max_expire[row_idx], 0) <= 0:
            self.prob += vars_colsum == 0, "AvoidSenderShortage{}".format(row_idx)

        else:
            self.prob += vars_colsum <= max(self.max_provide[row_idx], self.max_expire[row_idx]), "AvoidSenderShortage{}".format(row_idx)

    def _min_shipment_value_constraint(self, row_idx, col_idx):
        """Constraint transfers that are smaller than the **minimum shipment $ value** specified at the ``inventory report``.

        This constraint works by using an ``auxiliary variable`` that needs to be zero when ``x_vars(row_idx, col_idx)``
        is smaller than ``min_ship_prov[col_idx]``.

        Parameters
        ----------
        row_idx : int
            Index of the row (``sender``) from where we're transfering items from.
        col_idx : int
            Index of the column (``receiver``) from where we're transfering items to.

        """
        self.prob += self.x_vars[row_idx, col_idx] <= self.BIG_M * (self.y_vars[row_idx, col_idx]), "MinShip1_{}_{}".format(row_idx, col_idx)
        self.prob += self.x_vars[row_idx, col_idx] >= 0, "MinShip2_{}_{}".format(row_idx, col_idx)
        self.prob += (self.x_vars[row_idx, col_idx] - (self.min_ship_prov[row_idx] / self.price[col_idx])
                      >= - self.BIG_M * (1 - self.y_vars[row_idx, col_idx])), "MinShip3_{}_{}".format(row_idx, col_idx)

    def _receiver_avg_cons_constraint(self, col_idx):
        """Restrict business units with avg consumption equal to zero from receiving items
        """
        vars_rowsum = plp.lpSum(self.x_vars[row_idx, col_idx] for row_idx in self.set_i)

        # If receiving business unit has zero consumption rate, then we shouldn't
        # transfer her items.
        if self.avg_cons[col_idx] == 0:
            self.prob += vars_rowsum == 0, "ZeroCons_{}".format(col_idx)

    def _consume_before_expire_constraint(self, col_idx: int):
        """Constraint to limit item transfer to BUs that can consume them before they expire.

        For transfers to other BU's, we check subtract from column ``days_to_expire`` the total amount of
        days needed to make that transfer (column ``default_shipment_days``).

        This function also subtracts items that are being transferred from the receiving BU to other Business units.
        These items represent a reduction on the amount of days needed to consume that given Lot.
        For items being **transferred out of the BU** we don't need to take those ``default_shipment_days``.

        Parameters
        ----------
        col_idx : int
            Column of our ``solution_matrix``
        tot : int
            Total sum of days to expire.

        .. admonition:: Changelog
            :class: warning

            **02/01/2021 -** Made correction to code for constraint to consider receiver business unit own inventory
            when determining if it can consume transfers before expire.

        """
        if self.avg_cons[col_idx] != 0:
            tot = 0
            ordered_idx = np.argsort(self.days_expire)
            for row_idx in ordered_idx:
            # Add this condition to avoid DivisionError caused by dividing by zero.
                if self.bu_receive[col_idx] == self.bu_provide[row_idx]:
                    row_sum = (self.inv_provide[row_idx] - plp.lpSum(self.x_vars[row_idx, col] for col in self.set_j)) * self.avg_cons[col_idx]**-1
                else:
                    row_sum = plp.lpSum(self.x_vars[row_idx, col_idx] * self.avg_cons[col_idx]**-1)

                # Otherwise just use the increase of days of inventory, caused by items being received.
                tot += row_sum
                # Check if transfers were made to that BU. If there were transfers, compare with the days to expire left
                # for the lot that we're trying to transfer.
                if (self.x_vars[row_idx, col_idx] >= 1) or (self.bu_receive[col_idx] == self.bu_provide[row_idx]):
                    # To transfer items to other BU's, consider the default shipment days.
                    self.prob += plp.lpSum(tot) + self.provide_default_shipment_days[row_idx] <= self.days_expire[row_idx], "MaxExpire_{}_{}".format(row_idx, col_idx)
        else:
            self.prob += plp.lpSum(self.x_vars[row_idx, col_idx] for row_idx in self.set_i) == 0

    def _nonlot_constraints(self):
        """Constraints used to define the optimization problem for ``NONLOT`` items

        Since ``NONLOT`` items do not have expiration dates, there is no need to define
        constraints that restrict the amount of items that can be transferred based on the
        expiration date.

        This method adds the following constraints to the optimization model

        """
        for col_idx in self.set_j:

            self._receiver_avg_cons_constraint(col_idx)
            self._restrict_receiver_surplus(col_idx)

            for row_idx in self.set_i:

                # Constraint used to restraint transfer recommendations to be bigger than min transfer value.
                self._min_shipment_value_constraint(row_idx, col_idx)

    def _restrict_inv_balance_turnover(self):
        """Restrict business units with shortage from receiving more items than its shortage

        This rule doesn't apply if maximum quantity of items to expire exceeds total shortage
        of receiving business units
        """
        for row_idx in self.set_i:
            if self.opt_problem == "expire":
                self.prob += plp.lpSum(self.x_vars[row_idx, col_idx] for col_idx in self.set_j) <= self.max_expire[row_idx]
            else:
                self.prob += plp.lpSum(self.x_vars[row_idx, col_idx] for col_idx in self.set_j) <= max(self.max_provide[row_idx], self.max_expire[row_idx], 0)

    def _create_main_constraints(self):
        """Add the constraints to model.

        Function calls methods ``_row_constraints(self)`` and ``_column_constraints(self)``
        that define constraints row and column-wise. Row-wise constraints usually
        limits transfer recommendations based on Receiving BU limitations and column-wise
        constraints do the same for receiving BU's/Lot.

        NOTE
        ----
        If you want to add new constraints to the model, add them here.

        """
        # For nonlot items we don't need to verify items to expire and other
        # complex constraints.
        self._row_constraints()
        self._restrict_inv_balance_turnover()
        if self.lot_id[0] == 'NONLOT':
            self._nonlot_constraints()
        else:
            # Constraints used to limit
            self._column_constraints()

    def _add_mapped_columns(self, opt_df):
        """Adding provider and receiver information to output report"""

        return (opt_df
                .assign(provider_bu = lambda opt_df: opt_df['column_i'].map(self.smatrix_df.loc[self.smatrix_df.index, 0]))
                .assign(lot_id = lambda opt_df: opt_df['column_i'].map(self.smatrix_df.loc[self.smatrix_df.index, 9]))
                .assign(receiver_bu = lambda opt_df: opt_df['column_j'].map(self.smatrix_df.loc[0, self.smatrix_df.columns]))
                .pipe(self._add_providing_columns))

    def _add_providing_columns(self, opt_df):
        """Adds new columns necessary to the output report

        This method is called after the model finishes the optimization process and \
        adds extra columns required for the output report

        Parameters
        ----------
        opt_df : [type]
            Model output right after the optimization process

        Returns
        -------
        pd.DataFrame
            Transfer recommendation with the required additional columns

        """
        for key, value in self.PROVIDING_COLUMNS.items():
            opt_df = opt_df.pipe(self._assign_column, key, value, True)

        for key, value in self.RECEIVING_COLUMNS.items():
            opt_df = opt_df.pipe(self._assign_column, key, value, False)

        opt_df['transfer_value'] = opt_df['solution_value'] * opt_df['price_prov']
        opt_df['weekly_avg_cons'] = opt_df['avg_cons'] * 7
        opt_df['weekly_avg_cons_prov'] = opt_df['avg_cons_prov'] * 7
        opt_df['Sender BU Approves transfer? (YES/NO)'] = "-"
        opt_df['Receiver BU Approves transfer? (YES/NO)'] = "-"
        opt_df['Transfer Recommendation Reason'] = ""
        opt_df['PeopleSoft Transfer ID'] = ""
        opt_df['Runtime'] = self.totalTime

        return opt_df

    def _assign_column(self, opt_df, col_name, idx, providing: bool= True):
        """Add necessary columns from sender and receivers to output report
        """

        if providing:
            opt_df[col_name]=opt_df['column_i'].map(self.smatrix_df.loc[self.smatrix_df.index, idx])
        else:
            opt_df[col_name]=opt_df['column_j'].map(self.smatrix_df.loc[idx, self.smatrix_df.columns])

        return opt_df

    def _create_opt_df(self):
        """Creates optimization transfers table.

        The function used to transform optimization problem results in table format. \
        Furthermore, add additional columns are added and \
        renamed accordingly to their final name. \

        Returns
        -------
        Table with transfer recommendations for a given Item ID : pd.DataFrame

        """
        var_obj = 'variable_object'
        col_i = 'column_i'
        col_j = 'column_j'

        # Converting the result to a more readable format
        opt_df = pd.DataFrame.from_dict(self.x_vars, orient="index", columns=[var_obj])

        optimization = (opt_df
                        .set_index(pd.MultiIndex.from_tuples(opt_df.index, names=[col_i, col_j]))
                        .reset_index()
                        .assign(solution_value = lambda opt_df: opt_df[var_obj].apply(lambda item: item.varValue))
                        .assign(item_id = self.item_id)
                        .assign(column_j = lambda opt_df: opt_df[col_j] + self.EXTRA_COLUMNS)
                        .assign(column_i = lambda opt_df: opt_df[col_i] + self.EXTRA_ROWS)
                        .pipe(self._add_mapped_columns)
                        .query("solution_value > 0")
                        .rename(columns=COLUMNS_RENAME))

        return optimization[COLUMNS_TO_EXPORT]

    def save_model(self, savedir=None, dtype: str="mps"):
        """Method for saving optimization model results.

        Function used to save model results. To activate it, please change
        option ``save_model=False`` to ``save_model=True`` at ``constants.py``
        or **write on alteryx**

        .. code-block:: python

            >>> from optimization import constants
            >>> constants.save_model = True
            >>> # Continue normal code procedures
            ...

        Parameters
        ----------
        savedir : str, optional
            File directory to save model. If None is passed, results will be saved automatically on the same
            folder as log file, by default None
        dtype : str, optional
            Format user wants to save model, by default "mps".

        Raises
        ------
        AssertionError
            Format that was passed to function is not supported.
        FileNotFoundError
            Directory user wants to save model not found.

        Warn
        ----
        If you decide to save model at normal runtime, one individual file will be generated to every item id

        """
        possible_types = ["mps", "json"]
        try:
            assert dtype in possible_types

            if not savedir:
                save_where = os.path.dirname(os.path.abspath(solspace.__file__))
                savedir = os.path.join(save_where, 'logs', 'optimize_results_ITEM_ID_{}.{}'.format(self.item_id, dtype))
            else:
                if get_file_extension(savedir) not in possible_types:
                    savedir = savedir + '.' + dtype
            if dtype == "mps":
                self.prob.writeMPS(savedir)
            else:
                self.prob.to_json(savedir)
            if constants.LOG_MODE:
                    logging.warn("Model results successfully saved at: {} for Item ID: {}.".format(savedir, self.item_id))

        except AssertionError:
            raise AssertionError(
                "Please, select either json or mps as file format. {} not supported".format(dtype))
        except FileNotFoundError:
            raise FileNotFoundError(
                "No such file or directory: {}".format(savedir))

    # ================== Solving Optimization Problem ==================
    def solve(self) -> Union[pd.DataFrame, None]:
        """Solve optimization problem.

        Main method of our Model class. It calls all the other methods
        of the class, creates the model, defines its constraints, adds the
        objective function and optimizes it.

        Returns
        -------
        opt_df : pd.DataFrame
            Dataframe with the results of the optimization model.


        Note
        ----
        The model only returns results if they have an **optimal** solution status.

        """
        start_time = time.time()

        try:
            # Solve optimization problem
            self.prob.solve(plp.PULP_CBC_CMD(timeLimit=self.solver_time, msg=0))

            # Get time model took to find solution
            self.totalTime = round((time.time() - start_time),2)

            # Check if optimal solution was found
            if plp.LpStatus[self.prob.status] != "Optimal":
                # Possible status:
                # - LpStatusOptimal     “Optimal”      1
                # - LpStatusNotSolved   "Not Solved”   0
                # - LpStatusInfeasible  “Infeasible”  -1
                # - LpStatusUnbounded	“Unbounded”	  -2
                # - LpStatusUndefined   “Undefined”   -3
                if constants.LOG_MODE:
                    total_value = 0
                    logging.warning(
                        "Status: {} for Item ID: {}. Total Item Value: ${:20,.2f}. Took: {}".format(
                            plp.LpStatus[self.prob.status], self.item_id, total_value, self.totalTime))
            else:
                return self._create_opt_df()  # Optimization results formatting

        except PulpSolverError:  # Error raise when optimization model can't be solved for some unmapped reason
            logging.critical("Item ID: {} could not be solved. Unknown error was found.".format(self.item_id))
