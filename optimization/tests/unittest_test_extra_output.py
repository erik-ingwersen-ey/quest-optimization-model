"""Unit testing extra outputs new developments
"""

import unittest

import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal

from optimization.constants import Columns
from optimization.datatools.extra_output import FindLot
from optimization.datatools.extra_output import in_use
from optimization.datatools.extra_output import in_inventory


class TestExtraOutputs(unittest.TestCase):
    """Unit test for ``datatools.extra_outputs.FindInfo class``
    Test Cases
    ----------
    - ``test_get_status``: testing method that checks if a given Lot from the transfer recommendations report already \
        exists on the receiving business unit. Since ``get_status`` method is used by ``lot_usage``, we're also \
         testing it as part of the validation for the new ``lot_usage`` code.

    - ``test_lot_usage``: testing logic that determines if Lot from transfer recommendation is \
        "in use", "in inventory" or "not found" at receiver business unit.

    """
    def setUp(self):
        test_data = {
            Columns.inv_bu: ["1000"],
            Columns.item_id: ["1"],
            Columns.lot_id: ["LOT_A"],
            Columns.bu_item_last_lot_depleted: ["Y"],
            Columns.date_lot_added_to_bu_inv: [pd.datetime(year=2021, month=2, day=19)],
            Columns.minimum_days_of_inventory_for_lot: [6],
        }
        self.test_inventory = pd.DataFrame(test_data)
        self.testing_findlot = FindLot(inventory_report=self.test_inventory, inv_bu="1000", item_id="1", lot_id="LOT_A")

    def test_get_status(self, level="lot"):
        status = self.testing_findlot.get_status(level)
        assert_frame_equal(status, self.testing_findlot.inventory_report)

    def test_lot_usage(self):
        self.assertEqual(in_use, self.testing_findlot.lot_usage())
        self.testing_findlot.inventory_report[Columns.bu_item_last_lot_depleted] = np.nan
        # lot still in use (even after removing value of bu_item_last_lot_depleted because
        # it's the only lot on the given business unit
        self.assertEqual(in_use, self.testing_findlot.lot_usage())

        # adding one more lot to BU, with bu_item_last_lot_depleted = "Y" making it the lot in use
        self.testing_findlot.inventory_report = self.testing_findlot.inventory_report.append({
            Columns.inv_bu: "1000",
            Columns.item_id: "1",
            Columns.lot_id: "LOT_B",
            Columns.bu_item_last_lot_depleted: "Y",
            Columns.date_lot_added_to_bu_inv: pd.datetime(year=2021, month=2, day=19),
            Columns.minimum_days_of_inventory_for_lot: 6, }, ignore_index=True)
        # Check if after adding new lot, lot usage for "LOT_A" changes to "in inventory"
        self.assertEqual(in_inventory, self.testing_findlot.lot_usage())

        # "LOT_B" should be "in use"
        self.testing_findlot.lot_id = "LOT_B"
        self.assertEqual(in_use, self.testing_findlot.lot_usage())


if __name__ == '__main__':
    """Unit test 02/19/2021 results:
    Ran 2 tests in 0.027s
    OK
    Process finished with exit code 0
    """
    unittest.main()
