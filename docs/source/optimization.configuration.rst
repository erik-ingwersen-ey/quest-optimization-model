Model Configurations
====================

Basic Configurations
--------------------

The optimization model configurations are stored at :meth:`optimization.constants <optimization.constants>`.
Bellow, you can find all available configuration options and their respective description.

.. glossary::

    LOG_MODE
        Can be set to ``True`` or ``False``. When enabled the model will store all **logs**  into ``./optimization/logs/optimizer.log``.
        This configuration is set to ``True`` by default.

    SAVE_MODEL
        Can be set to ``True`` or ``False``. If enabled will save the model results into the folder ``./optimization/Results/`` as a ``.csv`` file. \
        Since the model is set to run on **Alteryx**, this configuration is set to ``False`` by default.

    USE_DYNAMIC_TIME
        It can be set to ``True`` or ``False``. If enabled will determine the maximum amount of time the model can spend trying
        to optimize a single ``Item ID`` dynamically, depending on the **total dollar value corresponding to that item**.
        This configuration is set to ``False`` by default.

    MAX_TIME
        It can be set to any **positive** ``integer``. When ``USE_DYNAMIC_TIME`` is set to ``True``, this parameter is then used as upper boundary
        to determine the maximum amount of time the model will spend trying to optimize a single item. **By default is set to 500 and the value represents time in seconds.**

    DEFAULT_TIME
        It can be set to any **positive** ``integer``. Default maximum amount of seconda the model can spend trying to find the optimal solution for a single ``Item ID``.

    LOT_DF
        Variable used to store inventory reports comming from **Alteryx**.
        This variable is then used at :meth:`optimization.datatools.pipelines.data_pipeline <optimization.datatools.pipelines.data_pipeline>`
        in conjunction with ``NONLOT_DF`` to get the final input that will be used by the model.

    NONLOT_DF
        Same as ``LOT_DF``, but for **Non Lot** items.


Columns Used by the Model
-------------------------

The columns below are the ones being used as input by the model. Some of these columns are used on the final output of the model,
and others are necessary to calculate attributes used by the model. \
These columns are referenced by a ``list`` that is stored inside the method :meth:`optimization.datatools.pipelines.data_pipeline <optimization.datatools.pipelines.data_pipeline>`.
If the user wants to pass new columns to the model, add their respective names at the list inside :meth:`optimization.datatools.pipelines.data_pipeline <optimization.datatools.pipelines.data_pipeline>`.


.. csv-table::
    :header: "Column Name", "Used By", "Description"
    :widths: 50, 30, 80
    :delim: ;

    Inv BU; ``Model``; Unique identification number assigned to every business unit from Quest Diagnostics.
    Item ID; ``Model``; Unique identification number assigned to every SKU from Quest Diagnostics.
    Lot ID; ``Model``; Identification number assigned to every Lot SKU from Quest Diagnostics. This column is only used for Lot items. **Lot items are items that have expiration date**.
    Report Date (Query Date); ``Model``; Date that the inventory report was generated. This column is used together with the column ``Expire Date`` to calculate the ``Days to Expire`` column. This last column is then used to determine items that might get expired before their consumption.
    Expire Date; ``Model``; Expiration date of Lot Items.
    Average Item Daily Use; ``Model``; Average daily consumption rate of a given item at a given business unit. This column is used in order to calculate ``Inventory Balance`` and ``Items to Expire`` columns used that are the main columns used to define the optimization problem **objective function**.
    BU Qty on Hand; ``Model``; This column shows the total number of items from a particular ``Item ID`` all a single BU have. When combined with the column ``DOI Target`` and ``Average Item Daily Use`` we obtain the **inventory balance** that the given BU has for a given item.
    Lot Qty on Hand; ``Model``;
    DOI Target; ``Model``;
    Can Receive Inventory; ``Model``;
    Can Transfer Inventory; ``Model``;
    Min Shipment  Value; ``Model``;
    Price; ``Model``;
    Default Shipment Days; ``Model``;
    Item Stats; ``Model``;
    BUv Item Qty in Transf; ``Model``;
    Item Description; ``Output``;
    BU Region; ``Output``;
    Contact Email; ``Output``;
    Supplier Name; ``Output``;
    On Site Email; ``Output``;
    STD UOM; ``Output``;
    Chart of Accounts; ``Output``;
    BU Address; ``Output``;
    BU Descrip; ``Output``;

