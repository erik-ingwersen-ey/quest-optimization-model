.. Quest Optimization Model documentation master file, created by
   sphinx-quickstart on Mon Jan  4 11:19:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/images/quest_logo.png
   :scale: 25 %
   :align: center
   :class: intro-logo

Quest Diagnostics Optimization Model Technical Documentation
============================================================

Documentation of the optimization algorithm created to optimize Quest Diagnostics business units inventories.

Changelog
---------

[1.1.3] - 2021-02-19
^^^^^^^^^^^^^^^^^^^^
Added
"""""
- New logic for determining if lot is currently in use, inventory or not found at receiving business units. The new code can be found at :meth:`optimization.datatools.extra_output.FindLots<optimization.datatools.extra_output.FindLots>`.
- Created Unit tests for validating new lot usage status. Code can be found at :meth:`optimization.tests.unittest_test_extra_output<optimization.tests.unittest_test_extra_outputs>`.

Changed
"""""""
- Clarified docstrings on all source code ``modules`` and ``submodules``.

How the Model Works
-------------------

The optimization model comprises of multiple scripts that are combined to generate the transfer recommendation between BU's report.

In summary, the process starts by **ingesting the necessary input files** this process is done at the :meth:`optimization.datatools.pipelines <optimization.datatools.pipelines>`.
There, we make sure that all the required fields are in place and also **validate critical columns**. Afterwards, if no fields present critical errors, the module proceeds to
**calculate all necessary columns required to formulate the optimization problem**. Then the output from the data ingestion and manipulation is passed on to
:meth:`optimization.solspace <optimization.solspace>` that generates a **solution matrix for every unique item ID** in the inventory reports. Finally the output matrix is fed
to the :meth:`optimization.model.optimizer <optimization.model.optimizer>`. Finally, the **problem is formulated**, and based on the **defined objective function**; the model generates
the **report with transfer recommendations for every item ID**.

What is This Documentation for?
-------------------------------
This documentation is best used as reference guide. Here we specify how the optimization model ``modules``, ``classes``, and ``functions`` work to generate the
transfer recommendations. It should not be used though, as a step-by-step guide on how an optimization model works.

Inside this documentation you will also find some configuration options that can be used for changing certain aspects of the model.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   source/optimization
   source/guides


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
