"""Optimization model scripts for generating transfer recommendations of SKUs between Quest Diagnostics business units.

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    solspace
    constants
    debug

"""

import os

try:
    import pulp as plp
except ImportError:
    if input("Pulp package not found. To install it, type ['Y']/'N' ") == "Y":
        os.system('pip install pulp')