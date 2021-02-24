User Guides
===========

Model Installation
------------

To install the :meth:`optimization <optimization>` package locally, first clone the source code available at the project **gitlab** page.
After cloning the repository, you can choose either to run it **with or without** adding the package to your python environment ``site-packages`` folder.

.. Attention:: If you choose **not** to install the package at your python environment, you will have to reference the folder where you stored the source
    code every time you try importing the :meth:`optimization <optimization>` modules and submodules.

Installing Using Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activate Conda Environment
""""""""""""""""""""""""""

To install install the optimization package to your ``conda`` environment, first activate the conda environment by running the Anaconda app or by adding
the following command at your ``command prompt``:

.. literalinclude:: ../_static/code/cmd.bat
    :language: console
    :caption: cmd.exe
    :linenos:
    :lines: 1-1

.. Note:: Change the name ``base`` to the respective name of your conda environment. If you don't know the names of your conda environments you can find
    them by running the following command:
        .. literalinclude:: ../_static/code/cmd.bat
            :language: console
            :caption: cmd.exe
            :linenos:
            :lines: 3-9

Go to Optimization Source Code Folder
"""""""""""""""""""""""""""""""""""""

After activating your conda environment, go to the folder where you placed the source code:

.. literalinclude:: ../_static/code/cmd.bat
    :language: console
    :caption: cmd.exe
    :linenos:
    :lines: 10-10

There you should find a file named ``setup.py``.

.. literalinclude:: ../_static/code/cmd.bat
    :language: console
    :caption: cmd.exe
    :linenos:
    :lines: 11-13
    :emphasize-lines: 3

The file ``setup.py`` is used to store all necessary package configurations that are needed to run the optimization model.

Pip Install Package
""""""""""""""""""""

After completing the previous steps ``pip install`` the package by running the following command:

.. literalinclude:: ../_static/code/cmd.bat
    :language: sh
    :caption: cmd.exe
    :linenos:
    :lines: 15-24

After installation is complete you can test the model by running the ``optimization/test_py`` script:

.. code-block:: python
    :linenos:

    ~/De/EY-Quest-Diagnostics/optimization  python test_model.py
    >>> 100%|███████████████████████████████████████| 7297/7297 [10:59<00:00, 21.90it/s]


Running Locally Without Installing the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't want to add the package to your Python environment ``site-packages`` folder, just place any testing scripts you created inside the ``/optimization`` folder.
This process is necessary since when importing modules, Python first searches the local folder where your scripts are stored before trying to find them in the ``site-packages`` folder.

.. Important:: The :meth:`optimization.model <optimization.model>` package requires other modules in order to run. If these packages are not found at your Python environment,
    the first time you import any of the optimization model modules, Python will inform that they are missing and ask if you want to download and install them.


        .. jupyter-execute::

            import optimization

        >>> "Pulp package not found. To install it, type ['Y']/N:  Y"
