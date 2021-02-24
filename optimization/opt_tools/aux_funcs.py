"""Python module used to store auxiliary functions that are used in the rest of the code.
"""
import logging
import os
import re

import pandas as pd


def get_num(string: str) -> int:
    """Return number from last results .csv version.

    Parameters
    ----------
    string : str
        String with the name of the results.csv file.

    Returns
    -------
    int : Number of last version of results.

    Examples
    --------
    >>> get_num('Results_V7.csv')
    7
    >>> get_num('Results_V12.csv')
    12
    """
    return int(re.sub(r"\D", '', string))


def last_result(fdir: str, fname: str) -> int:
    """Get last version number from all results file on directory.

    Parameters
    ----------
    fdir : str
        Directory where results are being stored.
    fname : str
        Name of the file to be searched for

    Returns
    -------
    int : Last version number from all results at the base directory.

    """
    numbers = [get_num(name) for name in os.listdir(fdir) if fname in name]
    numbers.append(0)
    return int(max(numbers))


def round_values(df: pd.DataFrame, column_name: str, decimal_points: int) -> pd.DataFrame:
    """Round a DataFrame to a variable number of decimal places.

    Method uses for...loop in order to be able to round values even when
    column has non-numeric values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe object with column to be rounded.

    column_name : str
        Name of the column with values to be rounded.

    decimal_points : int, dict, Series
        Number of decimal places to round each column to. If an int is given,
        round each column to the same number of places. Otherwise dict and Series
        round to variable numbers of places. Column names should be in the keys if
        decimals is a dict-like, or in the index if decimals is a Series. Any columns
        not included in decimals will be left as is. Elements of decimals which are
        not columns of the input will be ignored.

    Returns
    -------
    df : pd.DataFrame
        DataFrame object.

    Example
    -------
    >>> df = pd.DataFrame({"A":[10.222,20.2222,30.1111,"BBBB",50], "B":[10,20,30,40,50]})
    >>> df
             A   B
    0   10.222  10
    1  20.2222  20
    2  30.1111  30
    3     BBBB  40
    4       50  50
    >>> round_values(df, "A", 2)
           A   B
    0  10.22  10
    1  20.22  20
    2  30.11  30
    3   BBBB  40
    4     50  50
    """
    df[column_name] = df[column_name].apply(lambda row: row if not isinstance(row, (int, float)) else round(row, decimal_points))
    return df


def path_exists(fdir: str, raise_error: bool = False) -> bool:
    """Check if path exists.

    Parameters
    ----------
    fdir : str
        Directory to check for existence.
    raise_error : bool
        If set to ``True``, will raise error and stop code from running, defaults to ``False``.

    Returns
    -------
    Returns True if path exists and False if it doesn't : bool

    """
    exists = os.path.exists(fdir)

    if not exists and raise_error:
        raise ValueError("The file %s does not exists. We can't run the model if no file is found." % fdir)

    return exists


def folder_exists(fdir: str):
    """If folder does not exist, creates one.

    Parameters
    ----------
    fdir : str
        Directory to be created if does not exists.

    Example
    -------
    >>> os.listdir('.')
    ['__init__.py', '__pycache__', 'load_data.py', 'aux_funcs.py']
    >>> folder_exists('./test')
    >>> os.listdir('.')
    ['test', '__init__.py', '__pycache__', 'load_data.py', 'aux_funcs.py']
    >>> os.rmdir('./test')
    """
    if not path_exists(fdir):
        create_folder(fdir)


def create_folder(fdir: str) -> None:
    """Tries to create the directory specified.

    If new folder can't be created, function raises error.

    Parameters
    ----------
    fdir : str
        Directory of new folder to be created.

    Raises
    ------
    ValueError
        For some reason folder could not be created.

    Example
    -------
    >>> os.listdir('.')
    ['__init__.py', '__pycache__', 'load_data.py', 'aux_funcs.py']
    >>> create_folder('./test')
    >>> os.listdir('.')
    ['test', '__init__.py', '__pycache__', 'load_data.py', 'aux_funcs.py']
    >>> os.rmdir('./test')
    """
    try:
        os.makedirs(fdir)
    except AssertionError:
        raise ValueError("Creation of the directory %s failed" % fdir)
    else:
        logging.info("Successfully created the directory %s " % fdir)


if __name__ == "__main__":

    import doctest
    doctest.testmod()
