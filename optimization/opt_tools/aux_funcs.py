"""Python module used to store auxiliary functions that are used in the rest of the code.

"""
import logging
import os

import pandas as pd


def get_num(string: str) -> int:
    """
    Return number from last results .csv version.

    Parameters
    ----------
    string : str
        String with the name of the results.csv file.

    Returns
    -------
    int
        Number of last version of results.

    Examples
    --------

    * ``Results_V7.csv``: 7 (Function captures the number)

    * ``Results_V12.csv``: 12 (Function captures the whole number!)

    """
    num = ''
    # For every letter on string
    for s in string:
        if s.isdigit():
            num += s

    return int(num) if num != '' else 0


def last_result(fdir: str, fname: str) -> int:
    """Get last version number from all results file on directory.

    Parameters
    ----------
    fdir : str
        Directory where results are being stored.

    Returns
    -------
    int
        Last version number from all results at the base directory.

    """
    numbers = [get_num(name) for name in os.listdir(fdir) if fname in name]
    numbers.append(0)
    return int(max(numbers))


def round_values(df: pd.DataFrame, column_name: str, decimal_points: int) -> pd.DataFrame:
    """
    Round a DataFrame to a variable number of decimal places.

    Method uses for...loop in order to be able to round values even when
    column has non-numeric values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe object with column to be rounded.

    column_name : str
        Name of the column with values to be rounded.

    decimal_points : int, dict, Series
        Number of decimal places to round each column to.
        If an int is given, round each column to the same number of places.
        Otherwise dict and Series round to variable numbers of places.
        Column names should be in the keys if decimals is a dict-like, or
        in the index if decimals is a Series. Any columns not included in decimals
        will be left as is. Elements of decimals which are
        not columns of the input will be ignored.

    Returns
    -------
    df : pd.DataFrame
        DataFrame object.

    """
    for idx in range(len(df)):
        if isinstance(df[column_name].iloc[idx], (int, float)):
            df.loc[idx, column_name] = round(df.loc[idx, column_name], decimal_points)

    return df


def path_exists(fdir: str, raise_error: bool = False) -> bool:
    """
    Check if path exists.

    Parameters
    ----------
    fdir : str
        Directory to check for existence.

    Returns
    -------
    True, None
        Returns True if path exists and nothing if it doesn't.

    """
    exists = os.path.exists(fdir)

    if (not exists) and (raise_error):
        raise ValueError("The file %s does not exists. We can't run the model if no file is found." % fdir)

    return exists


def folder_exists(fdir: str) -> None:
    """
    If folder does not exist, create one.

    Parameters
    ----------
    fdir : str
        Directory to check for existence.

    """
    if not path_exists(fdir):
        create_folder(fdir)


def create_folder(fdir: str) -> None:
    """
    Tries to create the directory specified.

    If new folder can't be created, function raises error.

    Parameters
    ----------
    fdir : str
        Directory of new folder to be created.

    Raises
    ------
    ValueError
        For some reason folder could not be created.

    """
    try:
        os.makedirs(fdir)
    except AssertionError:
        raise ValueError("Creation of the directory %s failed" % fdir)
    else:
        logging.info("Successfully created the directory %s " % fdir)


def save_results(result, fdir: str, fname: str='results') -> None:
    """Save model results.

    Saves the results obtained from the model at the specified directory using the
    version control convention established.

    Parameters
    ----------
    fdir : str
        Directory that the results will be saved.
    result : pandas.core.frame.DataFrame
        Dataframe with the optimization model's results.
    fname : str Defaults='results'
        Filename in which results should be saved.

    Returns
    -------
    None.

    """
    # Verifying if folder that we're using to store results actually exists.
    # If it does not, we're going to create it.
    folder_exists(fdir)
    fullname = os.path.join(fdir, fname)
    try:
        result.to_excel('{}_V{}.xlsx'.format(fullname, last_result(fdir, fname) + 1), index=False)
    except Exception:
        result.to_csv('{}_V{}.csv'.format(fullname, last_result(fdir, fname) + 1), index=False)



