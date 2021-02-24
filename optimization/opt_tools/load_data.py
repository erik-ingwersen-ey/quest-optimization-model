"""Module with methods for loading inventory reports.

Note
----
    At this moment, this model supports .csv .xlsx file formats.
    
Warning
-------
    `.xlsx` files should have inventory report as first sheet (otherwise it won't work).
"""

##  Third Party Module Imports  #####################################

import os

import pandas as pd

##  load_data Module Constants  ###################################

# Supported file extensions for inventory dataset
supported_file_extensions = [".csv", ".xlsx"]


##  Module Methods  ######################################################

def get_file_extension(fname: str) -> str:
    """
    Return file extension.

    Parameters
    ----------
        fname (str): Filename to be analyzed
    
    Returns
    -------
        File extension.
    """
    return os.path.splitext(fname)[1]


def load_csv(fdir: str) -> pd.DataFrame:
    """Loads csv file.

    Parameters
    ----------
        fdir (str): Filepath of the csv file.

    Returns
    -------
        pd.DataFrame: Pandas dataframe of the loaded csv.
    """
    return pd.read_csv(fdir, low_memory=False)


def load_excel(fdir: str) -> pd.DataFrame:
    """Loads excel file into pandas DataFrame.
    
    This method is an extension of load_data.
    After load_data determines the file extension,
    and if the extension is of type .xlsx, load_data 
    calls this method to load the excel file.

    Parameters
    ----------
        fdir (str): Filepath of the dataframe.

    Returns
    -------
        pd.DataFrame: Dataframe loaded.
    """
    return pd.read_excel(fdir)


def check_extension(fext: str) -> None:
    """
    Check if file extension is supported.

    Parameters
    ----------
        fext (str): File extension.

    Raises
    ------
        ValueError: Raises error if extension is not supported.

    """
    try:
        assert fext in supported_file_extensions  # [".csv", ".xlsx"]

    except AssertionError:
        raise ValueError("Sorry, the input file should " + \
                         "be of types {} \n but {} was passed".format(supported_file_extensions, fext))


def load_data(fdir: str) -> pd.DataFrame:
    """
    Load dataframe.
    
    Method calls function that determines the file extension,
    then it calls the appropriate method to load the file.
    
    NOTE
    ----
    Supported file formats should be added at module constants list named
    `supported_file_extensions`
    
    **Methods**
    
        * get_file_extension(fdir)
        
        * load_csv(fdir)
        
        * load_excel(fdir)

    Parameters
    ----------
    fdir : str
        Filepath of the dataframe.

    Returns
    -------
    x_df : pd.DataFrame
        Dataframe loaded.

    """
    file_extension = get_file_extension(fdir)
    check_extension(file_extension)

    if file_extension == ".csv":
        return load_csv(fdir)

    return load_excel(fdir)
