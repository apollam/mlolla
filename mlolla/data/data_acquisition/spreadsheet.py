import pandas as pd
from pyxlsb import open_workbook


def get_csv(path, **kwargs):
    """Use pandas function to acquire a pandas DataFrame from csv file

    Parameters
    ----------
    path: str
        Path of the file

    **kwargs : optional
        Optional keyword arguments can be passed to pd.read_csv function.

    Returns
    -------
    A DataFrame with data
    """

    return pd.read_csv(path, **kwargs)


def get_xlsb(path, sheet_name=0, sheet_header=0):
    """Acquire a pandas DataFrame from a xlsb file

    Parameters
    ----------
    path : str
        Path of the file
    sheet_name : str, int, list, or None, default 0
        Strings are used for sheet names. Integers are used in zero-indexed
        sheet positions. Lists of strings/integers are used to request
        multiple sheets. Specify None to get all sheets.

    sheet_header : int, list of int, default 0
        Row (0-indexed) to use for the column labels of the parsed
        DataFrame. If a list of integers is passed those row positions will
        be combined into a ``MultiIndex``. Use None if there is no header.

    Returns
    -------
    A DataFrame with data
    """

    output_list = []

    with open_workbook(path) as wb:
        try:
            with wb.get_sheet(sheet_name) as sheet:
                for row in sheet.rows():
                    output_list.append([item.v for item in row])
        except IndexError:
            raise ValueError("'sheet_name' now found in the list of sheets: {}".format(str(wb.sheets)))

    print("Warning! First line is all None, pass a header in the parameter 'sheet_header'.") \
        if (output_list[0][0] is None) and (sheet_header == 0) else None

    return pd.DataFrame(output_list[sheet_header + 1:], columns=output_list[sheet_header])


def get_xlsx(path, **kwargs):
    """Use pandas function to acquire a pandas DataFrame from xlsx file

    Parameters
    ----------
    path : str
        Path of the file

    **kwargs : optional
        Optional keyword arguments can be passed to pd.read_excel function.

    Returns
    -------
    A DataFrame with data
    """

    return pd.read_excel(path, **kwargs)
