import pandas as pd
import numpy as np
import warnings


def value_counts(df, col):
    """Given a df and a column name return the `.value_counts()` for that column with
    their percentage in the same row."""

    vc = df[col].value_counts()
    vc1 = df[col].value_counts(1)

    new_vc = pd.Series([], index=[], dtype=np.dtype(object))
    for i, c in vc.iteritems():
        new_vc = new_vc.append(pd.Series(str(vc[i]) + ' ({:.3f}%)'.format(vc1[i]), index=[i]))

    return new_vc


def print_info(df, id_col=None, label=None):
    """Print the most frequent information requested from a dataframe.

    Parameters
    ----------
    id_col: str, default None
        Name of the id column. If None is passed, the information passed will be based
        on the DataFrame index.
    label: str, default None
        Name of the label column. If None, information regarding label will not be passed.
    Returns
    -------
    None
    """

    if id_col is None and df.index.name is None:
        warnings.warn('If your index is not correct pass an id_col or set a column in '
                       'your df as index. Possible columns are: {}' \
                       .format(df.columns.tolist()))

    print('{} unique ids.'.format(df[id_col].nunique() if id_col \
                                      else df.index.nunique()))
    print('{} rows e {} columns.'.format(df.shape[0], df.shape[1]))

    if label:
        print('\nLabel distribution:')
        if df[label].nunique() > 10:
            print(df[label].describe())
        else:
            print(value_counts(df, label))

    return None


def check_null(data):
    """Print quantity of null values of whole dataset."""

    raw = data.isnull().sum()
    percent = data.isnull().sum() / data.shape[0]

    values = []
    for i, x in enumerate(raw.values):
        values.append('{} ({:.5f}%)'.format(x, percent[i]))

    null = pd.Series(index=raw.keys(), data=values)

    return null


def check_merge(df1, df2, return_df=False, **kwargs):
    """Function to check merge between two DataFrames and, optionally, return a DataFrame.

    Parameters
    ----------
    df1: DataFrame
        Left DataFrame
    df2: DataFrame
        Right DataFrame
    return_df: bool, default True
        If to return the merged DataFrame or not
    **kwargs
        Other arguments to be passed to the merge function

    Returns
    -------
    Optional
        Merged DataFrame

    """

    if 'how' not in kwargs:
        df = pd.merge(df1, df2, **kwargs, indicator=True, how='outer')
    else:
        df = pd.merge(df1, df2, **kwargs, indicator=True)

    print(df['_merge'].value_counts())
    if return_df:
        return pd.merge(df1, df2, **kwargs)
