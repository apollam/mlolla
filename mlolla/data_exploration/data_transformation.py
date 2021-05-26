import pandas as pd
import numpy as np
import re


def inverse_dummies(df, **kwargs):
    """Transforms a dataset containing ONLY one hot encoded columns into
    a dataset with the original categorical column with same index."""

    df = pd.DataFrame(df.dot(df.columns), index=df.index, **kwargs)

    return df[df.columns[0]].apply(lambda x: np.nan if x == '' else x).isna()


def drop_null_cols(data, threshold=1, verbose=0):
    """Drops columns if their percentage of null values passes a threshold."""

    if verbose == 1:
        print("Previous:")
        print(data.isnull().sum() / data.shape[0])

    for col in data.columns:
        if data[col].isna().sum() / data.shape[0] >= threshold:
            data = data.drop(columns=col)
    if verbose == 1:
        print("\nAfter:")
        print(data.isnull().sum() / data.shape[0])

    return data


def normalize_string(string):
    string = string.lower()
    string = string.translate({ord(c): "c" for c in "ç"})
    string = string.translate({ord(c): "a" for c in "ãáâ"})
    string = string.translate({ord(c): "e" for c in "éê"})
    string = string.translate({ord(c): "i" for c in "íî"})
    string = string.translate({ord(c): "o" for c in "óõô"})
    string = string.translate({ord(c): "u" for c in "úüû"})
    regex = re.compile('[^a-zA-Z]')
    string = regex.sub(' ', string)  # Remove alphanumeric chars
    string = re.sub(' +', ' ', string)  # Remove multiple spaces
    return string.strip()


def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print('IQR: \n{}'.format(IQR))
    df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print('\nDataset size after outlier remotion: {}'.format(df.shape))

    return df


def get_transition_matrix(df, from_col, to_col, count_col):
    """Pass a dataframe containing the columns 'from' and
    'to' and the column to count no. of transitions."""

    transitions = df.groupby([from_col, to_col],
                             as_index=False)[count_col].count()
    transitions = pd.pivot_table(transitions, values=count_col,
                                 index=from_col, columns=to_col)
    transitions = transitions.fillna(0).astype(int)

    return transitions

