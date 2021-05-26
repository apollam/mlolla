from __future__ import print_function
import sys
import getopt
import pandas as pd
import random
import sklearn.datasets

local_path = 'input/data/'


def make_biased_data(new_file_name, priv_proportion):
    print("WAIT ==> Making biased data")
    X = pd.DataFrame(sklearn.datasets.load_breast_cancer()['data'],
                     columns=sklearn.datasets.load_breast_cancer()['feature_names'])
    y = pd.DataFrame(sklearn.datasets.load_breast_cancer()['target'], columns=['y'])

    N = max(X.index)
    priv_idx = random.sample(range(N), int(N * priv_proportion))
    X.loc[X.index.isin(priv_idx), 'sensitive_attribute'] = 1
    X.loc[~(X.index.isin(priv_idx)), 'sensitive_attribute'] = 0

    df = pd.merge(y, X, left_index=True, right_index=True)

    print("WAIT ==> Saving data locally")
    df.head(50).to_csv(local_path + new_file_name, index=True)
    print('Saved on ' + local_path)


if __name__ == '__main__':
    argv = sys.argv[1:]

    try:
        filename = 'biased_data.csv'
        priv_proportion = 0.9

        make_biased_data(new_file_name=filename, priv_proportion=priv_proportion)

    except getopt.GetoptError:
        print("Something went wrong :(\n")
        sys.exit(666)
