from __future__ import print_function
import pandas as pd
import sklearn.datasets

new_file_name = 'sample_class_data.csv'

X = pd.DataFrame(sklearn.datasets.load_breast_cancer()['data'],
                 columns=sklearn.datasets.load_breast_cancer()['feature_names'])
y = pd.DataFrame(sklearn.datasets.load_breast_cancer()['target'], columns=['y'])
df = pd.merge(y, X, left_index=True, right_index=True)

path = 'input/data/'
df.to_csv(path + new_file_name, index=True)
print('Saved on ' + path)
