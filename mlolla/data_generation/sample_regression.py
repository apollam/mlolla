import pandas as pd
import sklearn.datasets

new_file_name = 'sample_reg_data.csv'

X = pd.DataFrame(sklearn.datasets.load_boston()['data'],
                 columns=sklearn.datasets.load_boston()['feature_names'])
y = pd.DataFrame(sklearn.datasets.load_boston()['target'], columns=['y'])
df = pd.merge(y, X, left_index=True, right_index=True)

path = 'input/data/'
df.to_csv(path + new_file_name, index=True)
print('Saved on ' + path)

