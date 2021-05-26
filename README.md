# MLolla

My personal ML library. 

# Uses 

- Explore data using probabilistic functions, preconfigured plots and exploration functions
- Train a model with a variation of hyperparameters
- Generate predictions to be called through an API
- Evaluate fairness of your dataset given a Sensitive Attribute column


# Train and predict

To train you will need to:
1. Place the train data (or predict data) in the data folder 
2. Configure the `config.json` file with the train parameters. 
3. Run the `train.py` script on source folder. Train will output the
fitted model and the metrics into the `output` folder.
4. To make predictions using a CSV or Excel file, you can simply run `predict.py` on source folder. The 
data to be predicted must be placed in the `input/data/` folder and specified in the `config.json`
file.

## Configuration file (config.json)

Below are the possible fields that MLolla accepts in `config.json` file.
- `train_type`: Can be "train_grid_search" or "train_randomized_search"
- `train_data_name`: Name of the **train** data found in the `input/data/` folder. 
- `predict_data_name`: (OPTIONAL) Name of the **predict** data found in the 
`input/data/` folder. Only needed when making predictions. 
- `label_name`: Name of the label column in your dataset.
- `pipeline_steps`: List containing the transformers to go in your Pipeline.
- `sens_attr_name`: Name of the column which contains the sensitive attribute to make a
fairness evaluation over.
- `<transformer-name>__<parameter-name>`: If you want to specify your transformers parameters, you'll need to pass
them in like this.

Example of a config file:
```
{
  "project_name": "mlolla",
  "train_type": "train_grid_search",
  "train_data_name": "train_data.csv",
  "predict_data_name": "check_data.csv",
  "label_name": "y",
  "sens_attr_name": "sensitive_attribute",
  "pipeline_steps": "[('sklearn.preprocessing', 'MinMaxScaler')], [('lightgbm', 'LGBMRegressor')]",
  "LGBMRegressor__is_unbalance": "[True]",
  "LGBMRegressor__learning_rate": "[0.01]",
  "LGBMRegressor__max_depth": "[5]",
  "LGBMRegressor__n_estimators": "[1000]",
  "LGBMRegressor__n_jobs": "[-1]",
  "LGBMRegressor__random_state": "[420]",
  "LGBMRegressor__verbose": "[-1]"
}
```

# Using API

To use an API to send predictions receiving data form a json, you'll need to setup an `api.py` file in the source
folder. After this, just run `$ python api.py`.

