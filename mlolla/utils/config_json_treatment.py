import ast
from .train_utils import get_class_from_string, transform
from mlolla.data.acquisition.spreadsheet import *


def get_pipeline_steps(config_json):
    """Transforms pipeline steps json format to one that can be passed to a pipeline."""

    json_format_steps = [ast.literal_eval(config_json['pipeline_steps'])]

    pipeline_steps = []
    classes = {}
    for step in json_format_steps[0]:
        module_name, class_name = step[0][0], step[0][1]

        instance = get_class_from_string(module_name, class_name)

        if ('fit_resample' not in dir(instance)) and ('transform' not in dir(instance)):
            instance.transform = transform

        if class_name in classes.keys():
            classes[class_name] += 1
            class_name += '_' + str(classes[class_name])
            pipeline_steps.append((class_name, instance()))
        else:
            pipeline_steps.append((class_name, instance()))
            classes[class_name] = 1

    return pipeline_steps


def get_training_params(config_json):
    """Transforms the training params json format to one that can be passed to a pipeline."""

    training_params = {}
    pipeline_steps = get_pipeline_steps(config_json)
    for step in pipeline_steps:
        for key in config_json:
            if step[0] in key:
                training_params[key] = ast.literal_eval(config_json[key])

    return training_params


def get_x_y(config_json, type='train'):
    """Transforms input data located in input/data folder into a DataFrame and returns X and y

    Parameters
    ----------
    config_json: str
        Contents of config.json file
    type: str
        Type of the data to be acquired: if 'train' or 'predict' data
    """

    data_path = 'input/data/'
    data_name = config_json['train_data_name'] if type == 'train' else config_json['predict_data_name']

    if '.csv' in data_name or '.' not in data_name:
        data = get_csv(data_path + data_name, index_col=0)
    elif '.xlsx' in data_name:
        data = get_xlsx(data_path + data_name, index_col=0)
    elif '.xlsb' in data_name:
        data = get_xlsb(data_path + data_name)

    X = data.drop(config_json['label_name'], axis=1) if type == 'train' else data
    y = data[config_json['label_name']] if type == 'train' else None

    return X, y
