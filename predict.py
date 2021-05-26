import json
import sys
from mlolla.model.model_output import ModelOutput
import pickle as pkl
from mlolla.utils.config_json_treatment import get_x_y


if __name__ == "__main__":
    # Load config.json file
    config_json = json.load(open('input/config/config.json'))
    # Load pickle file (fitted model)
    model = pkl.load(open('output/model', 'rb'))
    # Get data for predictions to be made upon
    predict_data, _ = get_x_y(config_json, type='predict')
    # ModelOutput predictions
    ModelOutput(model).save_predictions(predict_data)
    # Exit
    sys.exit(0)
