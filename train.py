import json
import sys
from mlolla.model.trainer import Trainer
from mlolla.model.model_output import ModelOutput

if __name__ == "__main__":
    # Load config.json file
    config_json = json.load(open('input/config/config.json'))
    # Instantiate trainer object with parameters specified in config.json
    trainer = Trainer(config_json)
    # Train model
    fitted_obj = trainer.train()
    # ModelOutput train artifacts
    output = ModelOutput(fitted_obj)
    output.save_metrics()
    output.save_pkl_file()
    # Exit
    sys.exit(0)
