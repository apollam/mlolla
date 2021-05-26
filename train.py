import json
import sys
from mlolla.model.trainer import Trainer
from mlolla.outputs.output import Output

if __name__ == "__main__":
    # Load config.json file
    config_json = json.load(open('input/config/config.json'))
    # Instantiate trainer object with parameters specified in config.json
    trainer = Trainer(config_json)
    # Train model
    fitted_obj = trainer.train()
    # Output train artifacts
    output = Output(fitted_obj)
    output.save_metrics()
    output.save_pkl_file()
    # Exit
    sys.exit(0)
