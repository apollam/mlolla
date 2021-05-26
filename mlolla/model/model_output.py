import pandas as pd
import pickle as pkl


class ModelOutput(object):
    def __init__(self, fitted_obj):
        self.fitted_obj = fitted_obj

    def save_metrics(self):
        pd.DataFrame(self.fitted_obj.cv_results_).to_csv('output/metrics.csv')

    def save_predictions(self, predict_data):
        predictions = pd.DataFrame({
            'y_pred': self.fitted_obj.predict(predict_data),
            'y_proba': self.fitted_obj.predict_proba(predict_data) if
            hasattr(self.fitted_obj, 'predict_proba') else
            self.fitted_obj.predict(predict_data)
        })
        predictions.to_csv('output/predictions.csv')

    def save_pkl_file(self):
        with open('output/model', "wb") as out:
            pkl.dump(self.fitted_obj, out)
