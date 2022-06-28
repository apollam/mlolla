import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from mlolla.utils import get_random_state


def plot_roc(y_true, y_pred, figsize=(8, 8)):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1] ,'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def run_tsne(X, perplexities=None, hue_by=None, verbose=1):
    """Run multiple perplexity values and plot their tsne results.

    Parameters
    ----------
    X : DataFrame
        Input data
    perplexities : list, int
        Parameter for perplexity
    hue_by : str, optional
        Name of the column to hue the plots by
    verbose : int, default 1
        1 to show train params, 0 if not
    """

    if hue_by:
        labels = X[hue_by].values
        X = X.drop(columns=hue_by)

    perplexities = perplexities if perplexities else [1, 5, 10, 20, 30, 40, 50]
    perplexities = [perplexities] if not isinstance(perplexities, list) else perplexities

    for perplexity in perplexities:
        print('With perplexity: {}'.format(perplexity)) if verbose > 0 else None
        tsne = TSNE(n_components=2, verbose=verbose, perplexity=perplexity,
                    n_iter=1000, random_state=get_random_state())
        X_tsne = tsne.fit_transform(X)
        Xf = pd.DataFrame(X_tsne, columns=["COMP1", "COMP2"])
        if hue_by:
            Xf[hue_by] = labels

        sns.scatterplot("COMP1", "COMP2", hue=hue_by, data=Xf)
        plt.show()

    return Xf


def get_importances(X, model):
    feats = {}  # dict in the format {feature_name: feature_importance}
    for feature, importance in zip(X.columns, model.feature_importances_):
        feats[feature] = importance

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances = importances / importances.sum()

    return importances.sort_values('Gini-importance', ascending=False).reset_index()


def plot_importances(X, model, features_col='index', importances_col='Gini-importance'):
    df = get_importances(X, model)

    fig, ax = plt.subplots()

    sns.barplot(x=importances_col,
                y=features_col,
                data=df,
                order=df['index'],
                color='xkcd:azure',
                ax=ax,
                edgecolor=(0, 0, 0),
                linewidth=1)
    ax.set_xlabel('Normalized Importance')
    ax.set_ylabel('')
    ax.set_title('Feature Importances')

    plt.plot()
