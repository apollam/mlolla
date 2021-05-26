import matplotlib.pyplot as plt
import seaborn as sns


def countplot(df, label, title='', xlabel='', figsize=(5,5)):
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=label, data=df,
                       order=df[label].value_counts(dropna=False).index.tolist())
    plt.title(title)
    plt.xlabel(xlabel)

    total = len(df[label])

    for p in ax.patches:
        height = p.get_height()
        ax.text(x=p.get_x()+p.get_width()/2.,
                y=height + 20,
                s='{} ({:1.2f}%)'.format(height, height/total),
                ha="center")

    ax.annotate('Total = {}'.format(df[label].shape[0]), fontsize=14,
                xy=(1, 1), xytext=(0.95, 0.95),
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top')
    plt.show()
