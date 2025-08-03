import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy as sp
import os


def plot_scatter(dataset='colquitt_et_al', store=True, show=False):
    df = pd.read_csv(f'../../data/processed/{dataset}_correlations.csv')
    df['family'] = df.model.apply(lambda x: x.split('_')[0])
    df.nlp_metric = df.nlp_metric.apply(lambda x: x if x != 'htd' else 'htd_nlp')
    df.validity_metric = df.validity_metric.apply(lambda x: x if not x.startswith('sme') else x[4:])

    target_sim = 'mean_focal'
    target_val = 'htc'

    dfs = list()
    for model in os.listdir(f'../../data/processed/{dataset}/'):
        df = pd.read_csv(f'../../data/processed/{dataset}/{model}', index_col=0).rename(columns={'htd':'htd_nlp'})
        df['model'] = os.path.splitext(model)[0]
        df['family'] = df.model.apply(lambda x: x.split('_')[0])
        df['model'] = df.model.apply(lambda x: '_'.join(x.split('_')[1:]))

        if (df.model.isin(["nli_deberta_contradiction", "nli_deberta_neutral", "ft_wmd", "glove_wmd", "w2v_wmd"]).any()):
            df[target_sim]*=-1
        dfs.append(df)
    df = pd.concat(dfs)
    val = pd.read_csv(f'../../data/external/{dataset}/relations.csv', index_col=0)
    df = pd.merge(df, val, how='left', left_index=True, right_index=True)
    df.head()


    g = sns.FacetGrid(df, col='model', col_wrap=4, sharex=False, sharey=False, hue='family',
                      col_order=['gemini',
                                'mistral',
                                't5',
                                'roberta',
                                'nli_deberta_entailment',
                                'nli_deberta_contradiction',
                                'nli_deberta_neutral',
                                'sts_cross_encoder',
                                'ft_wmd',
                                'w2v_wmd',
                                'glove_wmd',
                                'ft_cosine',
                                'w2v_cosine',
                                'glove_cosine',
                                'lsa']
                      ).set_titles('{col_name}')
    g.map_dataframe(sns.regplot, y=target_val, x=target_sim, order=1,
                    )
    def annotate(data, **kws):
        r, p = sp.stats.pearsonr(data[target_val], data[target_sim])
        ax = plt.gca()
        ax.text(.8, .05, 'r={:.2f}'.format(r, p),
                transform=ax.transAxes)

    g.map_dataframe(annotate)

    all_handles, all_labels = list(), list()
    for ax in g.axes:

        handles, labels = ax.get_legend_handles_labels()
        if labels[0] not in all_labels:
            all_labels+=labels
            all_handles+=handles


    legend_ax = g.axes[-1]

    fig = g.fig


    fig.legend(handles=all_handles, labels=all_labels, title="model family",
               loc='lower left', bbox_to_anchor=(.8, 0.1), # Places legend outside to the right-top
               frameon=False, #fontsize='large', title_fontsize='x-large'
               )
    plt.tight_layout()
    if store:
        g.figure.savefig(f'../../reports/figures/{dataset}_scatter.png', bbox_inches='tight', dpi=300)
    if show:
        plt.show()

def plot_facetgrid(dataset, store=True, show=False, item=False):
    df = pd.read_csv(f'../../data/processed/{dataset}{"_item" if item else ""}_correlations.csv')
    df['abs_corr'] = df.r.apply(np.abs)
    # df['abs_corr'] = df.r
    df['family'] = df.model.apply(lambda x: x.split('_')[0])
    df.nlp_metric = df.nlp_metric.apply(lambda x: x if x != 'htd' else 'htd_nlp')
    df.validity_metric = df.validity_metric.apply(lambda x: x if not x.startswith('sme') else x[4:])
    df.family.unique()
    df.model.unique()
    g = sns.FacetGrid(df, col='validity_metric', row='nlp_metric').set_titles('{col_name} | {row_name}')
    g.map_dataframe(sns.barplot, x='model', y='abs_corr', hue='family', palette='husl',
                    order=['llm_gemini', 'llm_mistral', 'sentence_t5', 'sentence_roberta',
                           'task_nli_deberta_entailment', 'task_nli_deberta_contradiction',
                           'task_nli_deberta_neutral', 'task_sts_cross_encoder',
                           'word_ft_wmd', 'word_w2v_wmd', 'word_glove_wmd', 'word_ft_cosine',
                           'word_w2v_cosine', 'word_glove_cosine', 'bow_lsa'], hue_order=['llm', 'task', 'sentence', 'word', 'bow'])

    for ax in g.axes.flat:
        ax.tick_params(axis='x', which='both', rotation=90)
    g.figure.tight_layout()
    if store:
        g.figure.savefig(f'../../reports/figures/{dataset}_correlations.png', bbox_inches='tight', dpi=300)
    if show:
        g.figure.show()


if __name__ == '__main__':

    plot_facetgrid('colquitt_et_al')
    plot_facetgrid('matthews_et_al')
    plot_scatter('colquitt_et_al')
