import numpy as np
import pandas as pd
import seaborn as sns


def plot_facetgrid(dataset, store=True, show=False):
    df = pd.read_csv(f'../../data/processed/{dataset}_correlations.csv')
    df['abs_corr'] = df.r.apply(np.abs)
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
                           'word_w2v_cosine', 'word_glove_cosine', ], hue_order=['llm', 'task', 'sentence', 'word'])

    for ax in g.axes.flat:
        ax.tick_params(axis='x', which='both', rotation=90)
    g.figure.tight_layout()
    if store:
        g.figure.savefig(f'../../reports/figures/{dataset}_correlations.png', bbox_inches='tight', dpi=300)
    if show:
        g.figure.show()


if __name__ == '__main__':

    plot_facetgrid('colqitt_et_al')
    plot_facetgrid('matthews_et_al')
