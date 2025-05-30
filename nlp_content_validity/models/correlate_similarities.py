import os

import pandas as pd
from scipy.stats import spearmanr

from nlp_content_validity.data.read_data import read_validity


def main(dataset='colqitt_et_al'):
    base_dir = f'../../data/processed/{dataset}'
    df_rel = read_validity(dataset)
    results = list()
    for fname in os.listdir(base_dir):
        model = os.path.splitext(fname)[0]
        df_sim = pd.read_csv(f'{base_dir}/{fname}', index_col=0)
        for c1 in df_rel.columns:
            for c2 in df_sim.columns:
                r, p = spearmanr(df_rel[c1], df_sim[c2])
                results.append((c1, c2, r, p, model))
    pd.DataFrame(results, columns=['validity_metric', 'nlp_metric', 'r', 'p', 'model']).to_csv(
        f'../../data/processed/{dataset}_correlations.csv')


if __name__ == '__main__':
    main('colqitt_et_al')
    main('matthews_et_al')
