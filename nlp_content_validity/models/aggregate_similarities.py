from collections import defaultdict

import numpy as np
import pandas as pd
import os
import json

from nlp_content_validity.data.read_data import read_dataset


def focal_aggregate(data, func=np.mean):
    return {scale: func(vals['focal']) for scale, vals in data.items()}


def focal_rank_aggregate(data, func=np.mean):
    to_return = dict()
    for scale, vals in data.items():
        stacked = np.hstack((np.array(vals['focal']).reshape((-1, 1)),
                             np.array(vals['orbiting_scale_1']).reshape((-1, 1)),
                             np.array(vals['orbiting_scale_2']).reshape((-1, 1))))
        focal_rank = np.argsort(stacked, axis=1, )[:, 0]
        to_return[scale] = func(focal_rank)
    return to_return


def htd_aggregate(data):
    return {scale: (2 * np.sum(vals['focal']) - np.sum(vals['orbiting_scale_1']) - np.sum(vals['orbiting_scale_2'])) / (
            2 * len(vals['focal'])) for scale, vals in data.items()}

# def htd_aggregate2(data):
#     return {scale: np.mean([np.mean(vals['focal'])-np.mean(vals[orbiting]) for orbiting in ('orbiting_scale_1', 'orbiting_scale_2')])/6.
#             for scale, vals in data.items()}
def main(dataset='colqitt_et_al'):
    basedir = f'../../data/interim/{dataset}'
    out_dir = f'../../data/processed/{dataset}'
    os.makedirs(out_dir, exist_ok=True)
    definitions, df, focal_scales, orbiting_dict = read_dataset(dataset)
    relations = {(focal, scale2): label for focal, vals in orbiting_dict.items() for label, scale2 in vals.items()}
    for focal in focal_scales:
        relations[(focal, focal)] = 'focal'
    for filename in os.listdir(basedir):
        model_name = os.path.splitext(filename)[0]
        print(f'processing {basedir}/{filename}')
        with open(f'{basedir}/{filename}', 'rb') as f:
            data = json.load(f)
        data_dict = defaultdict(dict)
        for i in data:
            for k, v in i.items():
                for kk, vv in v.items():
                    data_dict[k][relations[(k, kk)]] = vv
        aggregates = dict()
        for func, func_name in (
                (np.mean, "mean_focal"), (np.min, "min_focal"), (np.max, "max_focal"), (np.median, "median_focal")):
            aggregates[func_name] = focal_aggregate(data_dict, func=func)
            if dataset == 'colqitt_et_al':
                aggregates[func_name + '_rank'] = focal_rank_aggregate(data_dict, func=func)
        if dataset == 'colqitt_et_al':
            aggregates['htd'] = htd_aggregate(data_dict)
            # aggregates['htd2'] = htd_aggregate2(data_dict)
        pd.DataFrame(aggregates).to_csv(f'{out_dir}/{model_name}.csv')


if __name__ == '__main__':
    main('colqitt_et_al')
    main('matthews_et_al')
