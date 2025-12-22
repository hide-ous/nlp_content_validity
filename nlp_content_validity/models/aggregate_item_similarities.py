import numpy as np
import pandas as pd
import os
import json



def main(dataset='colquitt_et_al'):
    basedir = f'../../data/interim/{dataset}'
    in_dir = f'{basedir}/item_similarities'
    out_dir = f'../../data/processed/{dataset}/item_similarities/'
    os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(in_dir):
        model_name = os.path.splitext(filename)[0]
        print(f'processing {in_dir}/{filename}')
        with open(f'{in_dir}/{filename}', 'rb') as f:
            data = json.load(f)
        data_dict = dict()
        for i in data:
            for k, v in i.items():
                v= np.array(v)
                # print(np.triu_indices_from(v, 1))
                # print(v[np.triu_indices_from(v, 1)])

                # v = np.triu(v, 1)
                if dataset=='colquitt_et_al':
                    v = v[np.triu_indices_from(v, 1)]
                    data_dict[k] = dict(mean=v.mean(),
                                        median=np.median(v),
                                        std=np.std(v),
                                        min=np.min(v),
                                        max=np.max(v)
                                        )
                elif dataset=='matthews_et_al':
                    data_dict[k] = dict(actual=v[0,0]
                                        )

        pd.DataFrame(data_dict).T.to_csv(f'{out_dir}/{model_name}.csv')


if __name__ == '__main__':
    main('colquitt_et_al')
    main('matthews_et_al')
