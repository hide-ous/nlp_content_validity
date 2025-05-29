import pandas as pd


def read_dataset(dataset):
    df_orbiting = pd.read_csv('../../data/external/%s/relations.csv' % dataset)
    orbiting_dict = df_orbiting[['focal_scale', 'orbiting_scale_1', 'orbiting_scale_2']].set_index(
        'focal_scale').to_dict(
        orient='index')
    focal_scales = list(sorted(orbiting_dict.keys()))
    df = pd.read_csv('../../data/external/%s/items.csv' % dataset)
    df.columns = ['scale', 'code', 'item']
    df.set_index('code', inplace=True)
    construct_df = pd.read_csv('../../data/external/%s/definitions.csv' % dataset)
    construct_df.columns = ['code', 'definition']
    construct_df.set_index('code', inplace=True)
    construct_df = construct_df[['definition']]
    definitions = construct_df.definition.to_dict()
    return definitions, df, focal_scales, orbiting_dict


if __name__ == '__main__':
    print(read_dataset('colqitt_et_al'))
    print(read_dataset('matthews_et_al'))
