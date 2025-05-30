from tqdm import tqdm

from nlp_content_validity.features.compute_similarities import main as compute_similarities


def main(dataset, nruns=10):
    for run in tqdm(range(nruns), 'stability check run'):
        compute_similarities(dataset, basedir=f'../../data/interim/stability/{run}')


if __name__ == '__main__':
    main('colqitt_et_al')
    main('matthews_et_al')
