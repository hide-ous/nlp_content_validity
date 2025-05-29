import json

from sentence_transformers import util

from sentence_transformers import CrossEncoder
from llama_cpp import Llama

import os
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from tqdm import tqdm

from nlp_content_validity.data.read_data import read_dataset
from nlp_content_validity.features.sentence_similarity import preprocess, tokenize, \
    embed_from_words, glove_model, w2v_model, fasttext_model, T5_model, \
    RoBERTa_model
import torch

from transformers import pipeline

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

# not used in this stub but often useful for finding various files
project_dir = '../../'

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())
dataset = 'colqitt_et_al'
logger = logging.getLogger(__name__)
logger.info('extracting item similarity')
logger.info(f'loading data')
# dataset = 'construct'
df = pd.read_csv(f'../../data/external/{dataset}/items.csv')
df.columns = ['scale', 'code', 'item']
df.set_index('code', inplace=True)
item_df = df[['item']]
items = item_df.item.to_dict()

construct_df = pd.read_csv(f'../../data/external/{dataset}/definitions.csv')
construct_df.columns = ['code', 'definition']
construct_df.set_index('code', inplace=True)
construct_df = construct_df[['definition']]
definitions = construct_df.definition.to_dict()

os.makedirs(os.path.join(project_dir, f'data/processed/{dataset}/'), exist_ok=True)

item_df['tokens'] = item_df.item.apply(preprocess).apply(tokenize)

# embed sentences via words
mean_word_embedding_cosine_similarities = dict()
word_embedding_wmd = dict()


def main():
    logger.info(f'computing sentence embedding similarity')
    sentence_embedding_cosine_similarities = dict()
    # model_name = 'lsa_model'
    # model = LSA_model()
    # item_df[f'{model_name}_sentence_embedding'] = item_df.item.apply(lambda x: model.transform([x])[0])
    # sentence_embedding_cosine_similarities[model_name] = pd.DataFrame(cosine_similarity_from_embeddings(
    #     item_df[f'{model_name}_sentence_embedding']), index=item_df.index.values, columns=item_df.index.values)
    for model_name, model in (('sentence-t5-base', T5_model()), ('stsb-roberta-base', RoBERTa_model())):
        item_df[f'{model_name}_sentence_embedding'] = item_df.item.apply(model.encode)
        construct_df[f'{model_name}_sentence_embedding'] = construct_df.definition.apply(model.encode)
        sentence_embedding_cosine_similarities[model_name] = pd.DataFrame(
            util.cos_sim(item_df[f'{model_name}_sentence_embedding'], construct_df[f'{model_name}_sentence_embedding']),
            index=item_df.index.values, columns=construct_df.index.values)
    for model_name, sentence_embedding_cosine_similarity in sentence_embedding_cosine_similarities.items():
        sentence_embedding_cosine_similarity.to_csv(
            os.path.join(project_dir,
                         f'data/processed/{dataset}/sentence_embedding_cosine_similarity_{model_name}.csv'))
    item_df.to_csv(os.path.join(project_dir, f'data/processed/{dataset}/item_df.csv'))
    item_df.to_pickle(os.path.join(project_dir, f'data/processed/{dataset}/item_df.pkl'))
    construct_df.to_csv(os.path.join(project_dir, f'data/processed/{dataset}/construct_df.csv'))
    construct_df.to_pickle(os.path.join(project_dir, f'data/processed/{dataset}/construct_df.pkl'))

    # %%
    logger.info(f'computing sts')

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid(),
                         device='cuda' if torch.cuda.is_available() else 'cpu')
    inputs = [(q1, q2) for q1 in item_df.item for q2 in construct_df.definition]
    outputs = model.predict(inputs)
    sts = [(q1, q2, p) for (q1, q2), p in zip([(q1, q2) for q1 in item_df.index for q2 in construct_df.index], outputs)]
    # sts+=[(q2, q1, p) for (q1, q2, p) in sts]

    # %%
    sts_comparisons = pd.DataFrame(sts).drop_duplicates(subset=[0, 1]).pivot(index=0, columns=1).fillna(1.)
    sts_comparisons.columns = [i[1] for i in sts_comparisons.columns]
    sts_comparisons.index = [i for i in sts_comparisons.index]

    sts_comparisons.loc[item_df.index, construct_df.index].to_csv(
        os.path.join(project_dir, f'data/processed/{dataset}/sts_comparisons.csv'))
    # %%
    item_name_pairs = list()
    item_pairs = list()

    pipe = pipeline("text-classification", model="tasksource/deberta-base-long-nli", top_k=None)
    for q1_name, q1 in item_df.item.to_dict().items():
        for q2_name, q2 in construct_df.definition.to_dict().items():
            item_pairs.append(dict(text=q1, text_pair=q2))
            item_name_pairs.append((q1_name, q2_name))
    res = pipe(item_pairs)

    df_nli = pd.DataFrame.from_records([{i['label']: i['score'] for i in j} for j in res], index=item_name_pairs)
    df_nli.to_csv(os.path.join(project_dir, f'data/processed/{dataset}/nli_deberta_base.csv'))

    deberta = pd.read_csv(f'../../data/processed/{dataset}/nli_deberta_base.csv', index_col=0).reset_index().rename(
        columns={'index': 'question_group_pair'})
    deberta['q1'] = deberta.question_group_pair.apply(lambda x: eval(x)[0])
    deberta['q2'] = deberta.question_group_pair.apply(lambda x: eval(x)[1])
    del deberta['question_group_pair']
    deberta.head()
    deberta.pivot_table(index='q1', columns='q2', values='entailment', fill_value=1).to_csv(
        os.path.join(project_dir, f'data/processed/{dataset}/nli_deberta_base_entailment.csv'))
    deberta.pivot_table(index='q1', columns='q2', values='neutral', fill_value=0).to_csv(
        os.path.join(project_dir, f'data/processed/{dataset}/nli_deberta_base_neutral.csv'))
    deberta.pivot_table(index='q1', columns='q2', values='contradiction', fill_value=0).to_csv(
        os.path.join(project_dir, f'data/processed/{dataset}/nli_deberta_base_contradiction.csv'))

    # %%
    llm = Llama(model_path="../../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", chat_format="llama-2",
                n_gpu_layers=-1,
                n_ctx=3584 if dataset == 'prompt' else 1200,
                # verbose=False
                )
    for same_response in [True, False]:
        results_mistral = list()
        item_name_pairs = list()
        for q1 in item_df.index:
            for q2 in construct_df.index:
                item_name_pairs.append((q1, q2))
                if same_response:
                    prompt = f'<s>[INST] You rated, on a scale 1 to 10, how likely is it that one person would answer similarly to the following two questions. [Q1]"{item_df.loc[q1].item}"[/Q1] [Q2]"{construct_df.loc[q2].definition}"[/Q2] The number that corresponds to that likelihood is [/INST]'
                else:
                    prompt = f'<s>[INST] You rated, on a scale 1 to 10, how likely is it that one person would give opposite answers to the following two questions. [Q1]"{item_df.loc[q1].item}"[/Q1] [Q2]"{construct_df.loc[q2].definition}"[/Q2] The number that corresponds to that likelihood is [/INST]'
                with torch.no_grad():
                    # print(prompt)
                    output = llm(
                        prompt,
                        temperature=.1,
                        max_tokens=2,
                    )
                    res = output['choices'][0]['text']
                    results_mistral.append(res)
        df_mistral = pd.DataFrame(
            zip(map(lambda x: int(x) if x.strip().isnumeric() else x, results_mistral), (i[0] for i in item_name_pairs),
                (i[1] for i in item_name_pairs)), columns=['score', 'q1', 'q2'])
        df_mistral.loc[df_mistral.score.apply(lambda x: (not isinstance(x, int)) and (
                'low' in x)), 'score'] = 0  # cases where it's a paraphrasis of "low likelihood"
        df_mistral.loc[df_mistral.score.apply(
            lambda x: not isinstance(x, int)), 'score'] = 5  # remaining cases, hard to tell: give middling answer
        df_mistral = df_mistral.pivot_table(values='score', index='q1', columns='q2').fillna(10 if same_response else 0)
        df_mistral /= 10
        df_mistral.to_csv(os.path.join(project_dir,
                                       f'data/processed/{dataset}/mistral_{"same" if same_response else "opposite"}_response_likelihood.csv'))


def word_model_similarity(definitions, df, focal_scales, orbiting_dict, model, cosine=True):
    results = list()

    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        items_tokenized = list(map(tokenize, map(preprocess, items)))
        item_token_embeddings = [model[[xx for xx in x if xx in model.key_to_index]] for x in items_tokenized]
        item_mean_word_embeddings = [embed_from_words(te) for te in item_token_embeddings]

        definition_tokenized = tokenize(preprocess(definition))
        definition_token_embeddings = model[[xx for xx in definition_tokenized if xx in model.key_to_index]]
        definition_mean_word_embeddings = embed_from_words(definition_token_embeddings)
        if cosine:
            results.append({scale_focal: {scale:
                                              util.cos_sim(item_mean_word_embeddings,
                                                           [definition_mean_word_embeddings]).ravel().tolist()}})
        else:
            distances = list()
            for item_tokens in items_tokenized:
                distances.append(model.wmdistance(list(filter(lambda x: x in model.key_to_index, item_tokens)),
                                                  list(filter(lambda x: x in model.key_to_index,
                                                              definition_tokenized))))
            results.append({scale_focal: {scale: distances}})
    return results


def sentence_model_similarity(definitions, df, focal_scales, orbiting_dict, model):
    results = list()
    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        item_embeddings = [model.encode(item) for item in items]
        definition_embeddings = model.encode(definition)
        results.append({scale_focal: {scale:
                                          util.cos_sim(item_embeddings,
                                                       [definition_embeddings]).ravel().tolist()}})
    return results


def iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
    for scale, itms in tqdm(df.groupby('scale'), total=df.scale.nunique()):
        if scale not in focal_scales: continue
        scale_focal = scale
        definition_focal = definitions[scale]
        scale_orbiting1 = orbiting_dict[scale]['orbiting_scale_1']
        definition_orbiting1 = definitions.get(scale_orbiting1, None)
        scale_orbiting2 = orbiting_dict[scale]['orbiting_scale_2']
        definition_orbiting2 = definitions.get(scale_orbiting2, None)

        items = itms.item.to_list()

        for scale, definition in ((scale_focal, definition_focal),
                                  (scale_orbiting1, definition_orbiting1),
                                  (scale_orbiting2, definition_orbiting2)):
            if not definition: continue
            yield scale_focal, items, scale, definition


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    project_dir = '../../'

    dataset = 'colqitt_et_al'

    definitions, df, focal_scales, orbiting_dict = read_dataset(dataset)
    logger.info(f'1. WORD MODELS')

    logger.info(f'loading models')
    model_ft = fasttext_model()
    model_w2v = w2v_model()
    model_glove = glove_model()

    for cosine in (True, False):
        for model_name, model in dict(word_ft=model_ft, word_w2v=model_w2v, word_glove=model_glove).items():
            logger.info(f'computing for model {model_name} with {"cosine" if cosine else "wmd"}')
            results = word_model_similarity(definitions, df, focal_scales, orbiting_dict, model, cosine)
            with open(f'../../data/interim/{dataset}/{model_name}_{"cosine" if cosine else "wmd"}.json', 'w') as f:
                json.dump(results, f)
    logger.info(f'2. SENTENCE MODELS')
    for model_name, model in (('sentence_t5', T5_model()), ('sentence_roberta', RoBERTa_model())):
        logger.info(f'computing for model {model_name}')
        results = sentence_model_similarity(definitions, df, focal_scales, orbiting_dict, model)
        with open(f'../../data/interim/{dataset}/{model_name}.json', 'w') as f:
            json.dump(results, f)
