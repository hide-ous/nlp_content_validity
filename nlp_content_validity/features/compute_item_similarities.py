import json
import os

import numpy as np
from sentence_transformers import util

from sentence_transformers import CrossEncoder
from llama_cpp import Llama

import logging

from tqdm import tqdm

from nlp_content_validity.data.read_data import read_dataset
from nlp_content_validity.features.get_llm_rating import PROMPT_TEMPLATE
from nlp_content_validity.features.sentence_similarity import preprocess, tokenize, \
    embed_from_words, glove_model, w2v_model, fasttext_model, T5_model, \
    RoBERTa_model, LSA_model
import torch

from transformers import pipeline

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def word_model_similarity(definitions, df, focal_scales, orbiting_dict, model, cosine=True):
    results = list()

    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        if scale!=scale_focal: continue
        items_tokenized = list(map(tokenize, map(preprocess, items)))
        item_token_embeddings = [model[[xx for xx in x if xx in model.key_to_index]] for x in items_tokenized]
        item_mean_word_embeddings = [embed_from_words(te) for te in item_token_embeddings]

        if cosine:
            results.append({scale_focal:
                                              util.cos_sim(item_mean_word_embeddings,item_mean_word_embeddings).tolist()})
        else:
            distances = list()
            for item_tokens in items_tokenized:
                distances.append(model.wmdistance(list(filter(lambda x: x in model.key_to_index, item_tokens)),
                                                  list(filter(lambda x: x in model.key_to_index, item_tokens))))
            results.append({scale_focal: distances})
    return results


def sentence_model_similarity(definitions, df, focal_scales, orbiting_dict, model):
    results = list()
    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        if scale != scale_focal: continue
        item_embeddings = [model.encode(item) for item in items]
        results.append({scale_focal:
                                          util.cos_sim(item_embeddings,
                                                       item_embeddings).tolist()})
    return results


def sts_similarity(definitions, df, focal_scales, orbiting_dict, model):
    results = list()
    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        if scale != scale_focal: continue
        inputs = [(item1, item2) for item1 in items for item2 in items]
        outputs = model.predict(inputs)
        results.append({scale_focal:
                                          outputs.reshape((len(items), len(items))).tolist()})
    return results


def nli_similarity(definitions, df, focal_scales, orbiting_dict, model, relation='entailment'):
    results = list()
    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        if scale != scale_focal: continue
        inputs = [dict(text=item1, text_pair=item2) for item1 in items for item2 in items]
        res = model(inputs)
        results.append({scale_focal:
                                          np.array([{j['label']: j['score'] for j in i}[relation] for i in res]).reshape(len(items), len(items)).tolist()})
    return results


def llm_similarity(definitions, df, focal_scales, orbiting_dict, model):
    results = list()
    with torch.no_grad():
        for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
            if scale != scale_focal: continue

            responses = np.zeros(shape=(len(items), len(items)))
            for item1_idx in range(len(items)):
                for item2_idx in range(item1_idx+1, len(items)):
                    item1 = items[item1_idx]
                    item2 = items[item2_idx]
                    f'<s>[INST] You rated, on a scale 1 to 10, how likely is it that one person would answer similarly to the following two questions. [Q1]"{item1}"[/Q1] [Q2]"{item2}"[/Q2] The number that corresponds to that likelihood is [/INST]'
                    prompt = PROMPT_TEMPLATE.format(item1=item1, item2=item2)
                    output = model(
                        prompt,
                        temperature=.1,
                        max_tokens=2,
                    )
                    x = output['choices'][0]['text']
                    int(x) if x.strip().isnumeric() else 5.5
                    responses[item1_idx, item2_idx] = x
                    responses[item2_idx, item1_idx] = x
                    responses[item1_idx, item1_idx] = 10.
            results.append(
                {scale_focal: responses.tolist()})

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

def lsa_similarity(definitions, df, focal_scales, orbiting_dict, model):
    results = list()

    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        if scale not in focal_scales: continue
        item_embeddings = model.transform(items)
        results.append({scale_focal:
                                          util.cos_sim(item_embeddings,
                                                       item_embeddings).tolist()})
    return results


def main(dataset, basedir='../../data/interim'):
    os.makedirs(f'{basedir}/{dataset}/item_similarities/', exist_ok=True)
    definitions, df, focal_scales, orbiting_dict = read_dataset(dataset)

    # logger.info(f'0. BAG OF WORD MODELS')
    # model_lsa = LSA_model()
    # # all_texts = list(definitions.values()) + df.item.tolist()
    # # model_lsa.fit(all_texts)
    # model_lsa.fit(df.item)
    # results = lsa_similarity(definitions, df, focal_scales, orbiting_dict, model_lsa)
    # with open(f'{basedir}/{dataset}/item_similarities/bow_lsa.json', 'w') as f:
    #     json.dump(results, f)
    #
    # logger.info(f'1. WORD MODELS')
    #
    # logger.info(f'loading models')
    # model_ft = fasttext_model()
    # model_w2v = w2v_model()
    # model_glove = glove_model()
    #
    # for cosine in (True, False):
    #     for model_name, model in dict(word_ft=model_ft, word_w2v=model_w2v, word_glove=model_glove).items():
    #         logger.info(f'computing for model {model_name} with {"cosine" if cosine else "wmd"}')
    #         results = word_model_similarity(definitions, df, focal_scales, orbiting_dict, model, cosine)
    #         with open(f'{basedir}/{dataset}/item_similarities/{model_name}_{"cosine" if cosine else "wmd"}.json', 'w') as f:
    #             json.dump(results, f)
    # logger.info(f'2. SENTENCE MODELS')
    # for model_name, model in (('sentence_t5', T5_model()), ('sentence_roberta', RoBERTa_model())):
    #     logger.info(f'computing for model {model_name}')
    #     results = sentence_model_similarity(definitions, df, focal_scales, orbiting_dict, model)
    #     with open(f'{basedir}/{dataset}/item_similarities/{model_name}.json', 'w') as f:
    #         json.dump(results, f)
    #
    # logger.info(f'3. TASK MODELS')
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid(),
                         device='cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'task_sts_cross_encoder'
    logger.info(f'computing for model {model_name}')
    results = sts_similarity(definitions, df, focal_scales, orbiting_dict, model)
    with open(f'{basedir}/{dataset}/item_similarities/{model_name}.json', 'w') as f:
        json.dump(results, f)

    # model = pipeline("text-classification", model="tasksource/deberta-base-long-nli", top_k=None)
    # model_name = 'task_nli_deberta'
    # logger.info(f'computing for model {model_name}')
    # for relation in ['entailment', 'neutral', 'contradiction']:
    #     results = nli_similarity(definitions, df, focal_scales, orbiting_dict, model, relation)
    #     with open(f'{basedir}/{dataset}/item_similarities/{model_name}_{relation}.json', 'w') as f:
    #         json.dump(results, f)

    # logger.info(f'4. LLM MODELS')
    # model = Llama(model_path="../../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", chat_format="llama-2",
    #               n_gpu_layers=-1,
    #               n_ctx=32768,
    #               verbose=False
    #               )
    # model_name = 'llm_mistral'
    # logger.info(f'computing for model {model_name}')
    # results = llm_similarity(definitions, df, focal_scales, orbiting_dict, model)
    # with open(f'{basedir}/{dataset}/item_similarities/{model_name}.json', 'w') as f:
    #     json.dump(results, f)


if __name__ == '__main__':
    main('colquitt_et_al')
    # main('matthews_et_al')
