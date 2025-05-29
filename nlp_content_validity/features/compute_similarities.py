import json
import os

from sentence_transformers import util

from sentence_transformers import CrossEncoder
from llama_cpp import Llama

import logging

from tqdm import tqdm

from nlp_content_validity.data.read_data import read_dataset
from nlp_content_validity.features.get_llm_rating import PROMPT_TEMPLATE
from nlp_content_validity.features.sentence_similarity import preprocess, tokenize, \
    embed_from_words, glove_model, w2v_model, fasttext_model, T5_model, \
    RoBERTa_model
import torch

from transformers import pipeline

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


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


def sts_similarity(definitions, df, focal_scales, orbiting_dict, model):
    results = list()
    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        inputs = [(item, definition) for item in items]
        outputs = model.predict(inputs)
        results.append({scale_focal: {scale:
                                          outputs.tolist()}})
    return results


def nli_similarity(definitions, df, focal_scales, orbiting_dict, model, relation='entailment'):
    results = list()
    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        inputs = [dict(text=item, text_pair=definition) for item in items]
        res = model(inputs)
        results.append({scale_focal: {scale:
                                          [{j['label']: j['score'] for j in i}[relation] for i in res]}})
    return results


def llm_similarity(definitions, df, focal_scales, orbiting_dict, model):
    results = list()
    with torch.no_grad():
        for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
            responses = list()
            for item in items:
                prompt = PROMPT_TEMPLATE.format(n_items=len(items), definition=definition, construct=scale,
                                                items=item)
                output = model(
                    prompt,
                    temperature=.1,
                    max_tokens=2,
                )
                responses.append(output['choices'][0]['text'])
            results.append(
                {scale_focal: {scale: list(map(lambda x: int(x) if x.strip().isnumeric() else 4, responses))}})

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


def main(dataset):
    os.makedirs(f'../../data/processed/{dataset}/', exist_ok=True)
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
            with open(f'../../data/processed/{dataset}/{model_name}_{"cosine" if cosine else "wmd"}.json', 'w') as f:
                json.dump(results, f)
    logger.info(f'2. SENTENCE MODELS')
    for model_name, model in (('sentence_t5', T5_model()), ('sentence_roberta', RoBERTa_model())):
        logger.info(f'computing for model {model_name}')
        results = sentence_model_similarity(definitions, df, focal_scales, orbiting_dict, model)
        with open(f'../../data/processed/{dataset}/{model_name}.json', 'w') as f:
            json.dump(results, f)

    logger.info(f'3. TASK MODELS')
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid(),
                         device='cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'task_sts_cross_encoder'
    logger.info(f'computing for model {model_name}')
    results = sts_similarity(definitions, df, focal_scales, orbiting_dict, model)
    with open(f'../../data/processed/{dataset}/{model_name}.json', 'w') as f:
        json.dump(results, f)

    model = pipeline("text-classification", model="tasksource/deberta-base-long-nli", top_k=None)
    model_name = 'task_nli_deberta'
    logger.info(f'computing for model {model_name}')
    for relation in ['entailment', 'neutral', 'contradiction']:
        results = nli_similarity(definitions, df, focal_scales, orbiting_dict, model, relation)
        with open(f'../../data/processed/{dataset}/{model_name}_{relation}.json', 'w') as f:
            json.dump(results, f)

    logger.info(f'4. LLM MODELS')
    model = Llama(model_path="../../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", chat_format="llama-2",
                  n_gpu_layers=-1,
                  n_ctx=32768,
                  verbose=False
                  )
    model_name = 'llm_mistral'
    logger.info(f'computing for model {model_name}')
    results = llm_similarity(definitions, df, focal_scales, orbiting_dict, model)
    with open(f'../../data/processed/{dataset}/{model_name}.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main('colqitt_et_al')
    main('matthews_et_al')
