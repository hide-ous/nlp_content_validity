import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import gensim.downloader as api
import re
from scipy.stats import percentileofscore
import urllib.parse
from fastapi.staticfiles import StaticFiles

from tqdm import tqdm


def htd_aggregate(data):
    return {scale: (2 * np.sum(vals['focal']) - np.sum(vals['orbiting_scale_1']) - np.sum(vals['orbiting_scale_2'])) / (
            2 * len(vals['focal'])) for scale, vals in data.items()}

def load_sentence_transformers(model_name):
    return SentenceTransformer(f'sentence-transformers/{model_name}', cache_folder='models/')


def T5_model():
    return load_sentence_transformers('sentence-t5-base')

def preprocess(text):
    return ''.join(c.lower() if c.isalpha() else ' ' for c in text)


def tokenize(text):
    return text.split()


def fasttext_model():
    model = api.load("fasttext-wiki-news-subwords-300")
    model.fill_norms()
    return model


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


app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/ping")
async def ping():
    return {"msg": "pong"}


# Load models once
MODELS = {
    "sentence-t5-base": T5_model(),

    "fasttext-wmd": fasttext_model(),
}

dataset = "colquitt_et_al"
reference_distributions = {
    "sentence-t5-base": pd.read_csv("data/processed/%s/sentence_t5.csv" % dataset).iloc[:, -1].dropna().tolist(),

    "fasttext-wmd": pd.read_csv("data/processed/%s/word_ft_wmd.csv" % dataset).iloc[:, -1].dropna().tolist(),
}

# Load examples into a dict keyed by escaped ID
df_orbiting = pd.read_csv('data/external/%s/relations.csv' % dataset)
orbiting_dict = df_orbiting[['focal_scale', 'orbiting_scale_1', 'orbiting_scale_2']].set_index(
    'focal_scale').to_dict(
    orient='index')
focal_scales = list(sorted(orbiting_dict.keys()))
df = pd.read_csv('data/external/%s/items.csv' % dataset)
df.columns = ['scale', 'code', 'item']
df.set_index('code', inplace=True)
construct_df = pd.read_csv('data/external/%s/definitions.csv' % dataset)
construct_df.columns = ['code', 'definition']
construct_df.set_index('code', inplace=True)
construct_df = construct_df[['definition']]
definitions = construct_df.definition.to_dict()

examples_by_id = {}

for scale, items, scale_name, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
    if scale != scale_name:
        continue

    adv1 = definitions.get(orbiting_dict[scale]["orbiting_scale_1"], "")
    adv2 = definitions.get(orbiting_dict[scale]["orbiting_scale_2"], "")

    example_id = re.sub('[^a-zA-Z]+', ' ', scale)
    examples_by_id[example_id] = {
        "id": example_id,
        "target_def": definition,
        "adversaries": [adv1, adv2],
        "items": items
    }


class PredictRequest(BaseModel):
    model_name: str
    target_def: str
    adversaries: List[str]
    items: List[str]


class PredictResponse(BaseModel):
    model_used: str
    item_scores: List[float]
    item_orbiting_1_scores: List[float]
    item_orbiting_2_scores: List[float]
    aggregated_score: float
    percentile_rank: float | None = None
    reference_mean: float | None = None
    reference_median: float | None = None


class ExampleResponse(BaseModel):
    id: str
    target_def: str
    adversaries: List[str]
    items: List[str]


@app.get("/api/models", response_model=List[str])
async def list_models():
    return list(MODELS.keys())


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if req.model_name not in MODELS:
        print("model not found", req.model_name)
        return PredictResponse(
            model_used=req.model_name,
            item_scores=[],
            item_orbiting_1_scores=[],
            item_orbiting_2_scores=[],
            aggregated_score=0.0
        )

    model = MODELS[req.model_name]
    if req.model_name=="fasttext-wmd":
        print("using model ", req.model_name, " with wmd")

        # Compute WMD distances
        target_def_tokens = list(filter(lambda x: x in model.key_to_index,
                                        tokenize(preprocess(req.target_def))))

        sims_target = []
        for item in req.items:
            item_tokens = list(filter(lambda x: x in model.key_to_index,
                                      tokenize(preprocess(item))))
            # WMD returns distance, convert to similarity (negative distance)
            distance = model.wmdistance(item_tokens, target_def_tokens)
            sims_target.append(-distance)  # Negative so higher is more similar

        sims_adv1 = []
        sims_adv2 = []
        if req.adversaries and len(req.adversaries) == 2:
            adv1_tokens = list(filter(lambda x: x in model.key_to_index,
                                      tokenize(preprocess(req.adversaries[0]))))
            adv2_tokens = list(filter(lambda x: x in model.key_to_index,
                                      tokenize(preprocess(req.adversaries[1]))))

            for item in req.items:
                item_tokens = list(filter(lambda x: x in model.key_to_index,
                                          tokenize(preprocess(item))))
                dist1 = model.wmdistance(item_tokens, adv1_tokens)
                dist2 = model.wmdistance(item_tokens, adv2_tokens)
                sims_adv1.append(-dist1)
                sims_adv2.append(-dist2)

            scale_data = {
                "example": {
                    "focal": sims_target,
                    "orbiting_scale_1": sims_adv1,
                    "orbiting_scale_2": sims_adv2
                }
            }
            agg = htd_aggregate(scale_data)["example"]
            print("using model with htd aggregate")
        else:
            print(' using average')
            agg = sum(sims_target) / len(sims_target) if sims_target else 0.0
    elif req.model_name=="sentence-t5-base":
        sims_adv1 = []
        sims_adv2 = []
        with torch.no_grad():
            item_embs = model.encode(req.items)
            def_emb = model.encode(req.target_def)
            sims_target = cos_sim(item_embs, [def_emb]).view(-1).tolist()

            if req.adversaries and len(req.adversaries) == 2:
                adv_embs = model.encode(req.adversaries)
                sims_adv = cos_sim(item_embs, adv_embs).T.tolist()
                scale_data = {
                    "example": {
                        "focal": sims_target,
                        "orbiting_scale_1": sims_adv[0],
                        "orbiting_scale_2": sims_adv[1]
                    }
                }
                agg = htd_aggregate(scale_data)["example"]
                print("using model ", req.model_name, " with htd")
                sims_adv1 = sims_adv[0]
                sims_adv2 = sims_adv[1]
            else:
                print('model ', req.model_name, ' using average')
                agg = sum(sims_target) / len(sims_target) if sims_target else 0.0
    else:
        return PredictResponse(
            model_used=req.model_name,
            item_scores=[],
            item_orbiting_1_scores=[],
            item_orbiting_2_scores=[],
            aggregated_score=0.0
        )


    if req.model_name in reference_distributions:
        reference_scores = reference_distributions[req.model_name]
        percentile = percentileofscore(reference_scores, agg, kind='rank')
        reference_mean = float(np.mean(reference_scores))
        reference_median = float(np.median(reference_scores))
    else:
        percentile = None
        reference_mean = None
        reference_median = None

    return PredictResponse(
        model_used=req.model_name,
        item_scores=sims_target,
        item_orbiting_1_scores=sims_adv1,
        item_orbiting_2_scores=sims_adv2,
        aggregated_score=agg,
        percentile_rank=percentile,
        reference_mean=reference_mean,
        reference_median=reference_median
    )


@app.get("/api/examples", response_model=List[str])
async def get_example_ids():
    return list(examples_by_id.keys())


@app.get("/api/example/{example_id}", response_model=ExampleResponse)
async def get_example(example_id: str):
    example_id = urllib.parse.unquote(example_id)
    print(example_id)
    if example_id not in examples_by_id:
        return JSONResponse(status_code=404, content={"error": "Example not found"})
    return examples_by_id[example_id]


docker = True
if docker:
    app.mount("/", StaticFiles(directory="/app/frontend/dist", html=True), name="static")
else:
    app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")