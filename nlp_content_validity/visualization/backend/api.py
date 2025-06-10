import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
from sentence_transformers.util import cos_sim

from nlp_content_validity.data.read_data import read_dataset
from nlp_content_validity.features.compute_similarities import iterate_on_scale
from nlp_content_validity.features.sentence_similarity import T5_model, RoBERTa_model
from nlp_content_validity.models.aggregate_similarities import htd_aggregate

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once
MODELS = {
    "sentence-t5-base": T5_model(),
    "stsb-roberta-base": RoBERTa_model(),
}

import urllib.parse


# Load examples into a dict keyed by escaped ID
dataset = "colquitt_et_al"
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
# definitions, df, focal_scales, orbiting_dict = read_dataset(dataset)

for scale, items, scale_name, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
    if scale != scale_name:
        continue

    adv1 = definitions.get(orbiting_dict[scale]["orbiting_scale_1"], "")
    adv2 = definitions.get(orbiting_dict[scale]["orbiting_scale_2"], "")

    example_id = urllib.parse.quote_plus(scale)
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
    aggregated_score: float

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
        return PredictResponse(
            model_used=req.model_name,
            item_scores=[],
            aggregated_score=0.0
        )

    model = MODELS[req.model_name]

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
        else:
            agg = sum(sims_target) / len(sims_target) if sims_target else 0.0

    return PredictResponse(
        model_used=req.model_name,
        item_scores=sims_target,
        aggregated_score=agg
    )

@app.get("/api/examples", response_model=List[str])
async def get_example_ids():
    # print('getting examples')
    # print(list(examples_by_id.keys()))
    return list(examples_by_id.keys())

@app.get("/api/example/{example_id}", response_model=ExampleResponse)
async def get_example(example_id: str):
    if example_id not in examples_by_id:
        return JSONResponse(status_code=404, content={"error": "Example not found"})
    return examples_by_id[example_id]
