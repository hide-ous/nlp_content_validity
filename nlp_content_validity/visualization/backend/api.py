from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import torch
from sentence_transformers.util import cos_sim
from nlp_content_validity.features.sentence_similarity import T5_model, RoBERTa_model
from nlp_content_validity.models.aggregate_similarities import htd_aggregate
import numpy as np

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

class PredictRequest(BaseModel):
    model_name: str
    target_def: str
    adversaries: List[str]
    items: List[str]

class PredictResponse(BaseModel):
    model_used: str
    item_scores: List[float]
    aggregated_score: float

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
