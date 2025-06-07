import pandas as pd

MODEL_PATH = "../../../models/item_relevancy_htd_regressor.pt"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.models.validity_model import load_model



from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import torch

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the best model once
model = load_model(MODEL_PATH, device)

class PredictRequest(BaseModel):
    target_def: str
    adversaries: List[str]
    items: List[str]

class PredictResponse(BaseModel):
    predicted_score: float
    item_scores: List[float]
    edited_since_last_prediction: bool

class ExampleData(BaseModel):
    id: str
    target_def: str
    adversaries: List[str]
    items: List[str]
    true_score: float

df_orbiting = pd.read_csv('../../../data/interim/item e costrutti - relazione_tra_scale.csv')
orbiting_dict = df_orbiting[['focal_scale', 'orbiting_scale_1', 'orbiting_scale_2']].set_index('focal_scale').to_dict(
    orient='index')
focal_scales = list(sorted(orbiting_dict.keys()))
df = pd.read_csv('../../../data/interim/item e costrutti - item.csv')
df.columns = ['scale', 'code', 'item']
df.set_index('code', inplace=True)
item_df = df[['item']]
items = item_df.item.to_dict()

construct_df = pd.read_csv('../../../data/interim/item e costrutti - definizione_di_costrutto.csv')
construct_df.columns = ['code', 'definition']
construct_df.set_index('code', inplace=True)
construct_df = construct_df[['definition']]
definitions = construct_df.definition.to_dict()

# %%
scores = df_orbiting.set_index('focal_scale')['htd'].to_dict()
# %%
samples = list()
for scale, itms in df.groupby('scale'):
    if scale not in focal_scales: continue
    scale_focal = scale
    definition_focal = definitions[scale]
    scale_orbiting1 = orbiting_dict[scale]['orbiting_scale_1']
    definition_orbiting1 = definitions[scale_orbiting1]
    scale_orbiting2 = orbiting_dict[scale]['orbiting_scale_2']
    definition_orbiting2 = definitions[scale_orbiting2]

    items = itms.item.to_list()
    item_codes = itms.index.to_list()

    samples.append({
        'id':f'example_{scale}',
        "items": items,
        "target_def": definition_focal,
        "adversaries": [definition_orbiting1, definition_orbiting2],
        "true_score": scores[scale_focal]})

test_examples = {example['id']: ExampleData(
        id=example['id'],
        target_def=example['target_def'],
        adversaries=example['adversaries'],
        items=example['items'],
        true_score=example['true_score'],
    ) for example in samples[100:]}

# # # Placeholder for held-out examples
# test_examples: Dict[str, ExampleData] = {
#     "example1": ExampleData(
#         id="example1",
#         target_def="Definition of Construct A",
#         adversaries=["Definition of Construct B"],
#         items=["Item 1", "Item 2", "Item 3"],
#         true_score=0.78
#     ),
#     "example2": ExampleData(
#         id="example2",
#         target_def="Definition of Construct X",
#         adversaries=["Definition of Construct Y", "Definition of Construct Z"],
#         items=["Item X1", "Item X2"],
#         true_score=0.65
#     )
# }
print(test_examples.keys())


@app.get("/api/examples", response_model=List[str])
async def get_example_ids():
    # return 'asd'
    # # return ['ex1']
    print('called get_example_ids')
    return list(test_examples.keys())

@app.get("/api/examples/{example_id}", response_model=ExampleData)
async def get_example(example_id: str):
    return test_examples[example_id]

@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    model.eval()
    with torch.no_grad():
        scores = torch.stack([
            model.item_model(item, req.target_def, req.adversaries).to(device)
            for item in req.items
        ], dim=0)  # [num_items, 1]

        attn_weights = torch.softmax(model.att(scores), dim=0)  # [num_items, 1]
        weighted_sum = (scores * attn_weights).sum(dim=0)  # [1]
        predicted_score = model.regressor(weighted_sum).item()

        item_scores = [s.item() for s in scores]

    return PredictResponse(
        predicted_score=predicted_score,
        item_scores=item_scores,
        edited_since_last_prediction=True  # client sets this to False after prediction
    )
