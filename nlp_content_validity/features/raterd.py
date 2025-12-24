import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import json
import os, subprocess, signal, time
import pandas as pd
from progressbar import progressbar
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys

import numpy as np


model_path = 'dobolyilab/RATER-D'

model = AutoAWQForCausalLM.from_pretrained(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

cuda_devices = '0'
set_max_workers = 30





for dataset in ['colquitt_et_al', 'matthews_et_al']:
    to_format = list()
    definitions, df, focal_scales, orbiting_dict = read_dataset(dataset)

    for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
        for item in items:
            to_format.append(dict(definition=definition, itemtext=item))

    pd.DataFrame(to_format).to_csv(f'../../data/interim/raterd_input_{dataset}.csv', index=False)


set_temperature = 0
set_max_tokens = 128
set_seed = 1234
set_extra_body = {
    "guided_choice": [
        "1. Item does an EXTREMELY BAD job of measuring the concept provided above",
        "2. Item does a VERY BAD job of measuring the concept provided above",
        "3. Item does a SOMEWHAT BAD job of measuring the concept provided above",
        "4. Item does an ADEQUATE job of measuring the concept provided above",
        "5. Item does a SOMEWHAT GOOD job of measuring the concept provided above",
        "6. Item does a VERY GOOD job of measuring the concept provided above",
        "7. Item does an EXTREMELY GOOD job of measuring the concept provided above",
        ]}

dfs = list()
for dataset in ['matthews_et_al', 'colquitt_et_al']:

    test = pd.read_csv(f'../../data/interim/raterd_input_{dataset}.csv')[['definition', 'itemtext']]

    thePerson = '''You are an academic expert. Please follow all the instructions very carefully. The questions are unique to survey measurement development and require detailed attention.
    
    Research projects often use survey items to measure concepts. Examples in the management field include work motivation, job satisfaction, and employee stress. When writing survey items, researchers must take great care to ensure that the items do a good job of measuring the concepts of interest (e.g., that an item intended to measure work motivation really seems to capture that concept well). Your purpose is to assess survey items used in the various literatures (e.g., management).'''

    thePrompt = '''### Your job is to assess the degree to which each survey item matches the concept statement provided.
    
    You will be given a concept statement below, followed by a survey item. For each item, you will rate the degree to which it matches the provided concept statement.
    
    Not all of the survey items will match the provided concept statement. Therefore, please pay close attention to each individual survey item as you decide whether it matches the provided concept statement.
    
    ### You will judge how well a survey item matches a particular statement using this response scale:
    1. Item does an EXTREMELY BAD job of measuring the concept provided above
    2. Item does a VERY BAD job of measuring the concept provided above
    3. Item does a SOMEWHAT BAD job of measuring the concept provided above
    4. Item does an ADEQUATE job of measuring the concept provided above
    5. Item does a SOMEWHAT GOOD job of measuring the concept provided above
    6. Item does a VERY GOOD job of measuring the concept provided above
    7. Item does an EXTREMELY GOOD job of measuring the concept provided above
    
    ### The concept statement:
    {statement}
    
    ### The survey item:
    {item}
    
    ### Answer the question by immediately stating one response from the scale above verbatim. YOUR RESPONSE MUST MATCH THE SCALE EXACTLY WITHOUT ALTERATIONS, INCLUDING THE SCALE RESPONSE NUMBER.'''


    theJustification = 'LASTLY, PROVIDE A JUSTIFICATION FOR YOUR CHOICE.'
    thePromptWithJustification = thePrompt + ' ' + theJustification

    prompts = []

    for index, row in test.iterrows():
        prompt = thePrompt.format(
          statement = row['definition'],
          item = row['itemtext'])
        prompts += [prompt]



    def answer_prompt (person, prompt, temperature, max_tokens, seed, extra_body):
      messages = [
          {"role": "system", "content": person},
          {"role": "user", "content": prompt}
      ]

      # Attempt to use the tokenizer's chat template if available, otherwise use a generic instruct format.
      if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
          formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      else:
          # Generic instruct format fallback
          formatted_prompt = f"### System:\n{person}\n\n### User:\n{prompt}\n\n### Assistant:\n"

      # Tokenize the input and move to the model's device (GPU)
      inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device) # Moved inputs to GPU

      # Generate response from the model
      with torch.no_grad():
          outputs = model.generate(
              **inputs,
              max_new_tokens=max_tokens,
              temperature=temperature if temperature > 0 else 0.001, # Use a small non-zero temp if 0 for do_sample to work
              do_sample=temperature > 0,
              pad_token_id=tokenizer.eos_token_id,
              # For logprobs, `output_scores=True` would be needed, followed by processing scores.
          )

      generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

      # Returning a placeholder for probability, as direct logprob calculation is more complex.
      return [generated_text, 1.0]

    def run_inference (each_prompt):
      res = answer_prompt(
          person = thePerson,
          prompt = each_prompt,
          temperature = set_temperature,
          max_tokens = set_max_tokens,
          seed = set_seed,
          extra_body = set_extra_body
          )
      return(res)

    res = list(tqdm(map(run_inference, prompts), total = len(prompts)))

    resp = [x[0] for x in res]
    prob = [x[1] for x in res]

    try:
        pred = [int(resp[0]) for resp in resp]
    except:
        print('### NOTE: Issue with integer coding -- manual recode required! ###')
        pred = resp

    df = pd.DataFrame({
      'result': res,
      'pred': pred,
      'prob': prob
      })
    df.to_json(f'../../data/interim/{dataset}.jsonl', lines=True, orient='records')
    dfs.append(df)


for dataset in ['colquitt_et_al', 'matthews_et_al']:
    results = list()

    definitions, df, focal_scales, orbiting_dict = read_dataset(dataset)

    with open(f'../../data/interim/{dataset}.jsonl') as f:

        for scale_focal, items, scale, definition in iterate_on_scale(definitions, df, focal_scales, orbiting_dict):
            item_results=list()
            for item in items:
                output = json.loads(next(f))
                item_results.append(output['pred'])
            results.append({scale_focal: {scale:
                              item_results}})

    with open(f'../../data/interim/{dataset}/llm_contval_raterd.json', 'w') as f:
        json.dump(results, f)