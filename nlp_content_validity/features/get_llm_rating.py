import json
import re
from dotenv import load_dotenv
from google.genai.errors import ServerError
from tqdm import tqdm
import os
import time
from typing import Optional
from google import genai

from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable

from nlp_content_validity.data.read_data import read_dataset

MODEL = "gemini-2.5-flash-preview-04-17"
_last_request_time = 0

PROMPT_TEMPLATE = """
Please read the instructions very carefully. The questions are unique to survey measurement development and require detailed attention. 
Research projects in the management field often use survey items to measure work concepts, such as work motivation, job satisfaction, and employee stress. When writing survey items, management researchers must take great care to ensure that the items do a good job of measuring the concepts of interest (e.g., that an item intended to measure work motivation really seems to capture that concept well). The goal of this study is to assess survey items used in the management literature. 
Your job in this survey is to assess the degree to which each item listed matches the statement provided. 
On the next few pages you will see a bolded statement, followed by several survey items. For each item, you will rate the degree to which it matches the bolded statement. The items will repeat themselves on three consecutive pages, but the bolded statements will change. Again, simply rate the degree to which each item matches the bolded statement on that page. 
Not all of the items will match the bolded statement. Therefore, please pay close attention to each individual question as you decide whether it matches the bolded statement. 
Before beginning the survey, below is an example to help guide your understanding of the survey. 
The survey asks you to judge how well a survey item matches particular statements, which will be presented to you in bold. You will make that judgment using this response scale:
1: Item does an EXTREMELY BAD job of measuring the bolded concept provided above
2: Item does an VERY BAD job of measuring the bolded concept provided above
3: Item does an SOMEWHAT BAD job of measuring the bolded concept provided above
4: Item does an ADEQUATE job of measuring the bolded concept provided above
5: Item does an SOMEWHAT GOOD job of measuring the bolded concept provided above
6: Item does an VERY GOOD job of measuring the bolded concept provided above
7: Item does an EXTREMELY GOOD job of measuring the bolded concept provided above
For example, let’s say the statement is: **Work Motivation: The effort expended in relation to work.** 
Since this statement refers to effort, an item that does a good job matching this statement might be, “I work hard in my job,” because it speaks to a certain effort level at work. An item that also does a good job matching this statement might be, “I often feel lazy at the office,” because it also speaks to a certain effort level at work. In contrast, an item that does a bad job matching this statement might be, “I work in a city,” because it has very little to do with the effort level at work. Please note that some of the items on the survey will focus on high levels of a given concept (like the “I work hard” item), whereas others will focus on low levels of a given concept (like the “I often feel lazy” item). Both can capture the concept of expending effort equally well.

Using the example above, please rate the following {n_items} items on how well each does matching our concept, **{construct}: {definition}**

Answer with only the numbers corresponding to the ratings of the following items:
{items}
"""


def query_gemini(client,
                 prompt: str,
                 min_interval: float = 60 / 10,  # 10 requests per minute
                 max_retries: int = 3,
                 model=MODEL
                 ) -> Optional[str]:
    """
    Queries Gemini with a prompt, respecting free-tier rate limits.

    Args:
        client (google.genai.Client): Google GenAI API client.
        prompt (str): The input prompt.
        min_interval (float): Minimum seconds between requests.
        max_retries (int): Max retry attempts on failure.
        model (str): The model to use.

    Returns:
        Optional[str]: The Gemini response, or None on failure.

    """
    global _last_request_time

    # Rate limit: ensure minimum interval between requests
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        print(f"{elapsed:.2f} seconds elapsed from last call, sleeping for {min_interval - elapsed:.2f} seconds")
        time.sleep(min_interval - elapsed)

    for attempt in range(max_retries):
        try:
            _last_request_time = time.time()
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response
        except (ResourceExhausted, InternalServerError, ServiceUnavailable, ServerError) as e:
            wait = 5 ** (attempt + 1)
            print(f"[Retry {attempt + 1}] Error: {e}. Waiting {wait}s...")
            time.sleep(wait)
    print("Request failed after max retries.")
    return None


def main(dataset='colqitt_et_al'):
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINIKEY"])

    definitions, df, focal_scales, orbiting_dict = read_dataset(dataset)


    responses = dict()
    for scale, itms in tqdm(df.groupby('scale'), total=df.scale.nunique()):
        print(scale)
        if scale not in focal_scales: continue
        scale_focal = scale
        definition_focal = definitions[scale]
        scale_orbiting1 = orbiting_dict[scale]['orbiting_scale_1']
        definition_orbiting1 = definitions.get(scale_orbiting1, None)
        scale_orbiting2 = orbiting_dict[scale]['orbiting_scale_2']
        definition_orbiting2 = definitions.get(scale_orbiting2, None)

        items = itms.item.to_list()
        # item_codes = itms.index.to_list()

        for scale, definition in ((scale_focal, definition_focal),
                                  (scale_orbiting1, definition_orbiting1),
                                  (scale_orbiting2, definition_orbiting2)):
            if ((scale_focal, scale) in responses) and (responses[(scale_focal, scale)] is not None): continue
            if not definition: continue

            prompt = PROMPT_TEMPLATE.format(n_items=len(items), definition=definition, construct=scale,
                                            items='\n'.join(
                                                map(lambda x: ': '.join((str(x[0]), x[1])), enumerate(items))))
            response = query_gemini(client, prompt)

            responses[(scale_focal, scale)] = response

    responses_parsed = list()
    for (s1, s2), res in responses.items():
        if res is None:
            print(s1, s2)
        elif s1 not in focal_scales:
            print(s1)
        else:
            responses_parsed.append((s1, s2, res.text))

    responses_parsed = [(x[0], x[1], [int(re.sub(r'[^0-9]', '', score)) for score in x[2].split()]) for x in
                        responses_parsed]
    responses_parsed = {(i[0], i[1]): i[2] for i in responses_parsed}

    os.makedirs(f'../../data/processed/{dataset}/', exist_ok=True)
    with open(f'../../data/processed/{dataset}/llm_gemini.json', 'w+') as outfile:
        json.dump({i: {j: k} for (i, j), k in responses_parsed.items()}, outfile)


if __name__ == '__main__':
    main('colqitt_et_al')
    main('matthews_et_al')
