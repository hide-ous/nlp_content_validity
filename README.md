# nlp_content_validity

Replication material for the paper _"Title to be Defined"_.

## Python Dependencies
This installation guide assumes you are on Windows on a GPU-endowed machine, with CUDA installed.

1. Install python 3.10+
2. Clone the project: `git clone ...`
3. Enter the project directory `cd nlp_content_validity`
4. *\[Optional but highly encouraged\]* create a virtual environment `python -m venv .venv` 
5. *\[Optional but highly encouraged\]* activate the virtual environment. On Windows: 
6. Install python dependencies: `pip install -r requirements.txt`
7. Download Mistral and install Llama cpp wrapper. On Windows:
    ```ps1
    huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models/ 
    
    # on *nix
    # $env:CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
    # on powershell:
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    
    # copy everything from: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\extras\visual_studio_integration\MSBuildExtensions To: C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations
    pip install --no-cache-dir llama-cpp-python
    ```
## Project Set-Up
- Create an API key to query Gemini and store it as the variable `GEMINIKEY` in a `.env` file in the root directory of the project (see the `.env.example` file)

## Replication Steps
1. Place input files 


- run `get_llm_rating.py`:
  - input: the files `data/external/DATASET/[items,definitions,relations].csv`, where dataset is `colqitt_et_al` or `matthews_et_al`
  - output: the file `data/processed/DATASET/gemini_responses.json`, structured as follows:
    ```json
    [{items_scale_id:{
        definition_scale_id:[item_1_vs_definition_similarity, ..., item_n_vs_definition_similarity]}},
    ...
    ]
    ```
## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         nlp_content_validity and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── nlp_content_validity   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes nlp_content_validity a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
