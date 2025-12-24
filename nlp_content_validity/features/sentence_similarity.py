import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import gensim.downloader as api

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, TextClassificationPipeline


def load_sentence_transformers(model_name):
    return SentenceTransformer(f'sentence-transformers/{model_name}', cache_folder='../../models/')


def T5_model():
    return load_sentence_transformers('sentence-t5-base')


def RoBERTa_model():
    return load_sentence_transformers('stsb-roberta-base')


def LSA_model():
    # corpus = api.load('text8')  # download the corpus and return it opened as an iterable

    lsa = make_pipeline(TfidfVectorizer(
        # max_df=0.5,
        # min_df=2,
        # stop_words='english',  # keep things like not, etc.
    ), TruncatedSVD(n_components=300), Normalizer(copy=False))
    return lsa


def fasttext_model():
    # https://github.com/piskvorky/gensim-data
    model = api.load("fasttext-wiki-news-subwords-300")
    model.fill_norms()
    return model


def w2v_model():
    model = api.load("word2vec-google-news-300")
    model.fill_norms()
    return model


def glove_model():
    model = api.load("glove-wiki-gigaword-300")
    model.fill_norms()
    return model


def preprocess(text):
    return ''.join(c.lower() if c.isalpha() else ' ' for c in text)


def tokenize(text):
    return text.split()


def embed_from_words(list_of_embeddings):
    return np.mean(list_of_embeddings, axis=0)

def contval_model(device, pretrained_model_name_or_path = 'dobolyilab/RATER-C'):


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=2
    ).to(device)

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, device=device)

    return pipe


if __name__ == '__main__':
    sentence_a = "paris is a beautiful city"
    sentence_b = "paris is a gorgeous city"
    # model = sentence_similarity(model_name='distilbert-base-uncased', embedding_type='cls_token_embedding')
    # score = model.get_score(sentence_a, sentence_b, metric="cosine")
    # print(score)
