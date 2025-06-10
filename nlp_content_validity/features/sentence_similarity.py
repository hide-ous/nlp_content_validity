import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import gensim.downloader as api

from sentence_transformers import SentenceTransformer


def load_sentence_transformers(model_name):
    return SentenceTransformer(f'sentence-transformers/{model_name}')


def T5_model():
    return load_sentence_transformers('sentence-t5-base')


def RoBERTa_model():
    return load_sentence_transformers('stsb-roberta-base')


def LSA_model():
    # corpus = api.load('text8')  # download the corpus and return it opened as an iterable

    lsa = make_pipeline(TfidfVectorizer(
        max_df=0.5,
        min_df=2,
        stop_words='english',  # keep things like not, etc.
    ), TruncatedSVD(n_components=40), Normalizer(copy=False))
    return lsa
    # return lsa.fit(map(lambda x: ' '.join(x), corpus))


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


# def embed_sentences(sentences, model_name='all-MiniLM-L6-v2'):
#     model = SentenceTransformer(model_name)
#     # Compute embeddings
#     embeddings = model.encode(sentences, convert_to_tensor=True)
#     return embeddings


# def cosine_similarity_from_embeddings(embeddings):
#     return util.cos_sim(embeddings, embeddings)


# def print_similarities(cosine_scores, sentences):
#     # Find the pairs with the highest cosine similarity scores
#     pairs = []
#     for i in range(len(cosine_scores) - 1):
#         for j in range(i + 1, len(cosine_scores)):
#             pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
#
#     # Sort scores in decreasing order
#     pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
#
#     for pair in pairs[0:10]:
#         i, j = pair['index']
#         print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))


def preprocess(text):
    return ''.join(c.lower() if c.isalpha() else ' ' for c in text)


def tokenize(text):
    return text.split()


def embed_from_words(list_of_embeddings):
    return np.mean(list_of_embeddings, axis=0)


if __name__ == '__main__':
    sentence_a = "paris is a beautiful city"
    sentence_b = "paris is a grogeous city"
    # model = sentence_similarity(model_name='distilbert-base-uncased', embedding_type='cls_token_embedding')
    # score = model.get_score(sentence_a, sentence_b, metric="cosine")
    # print(score)
