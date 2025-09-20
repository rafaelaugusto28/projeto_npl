import nltk
from nltk.corpus import movie_reviews
import random

def load_movie_reviews():
    """
    Carrega o dataset de reviews de filmes do NLTK.
    Retorna uma lista de tuplas (palavras, categoria).
    """
    nltk.download('movie_reviews')
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    return documents
