import nltk

def build_features(documents, num_features=2000):
    """
    Cria um conjunto de features baseado nas palavras mais frequentes.
    Retorna (featuresets, word_features).
    """
    all_words = nltk.FreqDist(w.lower() for w in nltk.corpus.movie_reviews.words())
    word_features = list(all_words)[:num_features]

    def extract_features(document):
        words = set(document)
        return {word: (word in words) for word in word_features}

    featuresets = [(extract_features(d), c) for (d, c) in documents]
    return featuresets, word_features
