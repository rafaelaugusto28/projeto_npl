import nltk

def train_classifier(train_set):
    """
    Treina um classificador Naive Bayes.
    """
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    return classifier

def evaluate_classifier(classifier, test_set):
    """
    Avalia o classificador e imprime a acurácia.
    """
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print(f"✅ Acurácia: {accuracy:.2f}")
    classifier.show_most_informative_features(10)

def predict_sentiment(classifier, text, word_features):
    """
    Prediz o sentimento de um texto novo.
    """
    words = set(text.split())
    features = {word: (word in words) for word in word_features}
    return classifier.classify(features)
