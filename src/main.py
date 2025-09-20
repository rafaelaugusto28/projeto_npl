from data_loader import load_movie_reviews
from preprocessing import build_features
from model import train_classifier, evaluate_classifier, predict_sentiment

def main():
    # 1. Carregar dataset
    documents = load_movie_reviews()

    # 2. Pré-processar (features + palavras importantes)
    featuresets, word_features = build_features(documents)

    # 3. Separar treino e teste
    train_set, test_set = featuresets[100:], featuresets[:100]

    # 4. Treinar modelo
    classifier = train_classifier(train_set)

    # 5. Avaliar modelo
    evaluate_classifier(classifier, test_set)

    # 6. Testar com frases novas
    frases = [
        "This movie was terrible and boring",
        "I really enjoyed this movie, it was fantastic!"
    ]

    for frase in frases:
        sentimento = predict_sentiment(classifier, frase, word_features)
        print(f"Frase: {frase} -> Sentimento previsto: {sentimento}")

if __name__ == "__main__":
    main()
