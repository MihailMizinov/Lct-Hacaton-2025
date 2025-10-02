import json
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("Загружаем модели...")

# ===== ТЕМЫ =====
topic_model = joblib.load("model.pkl")
print("topic_model загружен, ожидает признаков:", topic_model.n_features_in_)

# Эмбеддинги
embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# ===== СЕНТИМЕНТ =====
sentiment_model = pipeline("sentiment-analysis", model="cointegrated/rubert-tiny-sentiment")

# ===== ТЕСТОВЫЕ ТЕКСТЫ =====
test_texts = [
    "Не могу войти в приложение, пишет ошибка",
    "Проблема с банковской картой, не проходит оплата",
    "Как связаться с поддержкой?",
    "Оформил заявку на кредитную карту, все отлично!",
    "Сняли большую комиссию за перевод, это безобразие",
    "Не могу отменить подписку на премиум, помогите",
    "Сайт не работает, страница не грузится",
    "Общий вопрос по услугам банка",
    "Карта не работает в приложении, что делать?",
    "Написал в поддержку по проблеме с картой, спасибо за помощь!"
]

# ===== ПРЕДИКТ =====
print("\nПредсказываем темы...")
X_emb = embedder.encode(test_texts, show_progress_bar=False)
topic_preds = topic_model.predict(X_emb)

print("Предсказываем сентименты...")
sentiment_preds = sentiment_model(test_texts)

# ===== СБОР РЕЗУЛЬТАТОВ =====
result = {"predictions": []}
for idx, (text, topic, sent) in enumerate(zip(test_texts, topic_preds, sentiment_preds)):
    result["predictions"].append({
        "id": idx + 1,
        "text": text,
        "topics": [int(topic)] if np.isscalar(topic) else topic.tolist(),
        "sentiments": [sent["label"]]
    })

with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\nJSON сохранён в predictions.json")
