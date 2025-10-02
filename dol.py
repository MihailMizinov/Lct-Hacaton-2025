import json
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# Список ключевых слов для мультилейблинга
label_keywords = {
    'cards': [
        'карта', 'карту', 'карты', 'картой', 'карте',
        'банковская карта', 'дебетовая карта', 'кредитная карта',
        'выпустить карту', 'перевыпустить карту'
    ],
    'app': [
        'приложение', 'в приложении', 'через приложение',
        'мобильное приложение', 'скачал приложение', 'не работает приложение',
        'обновить приложение', 'ошибка в приложении'
    ],
    'support': [
        'поддержка', 'служба поддержки', 'саппорт',
        'техподдержка', 'связаться с поддержкой', 'обратиться в поддержку',
        'поддержке', 'написал в поддержку', 'ответ поддержки'
    ],
    'application': [
        'оформил', 'оформление', 'заявка', 'подал заявку', 'подать заявку',
        'анкета', 'рассмотрение заявки', 'одобрение', 'заполнил анкету',
        'ожидание решения', 'подача заявки'
    ],
    'commission': [
        'комиссия', 'комиссию', 'сняли комиссию', 'без комиссии',
        'взяли комиссию', 'почему комиссия', 'размер комиссии',
        'комиссионный сбор', 'списание комиссии'
    ],
    'subscription': [
        'подписка', 'подписку', 'подписки', 'отмена подписки', 'отписаться',
        'продлили подписку', 'продление подписки', 'списание за подписку',
        'оплата подписки', 'автосписание','автопродление', 'подписался', 'отменить подписку'
    ],
    'website': [
        'сайт', 'на сайте', 'работа сайта', 'сайт не работает', 'ошибка на сайте',
        'не загружается сайт', 'проблема с сайтом', 'зависает сайт', 'страница не открывается',
        'сайт недоступен', 'сбой сайта', 'не могу зайти на сайт'
    ],
    'general': [
        'вопрос', 'помогите', 'непонятно', 'не работает', 'проблема',
        'не могу', 'что делать', 'как быть', 'нужно уточнить'
    ]
}

# Перевод английских лейблов в русский
label_translation = {
    'cards': 'Карты',
    'app': 'Мобильное приложение',
    'support': 'Поддержка',
    'application': 'Оформление заявки',
    'commission': 'Комиссия',
    'subscription': 'Подписки',
    'website': 'Работа сайта',
    'general': 'Общее'
}

def assign_multilabels(review):
    labels = []
    review_low = review.lower()
    for label, keywords in label_keywords.items():
        if any(keyword in review_low for keyword in keywords):
            labels.append(label)
    if not labels:
        labels.append('general')
    return labels

# Функция для преобразования числового сентимента в строку
def prettify_sentiment(sentiment):
    # Если one-hot или массив — индекс максимума
    if isinstance(sentiment, np.ndarray):
        if sentiment.ndim == 0 or (sentiment.ndim == 1 and sentiment.shape[0] == 1):
            idx = int(sentiment.item())
        else:
            idx = int(np.argmax(sentiment))
    else:
        idx = int(sentiment)

    sentiment_map = {
        0: "отрицательно",
        1: "нейтрально",
        2: "положительно"
    }
    return sentiment_map.get(idx, "нейтрально")

# Загрузка эмбеддера и модели тональности
embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
model_sentiment = joblib.load('model.pkl')

# Тестовые данные. Если нужен input.json, раскомментируй ниже.
test_data = [
    {"id": 1, "text": "Очень понравилось оформление дебетовой карты, быстро и удобно!"},
    {"id": 2, "text": "Сняли комиссию за перевод, хотя обещали без комиссии."},
    {"id": 3, "text": "Сайт не работает уже второй день, невозможно выполнить операции."},
    {"id": 4, "text": "Приложение постоянно вылетает после обновления."},
    {"id": 5, "text": "Служба поддержки помогла решить проблему очень быстро."},
    {"id": 6, "text": "Подписку продлили без моего согласия, деньги списали автоматически."},
    {"id": 7, "text": "Оформление заявки на кредит прошло успешно."},
    {"id": 8, "text": "Вопрос: как быть, если не приходит СМС?"},
    {"id": 9, "text": "Спасибо банку за качественный сервис!"},
    {"id": 10, "text": "Нейтральный отзыв, просто проверяю функционал."}
]

# Если нужен реальный input.json — раскомментируй:
# with open('input.json', 'r', encoding='utf-8') as f:
#     test_data = json.load(f)

result = {"predictions": []}
texts = [item['text'] for item in test_data]
ids = [item['id'] for item in test_data]

X = embedder.encode(texts, show_progress_bar=True)
sentiments = model_sentiment.predict(X)

for idx, text in enumerate(texts):
    multilabels = assign_multilabels(text)
    topics = [label_translation.get(label, label) for label in multilabels]
    sentiment_raw = sentiments[idx]
    print(f"TEXT: {text}")
    print(f"RAW SENTIMENT: {sentiment_raw}")
    sentiment_pretty = prettify_sentiment(sentiment_raw)
    print(f"PRETTY SENTIMENT: {sentiment_pretty}")
    result["predictions"].append({
        "id": ids[idx],
        "topics": topics,
        "sentiments": [sentiment_pretty]
    })

with open('predictions.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("JSON сохранён в predictions.json")
