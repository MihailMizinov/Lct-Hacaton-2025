import joblib
from fastapi import FastAPI, UploadFile, Query
from fastapi.responses import HTMLResponse, JSONResponse
import json
import logging
from collections import Counter
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Запуск: python -m uvicorn main:app --reload


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Хранилище ---
data_storage = {
    "predictions": [],
    "all_topics": set(),
    "all_sentiments": set()
}

_models_lock = threading.Lock()
_embed_model = None
_sentiment_clf = None
_topic_model = None
_vectorizer = None
# --- Модель для эмбеддингов ---
def get_models():
    global _embed_model, _sentiment_clf, _topic_model, _vectorizer
    with _models_lock:
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
            with open("model_clf.pkl", "rb") as f:
                _sentiment_clf = pickle.load(f)
            try:
                _topic_model = joblib.load("topic_model.pkl")
                _vectorizer = joblib.load("vectorizer.pkl")
            except FileNotFoundError:
                _topic_model = None
                _vectorizer = None
    return _embed_model, _sentiment_clf, _topic_model, _vectorizer


# --- Ключевые слова для тем ---
label_keywords = {
    'cards': ['карта', 'карту', 'карты', 'картой', 'карте', 'банковская карта', 'дебетовая карта', 'кредитная карта', 'выпустить карту', 'перевыпустить карту'],
    'app': ['приложение', 'в приложении', 'через приложение', 'мобильное приложение', 'скачал приложение', 'не работает приложение', 'обновить приложение', 'ошибка в приложении'],
    'support': ['поддержка', 'служба поддержки', 'саппорт', 'техподдержка', 'связаться с поддержкой', 'обратиться в поддержку', 'поддержке', 'написал в поддержку', 'ответ поддержки'],
    'application': ['оформил', 'оформление', 'заявка', 'подал заявку', 'подать заявку', 'анкета', 'рассмотрение заявки', 'одобрение', 'заполнил анкету', 'ожидание решения', 'подача заявки'],
    'commission': ['комиссия', 'комиссию', 'сняли комиссию', 'без комиссии', 'взяли комиссию', 'почему комиссия', 'размер комиссии', 'комиссионный сбор', 'списание комиссии'],
    'subscription': ['подписка', 'подписку', 'подписки', 'отмена подписки', 'отписаться', 'продлили подписку', 'продление подписки', 'списание за подписку', 'оплата подписки', 'автосписание','автопродление', 'подписался', 'отменить подписку'],
    'website': ['сайт', 'на сайте', 'работа сайта', 'сайт не работает', 'ошибка на сайте', 'не загружается сайт', 'проблема с сайтом', 'зависает сайт', 'страница не открывается', 'сайт недоступен', 'сбой сайта', 'не могу зайти на сайт'],
    'general': ['вопрос', 'помогите', 'непонятно', 'не работает', 'проблема', 'не могу', 'что делать', 'как быть', 'нужно уточнить']
}

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

# --- Функция для мульти-тем ---
def assign_multilabels(text):
    text_lower = text.lower()
    labels = [label for label, kws in label_keywords.items() if any(k in text_lower for k in kws)]
    if not labels:
        labels = ['general']
    return labels

def ml_predict(text: str, idx: int):
    embed_model, sentiment_clf, topic_model, vectorizer = get_models()

    # 🔹 Темы
    topics = [label_translation.get(label, label) for label in assign_multilabels(text)]

    # 🔹 ML-топики (если все темы == 'Общее')
    if topics == ['Общее'] and topic_model is not None and vectorizer is not None:
        emb = embed_model.encode([text], show_progress_bar=False)
        probs = topic_model.predict_proba(emb)[0]
        top_indices = np.argsort(probs)[-3:][::-1]  # топ-3
        topic_names = getattr(topic_model, "classes_", [f"Тема {i}" for i in range(len(probs))])
        topics = [topic_names[i] for i in top_indices]

    # 🔹 Сентимент
    sentiments = []
    if sentiment_clf is not None and vectorizer is not None:
        X_vec = vectorizer.transform([text])
        pred = sentiment_clf.predict(X_vec)[0]
        probs = sentiment_clf.predict_proba(X_vec)[0]
        if "neutral" in sentiment_clf.classes_:
            neutral_idx = list(sentiment_clf.classes_).index("neutral")
            if probs[neutral_idx] >= max(probs):
                pred = "neutral"
        sentiments = [pred]

    return {"id": idx, "text": text, "topics": topics, "sentiments": sentiments}


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Анализ отзывов — Команда НКА</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                background: #f0f2f5;
                color: #333;
            }
            header {
                background: linear-gradient(90deg, #4CAF50, #2196F3);
                color: white;
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }
            header h1 { margin: 0; font-size: 28px; }
            header p { margin: 5px 0 0; font-size: 14px; }

            .container {
                max-width: 1100px;
                margin: 30px auto;
                padding: 20px;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            h2 {
                margin-top: 0;
                font-size: 20px;
                color: #444;
                border-left: 5px solid #2196F3;
                padding-left: 10px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-top: 15px;
                border-radius: 8px;
                overflow: hidden;
            }
            th, td {
                border: 1px solid #e0e0e0;
                padding: 10px;
                text-align: center;
            }
            th {
                background: #f9fafc;
                font-weight: bold;
            }
            tr:nth-child(even) { background: #fafafa; }
            select, button, input[type="file"] {
                margin: 5px;
                padding: 8px 12px;
                border: 1px solid #ccc;
                border-radius: 6px;
                font-size: 14px;
            }
            button {
                background: #2196F3;
                color: white;
                border: none;
                cursor: pointer;
                transition: 0.2s;
            }
            button:hover {
                background: #1976D2;
            }
            #sentimentChart {
                display: block;
                max-width: 400px;
                margin: 0 auto;
            }
            #statsText ul {
                list-style: none;
                padding: 0;
            }
            #statsText li {
                padding: 5px 0;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Анализ отзывов</h1>
            <p><i>Выполнено Командой <b>НКА — Нахальные Кодеры Алярм</b></i></p>
        </header>

        <div class="container">

            <div class="card">
                <h2>Загрузка данных</h2>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" name="file" />
                    <button type="submit">Загрузить и проанализировать</button>
                </form>
            </div>

            <div class="card">
                <h2>Фильтры</h2>
                <label>Тема: </label>
                <select id="topicFilter"><option value="">Все</option></select>
                <label>Сентимент: </label>
                <select id="sentimentFilter"><option value="">Все</option></select>
                <button id="applyFilters">Применить</button>
            </div>

            <div class="card">
                <h2>Результаты анализа</h2>
                <div id="results"></div>
            </div>

            <div class="card">
                <h2>Диаграмма</h2>
                <canvas id="sentimentChart"></canvas>
                <div id="statsText"></div>
            </div>
        </div>

        <script>
        const form = document.getElementById("uploadForm");
        const chartCtx = document.getElementById("sentimentChart").getContext("2d");
        const topicFilter = document.getElementById("topicFilter");
        const sentimentFilter = document.getElementById("sentimentFilter");
        const applyFilters = document.getElementById("applyFilters");
        let chart;

        async function loadData(topic="", sentiment="") {
            const url = `/get-data?topic=${topic}&sentiment=${sentiment}`;
            const response = await fetch(url);
            const data = await response.json();

            // Таблица (без текста!)
            let html = "<table><tr><th>ID</th><th>Темы</th><th>Сентименты</th></tr>";
            data.predictions.forEach(p => {
                html += `<tr>
                    <td>${p.id}</td>
                    <td>${p.topics.join(", ")}</td>
                    <td>${p.sentiments.join(", ")}</td>
                </tr>`;
            });
            html += "</table>";
            document.getElementById("results").innerHTML = html;

            // Диаграмма
            const stats = data.stats;
            const labels = Object.keys(stats);
            const values = Object.values(stats);
            const total = values.reduce((a, b) => a + b, 0);

            if (chart) chart.destroy();
            chart = new Chart(chartCtx, {
                type: "pie",
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        backgroundColor: ["#4CAF50", "#F44336", "#FFC107", "#2196F3"]
                    }]
                },
                options: {
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let count = context.raw;
                                    let percent = ((count / total) * 100).toFixed(1);
                                    return context.label + ": " + count + " (" + percent + "%)";
                                }
                            }
                        }
                    }
                }
            });

            // Текстовая статистика
            let statsHtml = "<h3>Подробная статистика</h3><ul>";
            labels.forEach((label, i) => {
                let count = values[i];
                let percent = ((count / total) * 100).toFixed(1);
                statsHtml += `<li>${label}: <b>${count}</b> отзывов (${percent}%)</li>`;
            });
            statsHtml += "</ul>";
            document.getElementById("statsText").innerHTML = statsHtml;

            // Фильтры (заполняем только при первой загрузке)
            if (topicFilter.options.length === 1) {
                data.all_topics.forEach(t => {
                    topicFilter.innerHTML += `<option value="${t}">${t}</option>`;
                });
            }
            if (sentimentFilter.options.length === 1) {
                data.all_sentiments.forEach(s => {
                    sentimentFilter.innerHTML += `<option value="${s}">${s}</option>`;
                });
            }
        }

        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            await fetch("/analyze-file", {
                method: "POST",
                body: formData
            });
            await loadData();
        });

        applyFilters.addEventListener("click", () => {
            loadData(topicFilter.value, sentimentFilter.value);
        });
        </script>
    </body>
    </html>
    """

@app.post("/analyze-file")
async def analyze_file(file: UploadFile):
    """Загрузка и анализ JSON"""
    try:
        raw = await file.read()
        data = json.loads(raw)
        predictions = [ml_predict(item["text"], idx) for idx, item in enumerate(data.get("data", []), start=1)]


        # сохраняем
        data_storage["predictions"] = predictions
        data_storage["all_topics"] = {t for p in predictions for t in p["topics"]}
        data_storage["all_sentiments"] = {s for p in predictions for s in p["sentiments"]}

        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/get-data")
async def get_data(topic: str = Query("", alias="topic"), sentiment: str = Query("", alias="sentiment")):
    """Возвращаем данные с фильтрацией"""
    preds = []
    for p in data_storage["predictions"]:
        if topic and topic not in p["topics"]:
            continue
        if sentiment and sentiment not in p["sentiments"]:
            continue
        preds.append(p)

    # статистика
    all_sentiments = [s for p in preds for s in p["sentiments"]]
    stats = dict(Counter(all_sentiments))

    return {
        "predictions": preds,
        "stats": stats,
        "all_topics": list(data_storage["all_topics"]),
        "all_sentiments": list(data_storage["all_sentiments"])
    }
