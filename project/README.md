# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»


# Прогноз оттока клиентов (Customer Churn Prediction)

**Паспорт проекта**

- **Название:** Прогноз оттока клиентов телеком-компании
- **Автор:** Леонов Степан Сергеевич
- **Группа:** КМБО-11-24
- **Контакт:** tg: @qqzwh; mail: leonoffstepan@yandex.ru
- **Ссылка на репозиторий:** <[URL GitHub-репозитория](https://github.com/sleonov11/aie-leonov-HW-project)>

**Краткое описание**

Проект решает задачу бинарной классификации: предсказать, уйдёт ли клиент (Churn) на основе его персональных данных, параметров контракта и услуг.  
Используется открытый датасет Telco Customer Churn. Обучена модель логистической регрессии, упакованная в пайплайн с предобработкой.  
Результат – REST API на FastAPI с эндпоинтами `/predict`, `/health` и `/metrics`.

---

## 1. Структура проекта

```
├── artifacts/                # артефакты (модель)
│   └── model_pipeline.pkl
├── configs/
│   └── .env.example          # шаблон переменных окружения
├── notebooks/
│   ├── 01_eda.ipynb          # EDA и визуализации
│   └── 02_model_experiments.ipynb  # сравнение моделей
├── src/
│   ├── __init__.py
│   ├── train.py              # скрипт обучения финальной модели
│   └── service/
│       ├── __init__.py
│       ├── main.py           # FastAPI приложение
│       ├── model.py          # загрузка модели
│       └── utils.py          # конфигурация и логгер
├── tests/
│   └── test_service.py       # тесты API
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md                 # этот файл
├── report.md                 # отчёт по проекту
└── self-checklist.md         # чеклист самопроверки
```

---

## 2. Требования и установка

- Python >= 3.10
- Виртуальное окружение рекомендуется

```bash
git clone https://github.com/sleonov11/aie-leonov-HW-project
cd project
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# или .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 3. Как обучить модель

Модель обучается скриптом `src/train.py` на датасете Telco Customer Churn.

```bash
python -m src.train
```

После успешного выполнения в папке `artifacts/` появится файл `model_pipeline.pkl`.  
Скрипт автоматически загружает данные, выполняет предобработку (заполнение пропусков, стандартизация числовых признаков, one-hot кодирование категориальных) и обучает LogisticRegression.

---

## 4. Как запустить сервис

### Локально

```bash
python -m src.service.main
```
Сервис поднимется на порту, указанном в переменной окружения `SERVICE_PORT` (по умолчанию 8000).

### Через Docker

```bash
docker build -t churn-service .
docker run -p 8000:8000 --env-file configs/.env churn-service
```

Убедитесь, что в `configs/.env` заданы переменные (скопируйте из `.env.example`).

---

## 5. API и демонстрация

После запуска доступны:

- `GET /health` – статус сервиса
- `GET /metrics` – метрики запросов (счётчики, среднее время ответа)
- `POST /predict` – предсказание оттока

**Пример запроса через curl:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
      }
    ]
  }'
```

**Пример ответа:**

```json
{
  "predictions": [1],
  "probabilities": [0.723]
}
```

---

## 6. Тесты

```bash
pytest tests/
```

Тесты проверяют `/health` и `/predict` на корректность ответа и наличие полей.

---

## 7. Наблюдаемость

- Все запросы логируются через `logging`.
- Эндпоинт `/metrics` отдаёт общее количество запросов по каждому пути и среднее время ответа.

---

## 8. Демонстрация на защите

1. Запустить сервис локально или в Docker.
2. Показать `GET /health`, затем `POST /predict` с примером из документации.
3. Продемонстрировать ноутбук `02_model_experiments.ipynb` с таблицей сравнения моделей.
4. Объяснить, почему выбрана логистическая регрессия, показать метрики на тесте.

---

## 9. Ограничения и развитие

- Модель простая, без подбора гиперпараметров.
- Не реализована валидация входных данных (проверка наличия всех колонок).
- Можно добавить более сложные модели (CatBoost, XGBoost), тюнинг, SHAP-анализ, streamlit-демо.
