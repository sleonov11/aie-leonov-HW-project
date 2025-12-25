```markdown
# S03/HW04 – eda_cli: мини-EDA для CSV с HTTP API

Небольшое CLI-приложение и HTTP API сервис для базового анализа CSV-файлов.
Используется в рамках Семинара 03 и Домашнего задания 04 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта:

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI (HW03)

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт с новыми параметрами из HW03

```bash
uv run eda-cli report data/example.csv \
  --out-dir reports \
  --max-hist-columns 6 \
  --top-k-categories 5 \
  --min-missing-share 0.3 \
  --title "Мой EDA отчет"
```

Новые параметры из HW03:
- `--max-hist-columns` – максимальное количество числовых колонок для гистограмм
- `--top-k-categories` – сколько top-значений выводить для категориальных признаков
- `--min-missing-share` – порог доли пропусков для проблемных колонок
- `--title` – заголовок отчёта

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Запуск HTTP API сервиса (HW04)

### Запуск сервера через uvicorn

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Сервис будет доступен по адресу: `http://localhost:8000`
Документация API (Swagger UI): `http://localhost:8000/docs`

### Доступные эндпоинты

#### 1. Проверка здоровья сервиса
```bash
GET /health
```

#### 2. Оценка качества датасета по агрегированным признакам
```bash
POST /quality
Content-Type: application/json

{
  "n_rows": 1000,
  "n_cols": 10,
  "max_missing_share": 0.1,
  "numeric_cols": 5,
  "categorical_cols": 5
}
```

#### 3. Оценка качества из CSV файла (с обработкой ошибок HTTP 400)
```bash
POST /quality-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
```

**Обрабатываемые ошибки (HTTP 400):**
- Неправильный content-type
- Ошибка чтения CSV
- Пустой CSV файл

#### 4. Полный набор флагов качества (дополнительный эндпоинт из HW03)
```bash
POST /quality-flags-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
```

**Возвращает полный набор флагов качества из HW03, включая:**
- Константные колонки (все значения одинаковые)
- Высокую кардинальность категориальных признаков (>100 уникальных значений)
- Дубликаты в ID-колонках (колонки с "id" в названии)
- Много нулевых значений в числовых колонках (>50%)

#### 5. Сводка по датасету
```bash
POST /summary-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
?example_values_per_column=3
```

#### 6. Топ категорий для категориальных признаков
```bash
POST /top-categories-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
?max_columns=5&top_k=5
```

#### 7. Матрица корреляций Пирсона
```bash
POST /correlation-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
```

#### 8. Первые N строк датасета
```bash
POST /head-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
?n=10
```

#### 9. Метрики работы сервиса
```bash
GET /metrics
```

### Примеры использования API через curl

#### Проверка здоровья:
```bash
curl -X GET "http://localhost:8000/health"
```

#### Оценка качества CSV:
```bash
curl -X POST "http://localhost:8000/quality-from-csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"
```

#### Получение флагов качества:
```bash
curl -X POST "http://localhost:8000/quality-flags-from-csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"
```

## Тесты

```bash
uv run pytest -q
```

## Структура проекта

```
src/eda_cli/
├── __init__.py
├── core.py                # Основная логика EDA (новые эвристики из HW03)
├── viz.py                 # Визуализации
├── cli.py                 # CLI интерфейс
└── api.py                 # HTTP API (FastAPI) - HW04

data/
└── example.csv            # Пример данных для тестирования

tests/
└── test_core.py           # Тесты для ядра EDA
```

## Зависимости

Основные зависимости:
- `pandas` - обработка данных
- `matplotlib` - визуализации
- `typer` - CLI интерфейс
- `fastapi` - HTTP API (HW04)
- `uvicorn[standard]` - ASGI сервер (HW04)
- `python-multipart` - обработка загрузки файлов (HW04)
- `pydantic` - валидация данных

## Особенности реализации (HW04)

1. **Обработка ошибок HTTP 400**: Все эндпоинты, принимающие CSV, возвращают 400 при:
   - Неправильном формате файла
   - Ошибках чтения CSV
   - Пустых данных

2. **Дополнительный эндпоинт**: `POST /quality-flags-from-csv` использует все новые эвристики из HW03

3. **Логирование**: Каждый запрос логируется с временем выполнения (latency_ms)

4. **Метрики**: Эндпоинт `/metrics` предоставляет статистику работы сервиса
