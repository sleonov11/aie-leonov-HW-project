from __future__ import annotations

from time import perf_counter
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import (
    compute_quality_flags,
    missing_table,
    summarize_dataset,
    top_categories,
    correlation_matrix,
    DatasetSummary
)

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.3.0",
    description=(
        "HTTP-сервис для оценки качества датасетов с дополнительными эндпоинтами. "
        "Использует EDA-логику из eda-cli проекта."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Модели запросов/ответов ----------

class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, Any] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


class DatasetSummaryResponse(BaseModel):
    """Ответ с полной сводкой датасета."""

    n_rows: int = Field(..., description="Количество строк")
    n_cols: int = Field(..., description="Количество колонок")
    columns: list[Dict[str, Any]] = Field(..., description="Детальная информация по колонкам")
    latency_ms: float = Field(..., description="Время обработки запроса, миллисекунды")


class QualityFlagsResponse(BaseModel):
    """Ответ с полным набором флагов качества."""

    quality_score: float = Field(..., ge=0.0, le=1.0, description="Общая оценка качества")
    flags: Dict[str, Any] = Field(..., description="Полный набор флагов качества")
    latency_ms: float = Field(..., description="Время обработки запроса, миллисекунды")
    dataset_shape: dict[str, int] = Field(..., description="Размеры датасета")


# ---------- Вспомогательные функции ----------

def _read_csv_file(file: UploadFile) -> pd.DataFrame:
    """Чтение CSV файла с проверками."""
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail="Ожидается CSV-файл (content-type text/csv)."
        )

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    return df


# ---------- Системный эндпоинт ----------

@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.3.0",
    }


# ---------- Заглушка /quality по агрегированным признакам ----------

@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков (если есть числовые и категориальные)
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Флаги, которые могут быть полезны для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------

@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.
    """

    start = perf_counter()

    df = _read_csv_file(file)

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(df, summary, missing_df)

    # Ожидаем, что compute_quality_flags вернёт quality_score в [0,1]
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета берём из summary
    n_rows = summary.n_rows
    n_cols = summary.n_cols

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- Новый эндпоинт: /quality-flags-from-csv (Вариант A) ----------

@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["quality"],
    summary="Полный набор флагов качества из CSV-файла",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    Эндпоинт, который принимает CSV-файл и возвращает полный набор флагов качества,
    включая новые эвристики из HW03 (константные колонки, высокая кардинальность,
    дубликаты ID, нулевые значения).
    """

    start = perf_counter()

    df = _read_csv_file(file)

    # Используем EDA-ядро
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(df, summary, missing_df)

    # Получаем оценку качества
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))

    latency_ms = (perf_counter() - start) * 1000.0

    print(
        f"[quality-flags-from-csv] filename={file.filename!r} "
        f"n_rows={summary.n_rows} n_cols={summary.n_cols} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityFlagsResponse(
        quality_score=score,
        flags=flags_all,
        latency_ms=latency_ms,
        dataset_shape={"n_rows": summary.n_rows, "n_cols": summary.n_cols},
    )


# ---------- Новый эндпоинт: /summary-from-csv (Вариант B) ----------

@app.post(
    "/summary-from-csv",
    response_model=DatasetSummaryResponse,
    tags=["analysis"],
    summary="Полная сводка по датасету",
)
async def summary_from_csv(
        file: UploadFile = File(...),
        example_values_per_column: int = 3
) -> DatasetSummaryResponse:
    """
    Эндпоинт, который принимает CSV-файл и возвращает полную сводку по датасету,
    аналогичную выводу CLI команды overview.
    """

    start = perf_counter()

    df = _read_csv_file(file)

    # Генерируем сводку с указанным количеством примеров
    summary = summarize_dataset(df, example_values_per_column=example_values_per_column)

    latency_ms = (perf_counter() - start) * 1000.0

    print(
        f"[summary-from-csv] filename={file.filename!r} "
        f"n_rows={summary.n_rows} n_cols={summary.n_cols} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return DatasetSummaryResponse(
        n_rows=summary.n_rows,
        n_cols=summary.n_cols,
        columns=[col.to_dict() for col in summary.columns],
        latency_ms=latency_ms,
    )


# ---------- Новый эндпоинт: /top-categories-from-csv ----------

@app.post(
    "/top-categories-from-csv",
    tags=["analysis"],
    summary="Топ категорий для категориальных признаков",
)
async def top_categories_from_csv(
        file: UploadFile = File(...),
        max_columns: int = 5,
        top_k: int = 5
) -> Dict[str, Any]:
    """
    Эндпоинт, который возвращает топ-K значений для категориальных колонок.
    """

    start = perf_counter()

    df = _read_csv_file(file)

    # Получаем топ категорий
    top_cats = top_categories(df, max_columns=max_columns, top_k=top_k)

    # Конвертируем DataFrame в словари для JSON-сериализации
    result = {}
    for col_name, df_table in top_cats.items():
        result[col_name] = df_table.to_dict(orient="records")

    latency_ms = (perf_counter() - start) * 1000.0

    print(
        f"[top-categories-from-csv] filename={file.filename!r} "
        f"max_columns={max_columns} top_k={top_k} "
        f"found_categories={len(result)} latency_ms={latency_ms:.1f} ms"
    )

    return {
        "categories": result,
        "parameters": {"max_columns": max_columns, "top_k": top_k},
        "latency_ms": latency_ms,
        "dataset_shape": {"n_rows": len(df), "n_cols": len(df.columns)},
    }


# ---------- Новый эндпоинт: /correlation-from-csv ----------

@app.post(
    "/correlation-from-csv",
    tags=["analysis"],
    summary="Матрица корреляций для числовых признаков",
)
async def correlation_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Эндпоинт, который возвращает матрицу корреляций Пирсона для числовых колонок.
    """

    start = perf_counter()

    df = _read_csv_file(file)

    # Получаем матрицу корреляций
    corr_df = correlation_matrix(df)

    latency_ms = (perf_counter() - start) * 1000.0

    if corr_df.empty:
        correlation_data = {}
        message = "Недостаточно числовых колонок для вычисления корреляций"
    else:
        correlation_data = corr_df.to_dict()
        message = f"Корреляционная матрица для {len(corr_df.columns)} числовых колонок"

    print(
        f"[correlation-from-csv] filename={file.filename!r} "
        f"numeric_columns={len(corr_df.columns) if not corr_df.empty else 0} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return {
        "correlation_matrix": correlation_data,
        "message": message,
        "latency_ms": latency_ms,
        "dataset_shape": {"n_rows": len(df), "n_cols": len(df.columns)},
    }


# ---------- Дополнительный эндпоинт: /head-from-csv (Вариант C) ----------

@app.post(
    "/head-from-csv",
    tags=["exploration"],
    summary="Первые N строк датасета",
)
async def head_from_csv(
        file: UploadFile = File(...),
        n: int = 10
) -> Dict[str, Any]:
    """
    Эндпоинт, который возвращает первые N строк CSV-файла.
    """

    start = perf_counter()

    df = _read_csv_file(file)

    # Берем первые N строк
    if n > len(df):
        n = len(df)

    head_df = df.head(n)

    # Конвертируем в словарь для JSON
    result = head_df.to_dict(orient="records")

    latency_ms = (perf_counter() - start) * 1000.0

    print(
        f"[head-from-csv] filename={file.filename!r} "
        f"n={n} total_rows={len(df)} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return {
        "data": result,
        "parameters": {"n": n},
        "total_rows": len(df),
        "returned_rows": len(result),
        "latency_ms": latency_ms,
        "columns": list(df.columns),
    }


# ---------- Дополнительный эндпоинт: /metrics (Вариант F) ----------

# Простая in-memory статистика
_request_stats = {
    "total_requests": 0,
    "avg_latency_ms": 0.0,
    "endpoint_counts": {},
    "last_quality_score": None,
    "last_ok_for_model": None,
}


@app.get(
    "/metrics",
    tags=["system"],
    summary="Метрики работы сервиса",
)
async def get_metrics() -> Dict[str, Any]:
    """
    Эндпоинт, который возвращает статистику по работе сервиса.
    """
    return {
        "service": "dataset-quality-api",
        "version": "0.3.0",
        "metrics": _request_stats,
        "timestamp": pd.Timestamp.now().isoformat(),
    }