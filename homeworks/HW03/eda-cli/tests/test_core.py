from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(df, summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# Новые тесты для новых эвристик качества данных
def test_quality_flags_constant_columns():
    """Тест для проверки константных колонок."""
    # DataFrame с константной колонкой
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50],
        'constant_col': ['same', 'same', 'same', 'same', 'same']  # Константная
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    # Проверяем, что флаг установлен правильно
    assert flags['has_constant_columns'] == True
    assert 'constant_col' in flags['constant_columns']
    assert flags['constant_columns_count'] == 1
    assert 0.0 <= flags['quality_score'] <= 1.0


def test_quality_flags_high_cardinality():
    """Тест для проверки высокой кардинальности."""
    # DataFrame с высокой кардинальностью
    data = {'cat_col': [f'value_{i}' for i in range(150)]}  # 150 уникальных значений
    df = pd.DataFrame(data)

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    # Проверяем, что флаг установлен
    assert flags['has_high_cardinality_categoricals'] == True
    assert flags['high_cardinality_count'] >= 1
    assert len(flags['high_cardinality_columns']) >= 1
    assert 0.0 <= flags['quality_score'] <= 1.0


def test_quality_flags_id_duplicates():
    """Тест для проверки дубликатов ID."""
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 1, 2],  # Дубликаты
        'value': [10, 20, 30, 40, 50]
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    assert flags['has_suspicious_id_duplicates'] == True
    assert 'user_id' in flags['id_duplicates']
    assert flags['id_duplicates']['user_id']['duplicate_count'] > 0
    assert 0.0 <= flags['quality_score'] <= 1.0


def test_quality_flags_many_zeros():
    """Тест для проверки нулевых значений."""
    df = pd.DataFrame({
        'numeric_col': [0, 0, 0, 0, 1],  # 80% нулей
        'another_col': [1, 2, 3, 4, 5]
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    assert flags['has_many_zero_values'] == True
    assert len(flags['many_zero_columns']) >= 1
    assert 'numeric_col' in [col['column'] for col in flags['many_zero_columns']]
    assert 0.0 <= flags['quality_score'] <= 1.0


def test_quality_flags_no_problems():
    """Тест для датасета без проблем качества."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B']
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    # Проверяем, что все флаги проблем ложные
    assert flags['has_constant_columns'] == False
    assert flags['has_high_cardinality_categoricals'] == False
    assert flags['has_suspicious_id_duplicates'] == False
    assert flags['has_many_zero_values'] == False
    assert 0.0 <= flags['quality_score'] <= 1.0


def test_top_categories_with_custom_k():
    """Тест функции top_categories с кастомным значением top_k."""
    df = pd.DataFrame({
        'category': ['A', 'B', 'C', 'A', 'B', 'D', 'E', 'F', 'G', 'H'] * 2,
        'value': range(20)
    })

    # Проверяем с разными значениями top_k
    top_3 = top_categories(df, max_columns=5, top_k=3)
    top_5 = top_categories(df, max_columns=5, top_k=5)

    assert 'category' in top_3
    assert 'category' in top_5
    assert len(top_3['category']) == 3
    assert len(top_5['category']) == 5


def test_quality_score_range():
    """Тест для проверки, что quality_score всегда в диапазоне 0-1."""
    # Создаем несколько разных датасетов
    test_cases = [
        pd.DataFrame({'col': [1, 2, 3]}),  # Хороший датасет
        pd.DataFrame({'col': [None, None, None]}),  # Все пропуски
        pd.DataFrame({'col': list(range(200))}),  # Много строк
        pd.DataFrame({f'col_{i}': list(range(10)) for i in range(150)})  # Много колонок
    ]

    for df in test_cases:
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(df, summary, missing_df)

        assert 0.0 <= flags['quality_score'] <= 1.0, f"Failed for df shape {df.shape}"