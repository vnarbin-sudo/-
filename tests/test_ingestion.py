"""
Тесты для модуля сбора и генерации первичных данных.
"""
import pytest
import duckdb
from pathlib import Path
from src.config import DB_PATH, RAW_DATA_DIR, PROCESSED_DATA_DIR

def test_raw_files_exist():
    expected_files = [
        "market_indicators.parquet",
        "telematics_nodes.parquet",
        "customs_declarations.parquet",
        "bank_transactions.parquet"
    ]
    for fname in expected_files:
        p = RAW_DATA_DIR / fname
        assert p.exists(), f"Файл {fname} не найден в {RAW_DATA_DIR}"
        assert p.stat().st_size > 0, f"Файл {fname} пуст"

def test_duckdb_tables():
    assert DB_PATH.exists(), "Файл DuckDB базы данных не существует"
    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    con.close()

    required = ["market_indicators", "telematics_nodes", "customs_declarations", "bank_transactions", "monthly_export_features"]
    for r in required:
        assert r in table_names, f"Таблица {r} отсутствует в DuckDB"

def test_processed_dataset():
    p = PROCESSED_DATA_DIR / "monthly_export_features.parquet"
    assert p.exists(), "Сводный датасет не найден"
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{p.as_posix()}')").df()
    con.close()

    assert len(df) > 0, "Сводный датасет пуст"
    assert "total_usd" in df.columns
    assert "total_units" in df.columns
    assert "avg_steel_price" in df.columns
    assert "avg_border_queue_hours" in df.columns
    assert (df["total_usd"] >= 0).all(), "Экспортная выручка не может быть отрицательной"
