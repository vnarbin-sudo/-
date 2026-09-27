"""
Тесты для предиктивного ядра.
"""
import pytest
import json
import pandas as pd
from src.config import MODELS_DIR, ENTERPRISES

def test_models_and_metrics_exist():
    metrics_file = MODELS_DIR / "metrics_summary.json"
    assert metrics_file.exists(), "Файл метрик metrics_summary.json отсутствует"
    
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    metrics = data["metrics"]
    for ent_id in ENTERPRISES.keys():
        assert ent_id in metrics, f"Метрики для {ent_id} отсутствуют"
        assert metrics[ent_id]["MAPE_pct"] < 20.0, f"MAPE для {ent_id} слишком высок: {metrics[ent_id]['MAPE_pct']}%"
        assert metrics[ent_id]["WAPE_pct"] < 20.0, f"WAPE для {ent_id} слишком высок: {metrics[ent_id]['WAPE_pct']}%"

def test_forecast_output():
    forecast_file = MODELS_DIR / "baseline_forecast.parquet"
    assert forecast_file.exists(), "Файл baseline_forecast.parquet отсутствует"
    df = pd.read_parquet(forecast_file)
    
    assert len(df) == len(ENTERPRISES) * 12, "Число прогнозных точек не соответствует 12 месяцам на предприятие"
    assert (df["forecast_usd"] > 0).all(), "Прогнозная стоимость экспорта должна быть строго положительной"
    assert (df["forecast_usd_upper"] >= df["forecast_usd"]).all(), "Верхняя граница CI должна быть >= точечного прогноза"
    assert (df["forecast_usd_lower"] <= df["forecast_usd"]).all(), "Нижняя граница CI должна быть <= точечного прогноза"
