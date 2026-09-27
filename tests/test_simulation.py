"""
Тесты для сценарного имитационного уровня (Simulation Layer).
"""
import pytest
import pandas as pd
from src.config import MODELS_DIR
from src.simulation.scenario_engine import ScenarioParameters, ScenarioSimulationEngine

def test_precomputed_scenarios():
    scen_file = MODELS_DIR / "precomputed_scenarios.parquet"
    assert scen_file.exists(), "Файл precomputed_scenarios.parquet отсутствует"
    df = pd.read_parquet(scen_file)

    assert "simulated_usd" in df.columns
    assert "delta_usd" in df.columns
    assert "scenario" in df.columns
    
    scenarios = df["scenario"].unique()
    expected = ["BASELINE", "LOGISTICS_CRISIS", "RAW_COMMODITY_SURGE", "TARIFF_AND_FX_SHOCK", "DIVERSIFICATION_BOOM"]
    for s in expected:
        assert s in scenarios, f"Сценарий {s} отсутствует в результатах"

def test_stress_test_logic():
    baseline_file = MODELS_DIR / "baseline_forecast.parquet"
    base_df = pd.read_parquet(baseline_file)
    engine = ScenarioSimulationEngine(base_df)

    # Тест 1: Жесткий логистический шок должен снижать экспортную выручку
    crisis_params = ScenarioParameters(
        scenario_id="TEST_CRISIS",
        scenario_name="Test Crisis",
        description="Test",
        closed_nodes=["KOZLOVICHI", "KAMENNY_LOG"],
        freight_rate_change_pct=50.0
    )
    sim_df = engine.simulate(crisis_params)
    assert (sim_df["simulated_usd"] <= sim_df["forecast_usd"]).all()
    assert (sim_df["delta_usd"] <= 0).all()

    # Тест 2: Позитивный сценарий должен увеличивать выручку
    positive_params = ScenarioParameters(
        scenario_id="TEST_POSITIVE",
        scenario_name="Test Pos",
        description="Test",
        freight_rate_change_pct=-20.0,
        rub_exchange_rate_change_pct=10.0
    )
    pos_df = engine.simulate(positive_params)
    assert (pos_df["simulated_usd"] >= pos_df["forecast_usd"]).all()
