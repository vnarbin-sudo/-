"""
Калькулятор эффектов и библиотека типовых сценарных стресс-тестов.
"""
import pandas as pd
from typing import Dict, List, Any
from src.config import MODELS_DIR
from src.simulation.scenario_engine import ScenarioParameters, ScenarioSimulationEngine

STANDARD_SCENARIOS = [
    ScenarioParameters(
        scenario_id="BASELINE",
        scenario_name="Базовый инерционный сценарий",
        description="Сохранение текущих макроэкономических и логистических трендов без резких шоков.",
        closed_nodes=[],
        freight_rate_change_pct=0.0,
        border_delay_additional_days=0,
        steel_price_change_pct=0.0,
        rub_exchange_rate_change_pct=0.0,
        utilization_fee_increase_pct=0.0,
        payment_delay_days=0
    ),
    ScenarioParameters(
        scenario_id="LOGISTICS_CRISIS",
        scenario_name="Логистический шок: Блокада западных пунктов пропуска",
        description="Полное закрытие ПП Козловичи и Каменный Лог, рост ставок морского фрахта на 35% и увеличение очередей на 18 дней.",
        closed_nodes=["KOZLOVICHI", "KAMENNY_LOG"],
        freight_rate_change_pct=35.0,
        border_delay_additional_days=18,
        steel_price_change_pct=5.0,
        rub_exchange_rate_change_pct=0.0,
        utilization_fee_increase_pct=0.0,
        payment_delay_days=10
    ),
    ScenarioParameters(
        scenario_id="RAW_COMMODITY_SURGE",
        scenario_name="Сырьевой ценовой шок (Сталь +30%)",
        description="Резкий взлет цен на листовой горячекатаный прокат LME и спецстали на 30%, удорожание себестоимости.",
        closed_nodes=[],
        freight_rate_change_pct=10.0,
        border_delay_additional_days=0,
        steel_price_change_pct=30.0,
        copper_price_change_pct=20.0,
        rub_exchange_rate_change_pct=0.0,
        utilization_fee_increase_pct=0.0,
        payment_delay_days=0
    ),
    ScenarioParameters(
        scenario_id="TARIFF_AND_FX_SHOCK",
        scenario_name="Тарифно-валютный шок на целевом рынке (ЕАЭС)",
        description="Ослабление RUB/BYN на 12%, повышение ставок утильсбора в РФ на 25% и рост срока банковского комплаенса платежей на 20 дней.",
        closed_nodes=[],
        freight_rate_change_pct=0.0,
        border_delay_additional_days=5,
        steel_price_change_pct=0.0,
        rub_exchange_rate_change_pct=-12.0,
        utilization_fee_increase_pct=25.0,
        payment_delay_days=20
    ),
    ScenarioParameters(
        scenario_id="DIVERSIFICATION_BOOM",
        scenario_name="Оптимистичный сценарий: Прорыв на рынки Азии и Африки",
        description="Снижение затрат на логистику коридора «Север-Юг» (-15%), стабилизация расчетов, рост экспортных контрактов на 18%.",
        closed_nodes=[],
        freight_rate_change_pct=-15.0,
        border_delay_additional_days=-5,
        steel_price_change_pct=-5.0,
        rub_exchange_rate_change_pct=5.0,
        utilization_fee_increase_pct=0.0,
        payment_delay_days=-5
    )
]

class ImpactCalculator:
    def __init__(self):
        baseline_file = MODELS_DIR / "baseline_forecast.parquet"
        self.baseline_df = pd.read_parquet(baseline_file)
        self.engine = ScenarioSimulationEngine(self.baseline_df)

    def run_all_scenarios(self) -> pd.DataFrame:
        print(">>> [Simulation Layer] Запуск моделирования стандартных сценарных стресс-тестов...")
        all_results = []
        for p in STANDARD_SCENARIOS:
            print(f"  - Расчет сценария: {p.scenario_name} ({p.scenario_id})...")
            res_df = self.engine.simulate(p)
            all_results.append(res_df)

        combined_df = pd.concat(all_results, ignore_index=True)
        out_file = MODELS_DIR / "precomputed_scenarios.parquet"
        combined_df.to_parquet(out_file, index=False)
        print(f">>> [Simulation Layer Complete] Все сценарии рассчитаны и сохранены в: {out_file}")
        return combined_df

    def summarize_scenario(self, scenario_df: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегация эффектов по сценарию за 12 месяцев.
        """
        summary = scenario_df.groupby(["scenario", "scenario_name"]).agg(
            total_simulated_usd=("simulated_usd", "sum"),
            total_baseline_usd=("forecast_usd", "sum"),
            delta_usd=("delta_usd", "sum"),
            total_simulated_units=("simulated_units", "sum"),
            total_baseline_units=("forecast_units", "sum"),
            delta_units=("delta_units", "sum"),
            frozen_cashflow_usd=("delayed_cashflow_usd", "sum"),
            added_logistics_cost_usd=("additional_logistics_cost_usd", "sum")
        ).reset_index()
        summary["delta_usd_pct"] = (summary["delta_usd"] / summary["total_baseline_usd"]) * 100.0
        return summary

if __name__ == "__main__":
    calc = ImpactCalculator()
    results = calc.run_all_scenarios()
    summary = calc.summarize_scenario(results)
    print("\n=== СВОДКА СЦЕНАРНЫХ ЭФФЕКТОВ (12 МЕСЯЦЕВ) ===")
    print(summary[["scenario_id" if "scenario_id" in summary else "scenario", "total_simulated_usd", "delta_usd", "delta_usd_pct"]])
