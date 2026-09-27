"""
Сценарный движок имитационного моделирования (What-If Analysis).
Позволяет проводить стресс-тестирование экспорта белорусского машиностроения
при изменении логистических, сырьевых и тарифных условий.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

@dataclass
class ScenarioParameters:
    scenario_id: str
    scenario_name: str
    description: str
    # 1. Логистические параметры
    closed_nodes: List[str] = field(default_factory=list)
    freight_rate_change_pct: float = 0.0      # например, +35%
    border_delay_additional_days: int = 0      # например, +14 дней
    # 2. Сырьевые параметры
    steel_price_change_pct: float = 0.0        # например, +25%
    copper_price_change_pct: float = 0.0       # например, +15%
    # 3. Валютные и регуляторные параметры
    rub_exchange_rate_change_pct: float = 0.0  # изменение курса RUB/BYN (-10% девальвация RUB)
    utilization_fee_increase_pct: float = 0.0  # рост утилизационного сбора
    payment_delay_days: int = 0                # дополнительная задержка банковских расчетов

class ScenarioSimulationEngine:
    def __init__(self, baseline_forecast_df: pd.DataFrame):
        self.baseline_df = baseline_forecast_df.copy()

    def simulate(self, params: ScenarioParameters) -> pd.DataFrame:
        """
        Применение стресс-теста к базовой прогнозной траектории экспорта.
        """
        sim_df = self.baseline_df.copy()
        sim_df["scenario"] = params.scenario_id
        sim_df["scenario_name"] = params.scenario_name

        # Эластичность спроса и предложения по факторам:
        # 1. Логистический эффект (увеличение задержек и закрытие узлов)
        # Если закрыты ключевые узлы, часть поставок переносится на следующий месяц
        # и теряется из-за штрафов/отмены заказов (отток ~ 3-8% от объема)
        nodes_impact = len(params.closed_nodes) * 0.035
        freight_impact = (params.freight_rate_change_pct / 100.0) * 0.12
        delay_shift_ratio = min(0.30, (params.border_delay_additional_days / 30.0) * 0.20)

        # 2. Сырьевой эффект: сталь занимает ~25-35% в себестоимости сельхоз и карьерной техники
        # Рост себестоимости снижает маржинальность и приводит к замедлению темпов контрактации
        steel_cost_share = 0.30
        cost_inflation = (params.steel_price_change_pct / 100.0) * steel_cost_share
        demand_elasticity = 0.45  # при удорожании на 10%, спрос снижается на 4.5%
        commodity_impact = - cost_inflation * demand_elasticity

        # 3. Тарифно-валютный эффект (РФ потребляет ~75% экспорта машиностроения)
        # Ослабление рубля к BYN делает белорусские тракторы дороже для российских аграриев
        rf_share = 0.75
        fx_rub_impact = (params.rub_exchange_rate_change_pct / 100.0) * rf_share * 0.60
        fee_impact = - (params.utilization_fee_increase_pct / 100.0) * rf_share * 0.35

        # Суммарный коэффициент изменения объема (volume adjustment)
        net_volume_multiplier = 1.0 - (nodes_impact + freight_impact) + commodity_impact + fx_rub_impact + fee_impact
        # Ограничитель разумных границ стресс-теста (от -50% до +50%)
        net_volume_multiplier = np.clip(net_volume_multiplier, 0.50, 1.50)

        # Пересчет прогнозных значений
        sim_df["simulated_usd"] = np.round(sim_df["forecast_usd"] * net_volume_multiplier, 2)
        sim_df["simulated_units"] = np.round(sim_df["forecast_units"] * net_volume_multiplier).astype(int)
        
        # Дельта относительно базового прогноза
        sim_df["delta_usd"] = np.round(sim_df["simulated_usd"] - sim_df["forecast_usd"], 2)
        sim_df["delta_units"] = sim_df["simulated_units"] - sim_df["forecast_units"]
        sim_df["delta_pct"] = np.round((sim_df["delta_usd"] / sim_df["forecast_usd"]) * 100.0, 2)

        # Влияние на кассовые разрывы и дебиторскую задолженность (с учетом банковских задержек)
        total_delay = params.border_delay_additional_days + params.payment_delay_days
        sim_df["delayed_cashflow_usd"] = np.round(sim_df["simulated_usd"] * min(1.0, total_delay / 60.0), 2)
        sim_df["additional_logistics_cost_usd"] = np.round(sim_df["simulated_usd"] * (params.freight_rate_change_pct / 100.0) * 0.06, 2)

        return sim_df
