"""
Модуль Feature Engineering для предиктивного ядра экспорта машиностроения.
Формирует опережающие индикаторы, временные лаги, сезонные гармоники и макро-факторы.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

class ExportFeatureEngineer:
    def __init__(self):
        self.feature_cols: List[str] = []

    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Генерация признаков для каждого предприятия во времени.
        """
        df = df.copy()
        df["month"] = pd.to_datetime(df["month"])
        df = df.sort_values(["enterprise_id", "month"]).reset_index(drop=True)

        dfs = []
        for ent_id, group in df.groupby("enterprise_id"):
            g = group.copy().sort_values("month").reset_index(drop=True)

            # 1. Календарные и сезонные признаки (гармоники Фурье)
            month_num = g["month"].dt.month
            g["month_sin"] = np.sin(2 * np.pi * month_num / 12)
            g["month_cos"] = np.cos(2 * np.pi * month_num / 12)
            g["quarter"] = g["month"].dt.quarter
            g["year_trend"] = g["month"].dt.year - 2021 + month_num / 12.0

            # 2. Авторегрессионные лаги целевой переменной (объем экспорта в млн USD)
            g["export_usd_lag1"] = g["total_usd"].shift(1)
            g["export_usd_lag2"] = g["total_usd"].shift(2)
            g["export_usd_lag3"] = g["total_usd"].shift(3)
            g["export_usd_lag12"] = g["total_usd"].shift(12)  # годовая сезонность

            # 3. Скользящие средние (Rolling Statistics)
            g["export_usd_roll_mean3"] = g["total_usd"].shift(1).rolling(3, min_periods=1).mean()
            g["export_usd_roll_mean6"] = g["total_usd"].shift(1).rolling(6, min_periods=1).mean()
            g["export_usd_roll_std3"] = g["total_usd"].shift(1).rolling(3, min_periods=1).std().fillna(0)

            # 4. Опережающие телематические индикаторы (партий в пути и очередей)
            # Техника в пути в прошлом месяце t-1 превращается в закрытый экспорт в t
            g["transit_units_lag1"] = g["total_units_in_transit"].shift(1)
            g["border_queue_lag1"] = g["avg_border_queue_hours"].shift(1)
            g["border_friction_index"] = (g["avg_border_queue_hours"].shift(1) / 24.0) * g["avg_traffic_load"].shift(1)

            # 5. Сырьевые и фрахтовые факторы с лагом (влияние на себестоимость и отгрузки)
            g["steel_price_lag1"] = g["avg_steel_price"].shift(1)
            g["steel_price_lag3"] = g["avg_steel_price"].shift(3)
            g["freight_index_lag1"] = g["avg_freight_index"].shift(1)
            g["steel_change_pct"] = (g["avg_steel_price"].shift(1) - g["avg_steel_price"].shift(3)) / g["avg_steel_price"].shift(3)

            # 6. Валютные факторы и финансовая дебиторка
            g["usd_byn_lag1"] = g["avg_usd_byn"].shift(1)
            g["rub_byn_lag1"] = g["avg_rub_byn"].shift(1)
            g["overdue_ratio"] = g["overdue_receivables_usd"].shift(1) / (g["total_usd"].shift(1) + 1e-5)

            dfs.append(g)

        result_df = pd.concat(dfs, ignore_index=True)

        if is_training:
            # Отбрасываем первые 12 месяцев для честной оценки лагов
            result_df = result_df.dropna(subset=["export_usd_lag12"]).reset_index(drop=True)

        feature_cols = [
            "month_sin", "month_cos", "quarter", "year_trend",
            "export_usd_lag1", "export_usd_lag2", "export_usd_lag3", "export_usd_lag12",
            "export_usd_roll_mean3", "export_usd_roll_mean6", "export_usd_roll_std3",
            "transit_units_lag1", "border_queue_lag1", "border_friction_index",
            "steel_price_lag1", "steel_price_lag3", "freight_index_lag1", "steel_change_pct",
            "usd_byn_lag1", "rub_byn_lag1", "overdue_ratio"
        ]
        self.feature_cols = feature_cols
        return result_df
