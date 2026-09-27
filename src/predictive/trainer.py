"""
Пайплайн обучения предиктивных моделей, кросс-валидации и генерации 12-месячного базового прогноза.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FORECAST_HORIZON_MONTHS, ENTERPRISES
from src.predictive.feature_engineering import ExportFeatureEngineer
from src.predictive.models import EnsembleExportModel, XGBoostExportModel, BaselineTrendModel

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100.0
    
    # Защита от деления на 0 при расчете MAPE
    nonzero_mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))) * 100.0
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    return {
        "MAE_USD": round(mae, 2),
        "RMSE_USD": round(rmse, 2),
        "WAPE_pct": round(wape, 2),
        "MAPE_pct": round(mape, 2),
        "R2_score": round(r2, 4)
    }

class PredictiveTrainer:
    def __init__(self):
        self.fe = ExportFeatureEngineer()
        self.metrics_summary: Dict[str, Any] = {}
        self.models_per_enterprise: Dict[str, EnsembleExportModel] = {}

    def run_training_pipeline(self) -> pd.DataFrame:
        print(">>> [Predictive Core] Старт обучения предиктивного ядра...")
        data_path = PROCESSED_DATA_DIR / "monthly_export_features.parquet"
        raw_df = pd.read_parquet(data_path)

        # Создаем фичи
        feat_df = self.fe.create_features(raw_df, is_training=True)
        feature_cols = self.fe.feature_cols
        
        all_forecasts = []
        importances_summary = {}

        for ent_id in ENTERPRISES.keys():
            print(f"  - Обучение ансамбля моделей для {ent_id} ({ENTERPRISES[ent_id]['name']})...")
            ent_data = feat_df[feat_df["enterprise_id"] == ent_id].copy().sort_values("month").reset_index(drop=True)

            # Holdout тест (последние 12 месяцев)
            test_size = 12
            train_df = ent_data.iloc[:-test_size]
            test_df = ent_data.iloc[-test_size:]

            X_train = train_df[feature_cols].values
            y_train = train_df["total_usd"].values
            X_test = test_df[feature_cols].values
            y_test = test_df["total_usd"].values

            # Оценка на тесте
            eval_model = EnsembleExportModel()
            eval_model.fit(X_train, y_train)
            y_pred_test, _, _ = eval_model.predict_with_intervals(X_test)
            ent_metrics = calculate_metrics(y_test, y_pred_test)
            self.metrics_summary[ent_id] = ent_metrics
            print(f"    > Метрики (Holdout 12M): WAPE = {ent_metrics['WAPE_pct']}%, MAPE = {ent_metrics['MAPE_pct']}%, R^2 = {ent_metrics['R2_score']}")

            # Обучение на полном объеме данных
            full_model = EnsembleExportModel()
            X_full = ent_data[feature_cols].values
            y_full = ent_data["total_usd"].values
            full_model.fit(X_full, y_full)
            self.models_per_enterprise[ent_id] = full_model

            # Важность признаков
            importances = full_model.xgb_model.get_feature_importances(feature_cols)
            importances_summary[ent_id] = importances

            # Сохранение обученной модели
            model_file = MODELS_DIR / f"ensemble_{ent_id}.joblib"
            joblib.dump(full_model, model_file)

            # 12-месячный прогноз в будущее (ауторегрессионная симуляция)
            last_month = ent_data["month"].max()
            future_dates = pd.date_range(last_month + pd.DateOffset(months=1), periods=FORECAST_HORIZON_MONTHS, freq="MS")

            # Текущее состояние для шага вперед
            curr_row = ent_data.iloc[-1].copy()
            future_preds = []

            for f_date in future_dates:
                # Обновляем временные признаки
                m_num = f_date.month
                curr_row["month"] = f_date
                curr_row["month_sin"] = np.sin(2 * np.pi * m_num / 12)
                curr_row["month_cos"] = np.cos(2 * np.pi * m_num / 12)
                curr_row["quarter"] = f_date.quarter
                curr_row["year_trend"] = f_date.year - 2021 + m_num / 12.0

                x_vector = np.array([[curr_row[col] for col in feature_cols]])
                p_val, p_low, p_high = full_model.predict_with_intervals(x_vector)
                pred_usd = float(p_val[0])
                pred_low = float(p_low[0])
                pred_high = float(p_high[0])

                future_preds.append({
                    "month": f_date,
                    "enterprise_id": ent_id,
                    "enterprise_name": ENTERPRISES[ent_id]["name"],
                    "forecast_usd": round(pred_usd, 2),
                    "forecast_usd_lower": round(pred_low, 2),
                    "forecast_usd_upper": round(pred_high, 2),
                    "forecast_units": int(round(pred_usd / ENTERPRISES[ent_id]["avg_unit_price_usd"])),
                    "scenario": "BASELINE"
                })

                # Сдвиг авторегрессионных переменных
                curr_row["export_usd_lag3"] = curr_row["export_usd_lag2"]
                curr_row["export_usd_lag2"] = curr_row["export_usd_lag1"]
                curr_row["export_usd_lag1"] = pred_usd

            all_forecasts.extend(future_preds)

        # Сохранение результатов
        forecast_df = pd.DataFrame(all_forecasts)
        forecast_file = MODELS_DIR / "baseline_forecast.parquet"
        forecast_df.to_parquet(forecast_file, index=False)

        metrics_file = MODELS_DIR / "metrics_summary.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": self.metrics_summary,
                "feature_importances": importances_summary,
                "feature_columns": feature_cols
            }, f, ensure_ascii=False, indent=2)

        print(f">>> [Predictive Core Complete] Прогноз сохранен в {forecast_file}")
        print(f">>> Метрики сохранены в {metrics_file}")
        return forecast_df

if __name__ == "__main__":
    trainer = PredictiveTrainer()
    trainer.run_training_pipeline()
