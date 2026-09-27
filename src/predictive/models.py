"""
Модели прогнозирования экспорта:
- BaselineTrendModel: Регуляризованная линейная модель с сезонными гармониками.
- XGBoostExportModel: Модель на основе градиентного бустинга.
- EnsembleExportModel: Взвешенный ансамбль с расчетом 95% доверительных интервалов.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class BaselineTrendModel:
    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class XGBoostExportModel:
    def __init__(self, n_estimators: int = 150, max_depth: int = 4, learning_rate: float = 0.05):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            importance_type="gain"
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(sorted(zip(feature_names, [float(x) for x in importances]), key=lambda x: x[1], reverse=True))

class EnsembleExportModel:
    def __init__(self, weight_xgb: float = 0.70, weight_baseline: float = 0.30):
        self.weight_xgb = weight_xgb
        self.weight_baseline = weight_baseline
        self.xgb_model = XGBoostExportModel()
        self.baseline_model = BaselineTrendModel()
        self.residual_std = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.xgb_model.fit(X, y)
        self.baseline_model.fit(X, y)
        
        # Оценка дисперсии остатков для доверительного интервала
        preds = self.predict_point(X)
        residuals = y - preds
        self.residual_std = float(np.std(residuals))
        return self

    def predict_point(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb_model.predict(X)
        p_base = self.baseline_model.predict(X)
        return self.weight_xgb * p_xgb + self.weight_baseline * p_base

    def predict_with_intervals(self, X: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_preds = self.predict_point(X)
        # Z-score для 95% = 1.96
        z = 1.96 if confidence == 0.95 else 1.645
        margin = z * self.residual_std
        lower_bound = np.maximum(0.0, point_preds - margin)
        upper_bound = point_preds + margin
        return point_preds, lower_bound, upper_bound
