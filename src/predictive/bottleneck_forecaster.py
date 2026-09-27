"""
Предиктивная модель краткосрочного прогнозирования узких мест и очередей на границах.
Прогнозирует время ожидания на 30 дней вперед для всех стратегических узлов.
"""
import duckdb
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.config import DB_PATH, LOGISTICS_NODES

class BottleneckForecaster:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def forecast_next_30_days(self) -> pd.DataFrame:
        """
        Прогноз очередей и статуса узлов на 30 дней вперед с учетом тренда и сезонности.
        """
        con = duckdb.connect(str(self.db_path), read_only=True)
        # Получаем исторические данные по узлам за последние 90 дней
        df_recent = con.execute("""
            SELECT date, node_id, queue_hours, traffic_load_ratio, status
            FROM telematics_nodes
            WHERE date >= (SELECT MAX(date) - INTERVAL 90 DAY FROM telematics_nodes)
            ORDER BY node_id, date
        """).df()
        con.close()

        last_date = df_recent["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq="D")
        
        forecast_records = []
        for node_id, ninfo in LOGISTICS_NODES.items():
            node_hist = df_recent[df_recent["node_id"] == node_id]
            if node_hist.empty:
                continue

            last_queue = node_hist["queue_hours"].iloc[-1]
            last_load = node_hist["traffic_load_ratio"].iloc[-1]
            hist_mean = node_hist["queue_hours"].mean()

            # Расчет локального тренда за 30 дней
            if len(node_hist) >= 30:
                y = node_hist["queue_hours"].values[-30:]
                x = np.arange(len(y))
                slope, _ = np.polyfit(x, y, 1)
            else:
                slope = 0.0

            # Симуляция 30 дней вперед
            for day_idx, f_date in enumerate(future_dates):
                # Сезонность дня недели (выходные дни / начало недели)
                dow = f_date.dayofweek
                dow_factor = 1.15 if dow in [0, 4, 5] else 0.95
                
                # Прогноз очереди с затуханием тренда и шумом
                pred_q = (last_queue + slope * (day_idx + 1) * 0.7) * dow_factor
                pred_q = max(1.0, float(pred_q))

                # Определение категории риска узкого места
                if pred_q < 12.0:
                    risk_cat = "LOW"
                    color_code = "green"
                    action_rec = "Ограничений нет. Рекомендуемый транзитный коридор."
                elif pred_q < 36.0:
                    risk_cat = "MODERATE"
                    color_code = "orange"
                    action_rec = "Умеренная нагрузка. Планировать подачу документов за 24ч."
                elif pred_q < 72.0:
                    risk_cat = "HIGH"
                    color_code = "red"
                    action_rec = "Высокий риск простоя. Рекомендуется перенаправление на Ж/Д."
                else:
                    risk_cat = "CRITICAL"
                    color_code = "darkred"
                    action_rec = "КРИТИЧЕСКИЙ БЛОК. Категорически избегать автотранзита!"

                forecast_records.append({
                    "forecast_date": f_date,
                    "day_ahead": day_idx + 1,
                    "node_id": node_id,
                    "node_name": ninfo["name"],
                    "node_type": ninfo["type"],
                    "predicted_queue_hours": round(pred_q, 1),
                    "risk_category": risk_cat,
                    "color_code": color_code,
                    "recommendation": action_rec
                })

        return pd.DataFrame(forecast_records)

    def get_current_bottleneck_radar(self) -> pd.DataFrame:
        """
        Сводный радар состояния узлов на ближайшую неделю.
        """
        df_30 = self.forecast_next_30_days()
        df_week = df_30[df_30["day_ahead"] <= 7].groupby(["node_id", "node_name", "node_type"]).agg(
            avg_7d_queue_hours=("predicted_queue_hours", "mean"),
            max_7d_queue_hours=("predicted_queue_hours", "max"),
            risk_category=("risk_category", lambda x: pd.Series.mode(x)[0]),
            recommendation=("recommendation", "first")
        ).reset_index().sort_values("avg_7d_queue_hours", ascending=False)
        return df_week

if __name__ == "__main__":
    bf = BottleneckForecaster()
    radar = bf.get_current_bottleneck_radar()
    print("=== ПРЕДИКТИВНЫЙ РАДАР УЗКИХ МЕСТ (7 ДНЕЙ) ===")
    print(radar[["node_name", "avg_7d_queue_hours", "risk_category"]])
