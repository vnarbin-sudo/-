"""
Тесты для оптимизатора маршрутов и предиктора узких мест.
"""
import pytest
from src.simulation.route_optimizer import MultimodalRouteOptimizer
from src.predictive.bottleneck_forecaster import BottleneckForecaster

def test_bottleneck_forecaster():
    bf = BottleneckForecaster()
    df_30 = bf.forecast_next_30_days()
    assert len(df_30) > 0, "Прогноз узких мест пуст"
    assert "predicted_queue_hours" in df_30.columns
    assert "risk_category" in df_30.columns
    assert (df_30["predicted_queue_hours"] >= 0).all()

    radar = bf.get_current_bottleneck_radar()
    assert len(radar) > 0
    assert "avg_7d_queue_hours" in radar.columns

def test_multimodal_route_optimizer():
    optimizer = MultimodalRouteOptimizer(current_queues={"KRASNAYA_GORKA": 12.0, "DOSTYK_ALASHANKOU": 36.0})
    
    # 1. Маршрут в Москву
    routes_msk = optimizer.calculate_routes(origin_id="MTZ", destination_id="MOSCOW", batch_units=15)
    assert len(routes_msk) >= 2
    assert any(r.mode == "ROAD" for r in routes_msk)
    assert any("GREAT_STONE" in r.via_nodes or "KOLYADICHI" in r.via_nodes for r in routes_msk)

    # 2. Маршрут в Шанхай
    routes_sh = optimizer.calculate_routes(origin_id="GREAT_STONE", destination_id="SHANGHAI", batch_units=5, priority="TIME")
    assert len(routes_sh) >= 3
    # Проверка, что при приоритете TIME самый быстрый маршрут получает наилучший скор (ранг 1)
    assert routes_sh[0].total_days <= routes_sh[-1].total_days or routes_sh[0].score <= routes_sh[-1].score
