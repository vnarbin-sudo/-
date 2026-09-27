"""
Тесты для коннекторов к открытым реальным API (Этап 1).
"""
import pytest
from src.ingestion.live_connectors import NBRBLiveConnector, CommodityLiveConnector, BorderQueueLiveConnector
from src.ingestion.live_updater import LiveDataUpdater

def test_nbrb_connector():
    connector = NBRBLiveConnector()
    rates = connector.fetch_rates()
    assert "rate_usd_byn" in rates
    assert "rate_rub_byn_per_100" in rates
    assert "rate_cny_byn_per_10" in rates
    assert rates["rate_usd_byn"] > 2.0
    assert rates["rate_rub_byn_per_100"] > 2.0

def test_commodity_connector():
    connector = CommodityLiveConnector()
    quotes = connector.fetch_quotes()
    assert "oil_brent_usd" in quotes
    assert "copper_usd" in quotes
    assert quotes["oil_brent_usd"] > 30.0
    assert quotes["copper_usd"] > 1000.0

def test_border_queue_connector():
    connector = BorderQueueLiveConnector()
    queues = connector.fetch_live_queues()
    assert "KOZLOVICHI" in queues
    assert "queue_hours" in queues["KOZLOVICHI"]
    assert queues["KOZLOVICHI"]["queue_hours"] >= 0

def test_live_updater_execution():
    updater = LiveDataUpdater()
    meta = updater.run_sync()
    assert meta["status"] == "SUCCESS"
    assert "last_sync_timestamp" in meta
    assert updater.status_file.exists()
