"""
ETL пайплайн загрузки и валидации данных в аналитическое хранилище DuckDB.
"""
import duckdb
import polars as pl
import pandas as pd
from pathlib import Path
from src.config import DB_PATH, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.ingestion.synthetic_generator import SyntheticDataGenerator

class IngestionPipeline:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.generator = SyntheticDataGenerator(seed=42)

    def run_full_ingestion(self):
        print(">>> [Data Ingestion] Запуск генерации первичных потоков данных...")
        
        # 1. Биржевые котировки и валюты
        print("  - Генерация мировых котировок сырья и курсов валют (LME, Freight, FX)...")
        df_market = self.generator.generate_commodity_and_fx()
        market_file = RAW_DATA_DIR / "market_indicators.parquet"
        df_market.to_parquet(market_file, index=False)

        # 2. Телематика узлов
        print("  - Генерация телематики логистических узлов и погранпереходов (GPS/ГЛОНАСС)...")
        df_telematics = self.generator.generate_telematics_and_nodes(df_market)
        telematics_file = RAW_DATA_DIR / "telematics_nodes.parquet"
        df_telematics.to_parquet(telematics_file, index=False)

        # 3. ГТД и банковские проводки
        print("  - Генерация реестра электронных таможенных деклараций и валютных сделок...")
        df_gtd, df_bank = self.generator.generate_customs_declarations(df_market, df_telematics)
        gtd_file = RAW_DATA_DIR / "customs_declarations.parquet"
        bank_file = RAW_DATA_DIR / "bank_transactions.parquet"
        df_gtd.to_parquet(gtd_file, index=False)
        df_bank.to_parquet(bank_file, index=False)

        print(f"  > Сгенерировано: ГТД = {len(df_gtd):,}, Телематика = {len(df_telematics):,}, Банковские сделки = {len(df_bank):,}")

        # 4. Сохранение в аналитическую базу DuckDB
        print(f">>> [Data Storage] Загрузка данных в аналитическую БД DuckDB: {self.db_path.name}...")
        con = duckdb.connect(str(self.db_path))

        con.execute(f"CREATE OR REPLACE TABLE market_indicators AS SELECT * FROM read_parquet('{market_file.as_posix()}');")
        con.execute(f"CREATE OR REPLACE TABLE telematics_nodes AS SELECT * FROM read_parquet('{telematics_file.as_posix()}');")
        con.execute(f"CREATE OR REPLACE TABLE customs_declarations AS SELECT * FROM read_parquet('{gtd_file.as_posix()}');")
        con.execute(f"CREATE OR REPLACE TABLE bank_transactions AS SELECT * FROM read_parquet('{bank_file.as_posix()}');")

        # 5. Создание материализованного аналитического временного ряда (помесячно по заводам)
        print(">>> [Data Processing] Формирование сводного помесячного датасета для предиктивного ядра...")
        monthly_query = """
        CREATE OR REPLACE TABLE monthly_export_features AS
        WITH monthly_trade AS (
            SELECT 
                DATE_TRUNC('month', date) AS month,
                enterprise_id,
                hs_code,
                SUM(units) AS total_units,
                SUM(contract_value_usd) AS total_usd,
                SUM(contract_value_byn) AS total_byn,
                COUNT(decl_id) AS shipments_count,
                AVG(contract_value_usd / NULLIF(units, 0)) AS avg_unit_price_usd
            FROM customs_declarations
            GROUP BY 1, 2, 3
        ),
        monthly_market AS (
            SELECT 
                DATE_TRUNC('month', date) AS month,
                AVG(steel_hrc_usd) AS avg_steel_price,
                AVG(copper_usd) AS avg_copper_price,
                AVG(freight_index_usd) AS avg_freight_index,
                AVG(rate_usd_byn) AS avg_usd_byn,
                AVG(rate_rub_byn_per_100) AS avg_rub_byn,
                AVG(rate_cny_byn_per_10) AS avg_cny_byn
            FROM market_indicators
            GROUP BY 1
        ),
        monthly_telematics AS (
            SELECT 
                DATE_TRUNC('month', date) AS month,
                AVG(queue_hours) AS avg_border_queue_hours,
                AVG(traffic_load_ratio) AS avg_traffic_load,
                SUM(units_in_transit) AS total_units_in_transit
            FROM telematics_nodes
            GROUP BY 1
        ),
        monthly_receivables AS (
            SELECT 
                DATE_TRUNC('month', shipment_date) AS month,
                enterprise_id,
                SUM(CASE WHEN payment_status = 'OVERDUE' THEN amount_usd_equiv ELSE 0 END) AS overdue_receivables_usd,
                AVG(compliance_delay_days) AS avg_compliance_delay
            FROM bank_transactions
            GROUP BY 1, 2
        )
        SELECT 
            t.month,
            t.enterprise_id,
            t.hs_code,
            t.total_units,
            t.total_usd,
            t.total_byn,
            t.shipments_count,
            t.avg_unit_price_usd,
            m.avg_steel_price,
            m.avg_copper_price,
            m.avg_freight_index,
            m.avg_usd_byn,
            m.avg_rub_byn,
            m.avg_cny_byn,
            tel.avg_border_queue_hours,
            tel.avg_traffic_load,
            tel.total_units_in_transit,
            COALESCE(r.overdue_receivables_usd, 0) AS overdue_receivables_usd,
            COALESCE(r.avg_compliance_delay, 0) AS avg_compliance_delay
        FROM monthly_trade t
        LEFT JOIN monthly_market m ON t.month = m.month
        LEFT JOIN monthly_telematics tel ON t.month = tel.month
        LEFT JOIN monthly_receivables r ON t.month = r.month AND t.enterprise_id = r.enterprise_id
        ORDER BY t.enterprise_id, t.month;
        """
        con.execute(monthly_query)

        # Выгрузка обработанного датасета в parquet
        processed_file = PROCESSED_DATA_DIR / "monthly_export_features.parquet"
        con.execute(f"COPY monthly_export_features TO '{processed_file.as_posix()}' (FORMAT PARQUET);")
        
        row_count = con.execute("SELECT COUNT(*) FROM monthly_export_features").fetchone()[0]
        con.close()

        print(f">>> [Data Ingestion Complete] Сводный датасет сформирован: {row_count} записей. Сохранен в: {processed_file}")
        return processed_file

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run_full_ingestion()
