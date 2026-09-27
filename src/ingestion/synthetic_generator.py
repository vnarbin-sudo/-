"""
Генератор реалистичных синтетических первичных данных для цифрового двойника экспорта РБ.
Охватывает 4 потока:
1. Электронные таможенные декларации (ГТД).
2. Спутниковая телематика грузопотоков и очередей на узлах.
3. Банковский реестр паспортно-валютных сделок и платежей.
4. Мировые биржевые котировки сырья и индексы фрахта.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
from src.config import ENTERPRISES, LOGISTICS_NODES, START_DATE, END_DATE

class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.start_dt = pd.to_datetime(START_DATE)
        self.end_dt = pd.to_datetime(END_DATE)
        self.dates_daily = pd.date_range(self.start_dt, self.end_dt, freq="D")
        self.dates_monthly = pd.date_range(self.start_dt, self.end_dt, freq="MS")

    def generate_commodity_and_fx(self) -> pd.DataFrame:
        """
        Генерация биржевых котировок (LME сталь, медь, нефть, фрахт) и курсов валют
        с реалистичными историческими трендами 2021-2026.
        """
        n_days = len(self.dates_daily)
        t = np.linspace(0, 1, n_days)

        # 1. Цены на горячекатаный прокат LME Steel HRC ($/тонна)
        # Рост в 2021-2022 (до 1000-1100$), коррекция в 2023-2024 к 650-750$
        base_steel = 550 + 450 * np.exp(-((t - 0.25) / 0.15) ** 2) + 120 * np.sin(2 * np.pi * t * 2)
        noise_steel = np.random.normal(0, 15, n_days)
        steel_price = np.maximum(450, base_steel + noise_steel)

        # 2. Медь LME Copper ($/тонна)
        base_copper = 7800 + 2200 * t + 800 * np.sin(2 * np.pi * t * 3)
        copper_price = base_copper + np.random.normal(0, 80, n_days)

        # 3. Индекс контейнерного фрахта ($/FEU 40ft контейнер)
        # Огромный скачок 2021-2022 (до 8000$), спад в 2023, рост в 2024 из-за Суэца
        freight_index = 2000 + 6000 * np.exp(-((t - 0.28) / 0.12) ** 2) + 1500 * (t > 0.6) + np.random.normal(0, 100, n_days)
        freight_index = np.maximum(1400, freight_index)

        # 4. Курсы валют:
        # USD/BYN: постепенный рост с 2.5 до 3.3
        usd_byn = 2.55 + 0.75 * t + 0.15 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.02, n_days)
        # RUB/BYN: за 100 рос. руб. (колебания 3.3 - 3.7 BYN)
        rub_byn = 3.45 + 0.25 * np.sin(2 * np.pi * t * 2.5) + np.random.normal(0, 0.03, n_days)
        # CNY/BYN: рост юаня в расчетах
        cny_byn = (usd_byn / 7.15) * 10

        df = pd.DataFrame({
            "date": self.dates_daily,
            "steel_hrc_usd": np.round(steel_price, 2),
            "copper_usd": np.round(copper_price, 2),
            "freight_index_usd": np.round(freight_index, 2),
            "rate_usd_byn": np.round(usd_byn, 4),
            "rate_rub_byn_per_100": np.round(rub_byn, 4),
            "rate_cny_byn_per_10": np.round(cny_byn, 4),
        })
        return df

    def generate_telematics_and_nodes(self, fx_df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация телематических данных с погранпереходов и портов:
        очереди, среднее время простоя, статус узла.
        Учитывает закрытие пунктов пропуска на западной границе в 2022-2024.
        """
        records = []
        for dt in self.dates_daily:
            t = (dt - self.start_dt).days / (self.end_dt - self.start_dt).days
            for node_id, info in LOGISTICS_NODES.items():
                cap = info["capacity_trucks_day"] if "capacity_trucks_day" in info else info["capacity_teu_day"]

                # Симуляция западных санкций и ограничений на границах с Польшей и Литвой
                is_western = node_id in ["KOZLOVICHI", "KAMENNY_LOG", "GRIGOROVSHCHINA"]
                if is_western and dt >= pd.to_datetime("2022-03-01"):
                    # Резкий рост очередей, падение пропускной способности
                    traffic_load = np.random.uniform(0.85, 1.4)
                    queue_hours = 24 + 96 * t + np.random.exponential(18)
                    status = "RESTRICTED" if queue_hours < 72 else "CONGESTED"
                    if node_id == "KAMENNY_LOG" and dt >= pd.to_datetime("2024-03-01"):
                        queue_hours = 120 + np.random.exponential(30)
                        status = "CRITICAL"
                elif node_id in ["GREAT_STONE", "KOLYADICHI"]:
                    # Внутренние высокотехнологичные мультимодальные хабы РБ
                    traffic_load = 0.5 + 0.5 * t + np.random.uniform(-0.05, 0.1)
                    queue_hours = 3.5 + 2.0 * t + np.random.exponential(1.5)
                    status = "NORMAL" if traffic_load < 0.9 else "HIGH_LOAD"
                elif node_id == "MINSK2_AIR_CARGO":
                    # Экспресс-авиагрузовой терминал
                    traffic_load = 0.4 + 0.3 * t + np.random.uniform(-0.05, 0.05)
                    queue_hours = 1.5 + np.random.exponential(0.8)
                    status = "NORMAL"
                elif node_id in ["DOSTYK_ALASHANKOU", "ZABAIKALSK"]:
                    # Крупные сухопутные шлюзы в Китай (высокая загрузка контейнерами)
                    traffic_load = 0.75 + 0.35 * t + np.random.uniform(-0.05, 0.15)
                    queue_hours = 20 + 25 * t + np.random.exponential(6)
                    status = "NORMAL" if traffic_load < 0.95 else "HIGH_LOAD"
                elif node_id in ["PORT_BRONKA", "PORT_NOVOROSSIYSK", "ASTRAKHAN_OLYA", "KRASNAYA_GORKA"]:
                    # Переориентация на Восток и порты РФ (бум загрузки)
                    traffic_load = 0.7 + 0.4 * t + np.random.uniform(-0.1, 0.15)
                    queue_hours = 8 + 14 * t + np.random.exponential(4)
                    status = "NORMAL" if traffic_load < 0.95 else "HIGH_LOAD"
                else:
                    traffic_load = np.random.uniform(0.6, 0.9)
                    queue_hours = 10 + np.random.exponential(5)
                    status = "NORMAL"

                active_units_in_transit = int(cap * traffic_load * np.random.uniform(1.2, 2.5))

                records.append({
                    "date": dt,
                    "node_id": node_id,
                    "node_name": info["name"],
                    "node_type": info["type"],
                    "capacity_daily": cap,
                    "queue_hours": round(queue_hours, 1),
                    "traffic_load_ratio": round(traffic_load, 3),
                    "units_in_transit": active_units_in_transit,
                    "status": status
                })
        return pd.DataFrame(records)

    def generate_customs_declarations(self, fx_df: pd.DataFrame, telematics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Генерация реестра экспортных ГТД и связанных валютно-банковских сделок.
        """
        declarations = []
        bank_transactions = []
        decl_counter = 100000

        # Веса рынков до 2022 и после 2022 (переориентация экспорта)
        for dt in self.dates_monthly:
            month_idx = dt.month
            year = dt.year
            is_post_2022 = dt >= pd.to_datetime("2022-04-01")

            # Сезонный фактор для машиностроения (пик весной к посевной и осенью перед закрытием бюджетов)
            seasonality = 1.0 + 0.25 * np.sin((month_idx - 3) * np.pi / 6)

            # Получаем курсы на начало месяца
            fx_row = fx_df[fx_df["date"] == dt].iloc[0]
            usd_rate = fx_row["rate_usd_byn"]
            rub_rate = fx_row["rate_rub_byn_per_100"]

            for ent_id, ent_info in ENTERPRISES.items():
                base_vol = ent_info["base_monthly_volume"]
                unit_price = ent_info["avg_unit_price_usd"]

                # Корректировка на тренд адаптации
                growth_factor = 1.0 + 0.05 * (year - 2021)
                monthly_units = int(base_vol * seasonality * growth_factor * np.random.uniform(0.9, 1.12))

                # Распределение по рынкам
                markets = ent_info["key_markets"]
                if is_post_2022:
                    # Увеличение доли РФ и ЕАЭС до 80-85%
                    weights = [0.75 if m == "РФ" else 0.12 if m in ["Казахстан", "Узбекистан"] else 0.03 for m in markets]
                else:
                    weights = [0.60 if m == "РФ" else 0.15 if m in ["Казахстан", "Узбекистан"] else 0.07 for m in markets]
                weights = np.array(weights) / np.sum(weights)

                # Генерация партий отгрузок в течение месяца
                n_shipments = np.random.randint(15, 35)
                batch_sizes = np.random.multinomial(monthly_units, [1/n_shipments]*n_shipments)

                for i, batch_size in enumerate(batch_sizes):
                    if batch_size == 0:
                        continue
                    decl_counter += 1
                    ship_day = np.random.randint(1, 28)
                    ship_date = dt.replace(day=ship_day)
                    destination = np.random.choice(markets, p=weights)

                    # Логистический узел в зависимости от направления и типа груза
                    if destination == "РФ":
                        node = "GREAT_STONE" if ent_id == "GREAT_STONE" else "KRASNAYA_GORKA"
                    elif destination in ["Китай", "Монголия"]:
                        node = np.random.choice(["GREAT_STONE", "DOSTYK_ALASHANKOU", "ZABAIKALSK", "PORT_VLADIVOSTOK"], p=[0.35, 0.35, 0.15, 0.15])
                    elif destination in ["Казахстан", "Узбекистан"]:
                        node = np.random.choice(["KOLYADICHI", "GREAT_STONE", "KRASNAYA_GORKA"], p=[0.4, 0.3, 0.3])
                    elif destination in ["Индия", "Пакистан"]:
                        node = np.random.choice(["ASTRAKHAN_OLYA", "PORT_NOVOROSSIYSK", "PORT_BRONKA"], p=[0.5, 0.3, 0.2])
                    elif destination in ["ЮАР", "Зимбабве", "Египет"]:
                        node = np.random.choice(["PORT_BRONKA", "PORT_NOVOROSSIYSK"], p=[0.6, 0.4])
                    else:
                        node = "KOZLOVICHI" if not is_post_2022 else "PORT_BRONKA"

                    # Стоимость контракта
                    vol_units = int(batch_size)
                    total_usd = vol_units * unit_price * np.random.uniform(0.96, 1.04)
                    total_byn = total_usd * usd_rate
                    total_rub = (total_byn / (rub_rate / 100))

                    contract_currency = "RUB" if destination in ["РФ", "Казахстан"] else ("CNY" if destination == "Китай" else "USD")

                    decl_id = f"GTD-06532-{ship_date.strftime('%Y%m%d')}-{decl_counter}"
                    declarations.append({
                        "decl_id": decl_id,
                        "date": ship_date,
                        "enterprise_id": ent_id,
                        "enterprise_name": ent_info["name"],
                        "hs_code": ent_info["hs_code"],
                        "product_name": ent_info["product"],
                        "destination_country": destination,
                        "logistics_node": node,
                        "units": vol_units,
                        "contract_value_usd": round(total_usd, 2),
                        "contract_value_byn": round(total_byn, 2),
                        "contract_value_rub": round(total_rub, 2),
                        "contract_currency": contract_currency,
                        "incoterms": np.random.choice(["FCA", "CPT", "DAP", "CIF"], p=[0.4, 0.35, 0.15, 0.10])
                    })

                    # Валютно-банковская транзакция (поступление оплаты)
                    # Отсрочка платежа от 30 до 90 дней, возможные задержки из-за комплаенса
                    delay_days = int(np.random.choice([30, 45, 60, 90], p=[0.4, 0.3, 0.2, 0.1]))
                    compliance_lag = int(np.random.exponential(10)) if (is_post_2022 and contract_currency != "RUB") else 0
                    actual_payment_date = ship_date + timedelta(days=delay_days + compliance_lag)
                    is_paid = actual_payment_date <= self.end_dt

                    bank_transactions.append({
                        "deal_passport_id": f"DP-{decl_id[-10:]}",
                        "decl_id": decl_id,
                        "enterprise_id": ent_id,
                        "shipment_date": ship_date,
                        "expected_payment_date": ship_date + timedelta(days=delay_days),
                        "actual_payment_date": actual_payment_date if is_paid else None,
                        "currency": contract_currency,
                        "amount_contract_curr": round(total_rub if contract_currency == "RUB" else (total_usd if contract_currency == "USD" else total_usd * 7.1), 2),
                        "amount_usd_equiv": round(total_usd, 2),
                        "payment_status": "PAID" if is_paid else ("OVERDUE" if (self.end_dt - ship_date).days > delay_days else "PENDING"),
                        "compliance_delay_days": compliance_lag
                    })

        return pd.DataFrame(declarations), pd.DataFrame(bank_transactions)
