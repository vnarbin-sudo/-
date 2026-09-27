"""
Интеллектуальный мультимодальный оптимизатор маршрутов для логистов (Dynamic Route Intelligence).
Учитывает:
1. Эффект масштаба партии (FTL/LTL для авто против FCL/вагонов для Ж/Д с фиксированными затратами бронирования).
2. Станционные сборы (THC) и стоимость автодоставки «последней мили» (Last Mile).
3. Стоимость простоя в очередях на границе (Demurrage cost $/сутки).
4. Особенности номенклатуры (негабаритная карьерная техника БЕЛАЗ, тракторы МТЗ, агрегаты Великого Камня).
5. Время подготовки и подачи подвижного состава (Lead time).
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.config import ENTERPRISES, LOGISTICS_NODES, DESTINATION_CITIES

@dataclass
class RouteOption:
    route_id: str
    route_name: str
    mode: str                  # "ROAD", "RAIL_EXPRESS", "MULTIMODAL_SEA", "MULTIMODAL_INSTC", "AIR_EXPRESS"
    origin: str
    destination: str
    via_nodes: List[str]
    base_transit_days: float
    queue_delay_days: float
    lead_time_days: float      # Время подачи и согласования подвижного состава
    total_days: float
    cost_per_unit_usd: float
    total_cost_usd: float
    base_freight_usd: float    # Чистый фрахт
    terminal_thc_usd: float    # Станционные крановые сборы (THC) и последняя миля
    demurrage_cost_usd: float  # Оплата простоя тягачей/вагонов в очередях
    risk_score: float          # 0 - 100
    co2_per_unit_kg: float
    ai_recommendation_label: str
    pros: List[str]
    cons: List[str]
    vehicles_needed: str       # например: "1 фура FTL" или "2 ж/д платформы"
    score: float = 0.0

class MultimodalRouteOptimizer:
    def __init__(self, current_queues: Optional[Dict[str, float]] = None):
        """
        current_queues: словарь {node_id: queue_hours}
        """
        self.current_queues = current_queues or {}

    def get_node_queue_delay(self, node_id: str) -> float:
        """Перевод часов очереди узла в дни."""
        hours = self.current_queues.get(node_id, 12.0)
        return hours / 24.0

    def calculate_routes(
        self,
        origin_id: str,
        destination_id: str,
        batch_units: int = 10,
        priority: str = "BALANCE"  # "COST", "TIME", "RISK", "BALANCE"
    ) -> List[RouteOption]:
        """
        Построение и экономическая оценка всех доступных маршрутов.
        """
        dest_info = DESTINATION_CITIES.get(destination_id, DESTINATION_CITIES["MOSCOW"])
        orig_info = ENTERPRISES.get(origin_id, ENTERPRISES["MTZ"])
        is_belaz = (origin_id == "BELAZ")
        is_components = (origin_id == "GREAT_STONE")
        
        # Вместимость одного транспортного средства в зависимости от типа техники:
        # Для тракторов (МТЗ/Амкодор): 2 шт на автофуру/трал, 3 шт на Ж/Д платформу/40ft
        # Для грузовиков (МАЗ): 1 шт на автовоз, 2 шт на платформу
        # Для компонентов Великого Камня: 12 двигателей на фуру, 24 на Ж/Д контейнер
        # Для БЕЛАЗ (тяжеловес): 1 самосвал требует 1 спец-трал с сопровождением или 3 Ж/Д платформы!
        if is_components:
            units_per_truck = 12
            units_per_rail_car = 24
        elif is_belaz:
            units_per_truck = 1      # 1 тяжелый спецтрал с разрешением
            units_per_rail_car = 0.34 # 1 самосвал занимает 3 платформы
        elif origin_id == "MAZ":
            units_per_truck = 1
            units_per_rail_car = 2
        else: # MTZ, AMKODOR
            units_per_truck = 2
            units_per_rail_car = 3

        trucks_count = math.ceil(batch_units / units_per_truck)
        rail_cars_count = math.ceil(batch_units / units_per_rail_car)

        routes: List[RouteOption] = []

        # =========================================================================
        # 1. НАПРАВЛЕНИЕ: МОСКВА (РФ / Центр)
        # =========================================================================
        if destination_id == "MOSCOW":
            # --- Вариант А: Автотранспорт (М1 Красная Горка) ---
            q_kg = self.get_node_queue_delay("KRASNAYA_GORKA")
            truck_charter_rate = 1400.0 if not is_belaz else 4500.0 # ставка за рейс фуры
            demurrage_per_day = 80.0 if not is_belaz else 200.0
            
            base_freight_road = trucks_count * truck_charter_rate
            demurrage_road = trucks_count * demurrage_per_day * q_kg
            total_road_cost = base_freight_road + demurrage_road
            road_days = 1.5 + q_kg + 0.5 # 0.5 дня подача машины

            routes.append(RouteOption(
                route_id="RU_ROAD_M1",
                route_name="Прямой автокоридор М1 (Редьки — Красная Горка)",
                mode="ROAD",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["KRASNAYA_GORKA"],
                base_transit_days=1.5,
                queue_delay_days=round(q_kg, 1),
                lead_time_days=0.5,
                total_days=round(road_days, 1),
                cost_per_unit_usd=round(total_road_cost / batch_units, 2),
                total_cost_usd=round(total_road_cost, 2),
                base_freight_usd=round(base_freight_road, 2),
                terminal_thc_usd=0.0, # Доставка Door-to-Door без перевалки
                demurrage_cost_usd=round(demurrage_road, 2),
                risk_score=10.0,
                co2_per_unit_kg=380 if not is_components else 65,
                vehicles_needed=f"{trucks_count} фур(ы) FTL" if not is_belaz else f"{trucks_count} спец-трал(а)",
                ai_recommendation_label="Оптимально для малых и средних партий: прямой выезд без перегрузок",
                pros=["Подача под погрузку за 12 часов", "Прямая доставка Door-to-Door до склада покупателя", "Нет крановых станционных сборов"],
                cons=["Зависимость от заторов на М1 в Подмосковье"]
            ))

            # --- Вариант Б: Ж/Д Экспресс через Великий Камень / Колядичи ---
            # Фиксированная стоимость бронирования платформы/контейнера + THC + Last Mile
            rail_platform_rate = 1200.0 if not is_belaz else 1800.0
            thc_and_last_mile = 550.0 * rail_cars_count # станционные сборы кранов + доставка с вокзала
            base_freight_rail = rail_cars_count * rail_platform_rate
            total_rail_cost = base_freight_rail + thc_and_last_mile
            rail_days = 2.0 + 2.0 # 2 дня в пути + 2 дня на подачу платформы и маневровые работы

            routes.append(RouteOption(
                route_id="RU_RAIL_GS",
                route_name="Контейнерный Ж/Д шаттл «Великий Камень — Москва»",
                mode="RAIL_EXPRESS",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["GREAT_STONE", "KOLYADICHI"],
                base_transit_days=2.0,
                queue_delay_days=0.2,
                lead_time_days=2.0,
                total_days=round(rail_days, 1),
                cost_per_unit_usd=round(total_rail_cost / batch_units, 2),
                total_cost_usd=round(total_rail_cost, 2),
                base_freight_usd=round(base_freight_rail, 2),
                terminal_thc_usd=round(thc_and_last_mile, 2),
                demurrage_cost_usd=0.0,
                risk_score=8.0,
                co2_per_unit_kg=160 if not is_components else 25,
                vehicles_needed=f"{rail_cars_count} вагон(ов) / платформ",
                ai_recommendation_label="Выгодно при крупных партиях (от 10 ед.): экономия на масштабе Ж/Д",
                pros=["Фиксированный график движения", "Низкий углеродный след (-55% CO2)", "Безопасность и отсутствие очередей на границе"],
                cons=["Минимальный порог оплаты за вагон/платформу", "Требуется оплата автодоставки с товарной станции (Last Mile)"]
            ))

        # =========================================================================
        # 2. НАПРАВЛЕНИЕ: АЛМАТЫ / ТАШКЕНТ (Центральная Азия)
        # =========================================================================
        elif destination_id in ["ALMATY", "TASHKENT"]:
            q_kg = self.get_node_queue_delay("KRASNAYA_GORKA")
            q_gs = self.get_node_queue_delay("GREAT_STONE")

            # --- Вариант А: Автотранспорт ---
            truck_rate_ca = 5200.0 if not is_belaz else 14000.0
            base_freight_road = trucks_count * truck_rate_ca
            demurrage_road = trucks_count * 100.0 * (q_kg + 1.5) # задержка на границах
            total_road_cost = base_freight_road + demurrage_road
            road_days = 8.5 + q_kg + 1.5 + 1.0

            routes.append(RouteOption(
                route_id="CA_ROAD_DIRECT",
                route_name="Автоколонна через трассу М1/М5 и Казахстан",
                mode="ROAD",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["KRASNAYA_GORKA"],
                base_transit_days=8.5,
                queue_delay_days=round(q_kg + 1.5, 1),
                lead_time_days=1.0,
                total_days=round(road_days, 1),
                cost_per_unit_usd=round(total_road_cost / batch_units, 2),
                total_cost_usd=round(total_road_cost, 2),
                base_freight_usd=round(base_freight_road, 2),
                terminal_thc_usd=0.0,
                demurrage_cost_usd=round(demurrage_road, 2),
                risk_score=36.0,
                co2_per_unit_kg=1850 if not is_components else 220,
                vehicles_needed=f"{trucks_count} фур(ы) FTL",
                ai_recommendation_label="Оптимально для срочных партий 1-3 ед. без ожидания Ж/Д состава",
                pros=["Прямая разгрузка в региональных дилерских центрах", "Быстрый выезд без накопления контейнерного поезда"],
                cons=["Высокая себестоимость на больших партиях", "Двойной досмотр на границах РФ и РК"]
            ))

            # --- Вариант Б: Ускоренный Ж/Д контейнерный поезд «Великий Камень» ---
            rail_rate_ca = 3800.0 if not is_belaz else 5200.0
            thc_and_last_mile = 850.0 * rail_cars_count
            base_freight_rail = rail_cars_count * rail_rate_ca
            total_rail_cost = base_freight_rail + thc_and_last_mile
            rail_days = 6.0 + q_gs + 2.5 # 6 дней ход + 2.5 дня оформление и формирование

            routes.append(RouteOption(
                route_id="CA_RAIL_GS",
                route_name="Ускоренный контейнерный поезд «Великий Камень — Центр. Азия»",
                mode="RAIL_EXPRESS",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["GREAT_STONE", "KOLYADICHI"],
                base_transit_days=6.0,
                queue_delay_days=round(q_gs, 1),
                lead_time_days=2.5,
                total_days=round(rail_days, 1),
                cost_per_unit_usd=round(total_rail_cost / batch_units, 2),
                total_cost_usd=round(total_rail_cost, 2),
                base_freight_usd=round(base_freight_rail, 2),
                terminal_thc_usd=round(thc_and_last_mile, 2),
                demurrage_cost_usd=0.0,
                risk_score=15.0,
                co2_per_unit_kg=620 if not is_components else 85,
                vehicles_needed=f"{rail_cars_count} Ж/Д платформ(ы)",
                ai_recommendation_label="Выбор для партий от 4+ единиц: дешевле и надежнее",
                pros=["Опломбирование в бондовой зоне «Великого Камня»", "Экономия до 30% на крупных партиях", "Защита техники от дорожных повреждений"],
                cons=["Минимальная плата за вагон: для 1 шт нерентабельно"]
            ))

        # =========================================================================
        # 3. НАПРАВЛЕНИЕ: ШАНХАЙ / КИТАЙ
        # =========================================================================
        elif destination_id == "SHANGHAI":
            q_dostyk = self.get_node_queue_delay("DOSTYK_ALASHANKOU")
            q_bronka = self.get_node_queue_delay("PORT_BRONKA")
            q_m2 = self.get_node_queue_delay("MINSK2_AIR_CARGO")

            # Вариант 1: Ж/Д Экспресс через Достык
            rail_rate_cn = 5800.0 if not is_belaz else 8500.0
            thc_last_mile = 950.0 * rail_cars_count
            base_freight_rail = rail_cars_count * rail_rate_cn
            total_rail_cost = base_freight_rail + thc_last_mile

            routes.append(RouteOption(
                route_id="CN_RAIL_DOSTYK",
                route_name="Контейнерный Ж/Д экспресс «Великий Камень — Достык — Шанхай»",
                mode="RAIL_EXPRESS",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["GREAT_STONE", "DOSTYK_ALASHANKOU"],
                base_transit_days=14.0,
                queue_delay_days=round(q_dostyk, 1),
                lead_time_days=3.0,
                total_days=round(17.0 + q_dostyk, 1),
                cost_per_unit_usd=round(total_rail_cost / batch_units, 2),
                total_cost_usd=round(total_rail_cost, 2),
                base_freight_usd=round(base_freight_rail, 2),
                terminal_thc_usd=round(thc_last_mile, 2),
                demurrage_cost_usd=0.0,
                risk_score=22.0,
                co2_per_unit_kg=1100,
                vehicles_needed=f"{rail_cars_count} 40ft конт. / платформ",
                ai_recommendation_label="Оптимально по срокам: в 2.5 раза быстрее морского пути",
                pros=["Срок доставки 15-18 дней вместо 45 дней морем", "Сухопутный транзит по единой накладной ЦИМ/СМГС"],
                cons=["Возможная смена тележек на границе KZ-CN"]
            ))

            # Вариант 2: Мультимодальный (Ж/Д до Бронки + Морской фрахт)
            sea_container_rate = 3400.0 if not is_belaz else 6000.0
            port_thc = 1200.0 * rail_cars_count
            base_freight_sea = rail_cars_count * sea_container_rate
            total_sea_cost = base_freight_sea + port_thc

            routes.append(RouteOption(
                route_id="CN_SEA_BRONKA",
                route_name="Мультимодальный (Ж/Д до порта Бронка / СПб + Deep Sea в Шанхай)",
                mode="MULTIMODAL_SEA",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["PORT_BRONKA"],
                base_transit_days=38.0,
                queue_delay_days=round(q_bronka + 2.0, 1),
                lead_time_days=4.0,
                total_days=round(44.0 + q_bronka, 1),
                cost_per_unit_usd=round(total_sea_cost / batch_units, 2),
                total_cost_usd=round(total_sea_cost, 2),
                base_freight_usd=round(base_freight_sea, 2),
                terminal_thc_usd=round(port_thc, 2),
                demurrage_cost_usd=0.0,
                risk_score=34.0,
                co2_per_unit_kg=880,
                vehicles_needed=f"{rail_cars_count} морских слота(ов)",
                ai_recommendation_label="Минимальная стоимость для тяжеловесных партий без ограничения по габаритам",
                pros=["Самая низкая стоимость фрахта за тонну на дальнем плече", "Идеально для сверхтяжелых самосвалов БЕЛАЗ"],
                cons=["Длительное плечо доставки (>40 дней)", "Волатильность судовых расписаний"]
            ))

            # Вариант 3: Авиакарго экспресс из Великого Камня (Минск-2)
            if is_components or batch_units <= 2:
                air_cost_per_u = 4800.0 if is_components else 22000.0
                total_air_cost = air_cost_per_u * batch_units + 800.0 # сбор оформления AWB
                routes.append(RouteOption(
                    route_id="CN_AIR_MINSK2",
                    route_name="Авиагрузовой экспресс (Минск-2 / Великий Камень — Шанхай)",
                    mode="AIR_EXPRESS",
                    origin=orig_info["name"],
                    destination=dest_info["name"],
                    via_nodes=["MINSK2_AIR_CARGO"],
                    base_transit_days=2.0,
                    queue_delay_days=round(q_m2, 1),
                    lead_time_days=0.5,
                    total_days=round(2.5 + q_m2, 1),
                    cost_per_unit_usd=round(total_air_cost / batch_units, 2),
                    total_cost_usd=round(total_air_cost, 2),
                    base_freight_usd=round(air_cost_per_u * batch_units, 2),
                    terminal_thc_usd=800.0,
                    demurrage_cost_usd=0.0,
                    risk_score=10.0,
                    co2_per_unit_kg=3500,
                    vehicles_needed=f"Грузовой борт Ил-76 / B747F",
                    ai_recommendation_label="Экстренная экспресс-доставка (2 дня): только для срочных запчастей/электроники",
                    pros=["Сверхбыстрая доставка за 48 часов", "Максимальная сохранность и страховая защита"],
                    cons=["Критически высокий авиатариф"]
                ))

        # =========================================================================
        # 4. НАПРАВЛЕНИЕ: МУМБАИ / ИНДИЯ (МТК «Север-Юг»)
        # =========================================================================
        elif destination_id == "MUMBAI":
            q_olya = self.get_node_queue_delay("ASTRAKHAN_OLYA")
            q_novo = self.get_node_queue_delay("PORT_NOVOROSSIYSK")

            # Коридор «Север-Юг» через Каспий
            instc_rate = 4400.0 * rail_cars_count
            instc_thc = 1100.0 * rail_cars_count
            total_instc = instc_rate + instc_thc

            routes.append(RouteOption(
                route_id="IN_INSTC_CASPIAN",
                route_name="МТК «Север — Юг» (Великий Камень — Порт Оля/Астрахань — Каспий — Мумбаи)",
                mode="MULTIMODAL_INSTC",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["GREAT_STONE", "ASTRAKHAN_OLYA"],
                base_transit_days=18.0,
                queue_delay_days=round(q_olya + 1.5, 1),
                lead_time_days=3.0,
                total_days=round(22.5 + q_olya, 1),
                cost_per_unit_usd=round(total_instc / batch_units, 2),
                total_cost_usd=round(total_instc, 2),
                base_freight_usd=round(instc_rate, 2),
                terminal_thc_usd=round(instc_thc, 2),
                demurrage_cost_usd=0.0,
                risk_score=24.0,
                co2_per_unit_kg=1050,
                vehicles_needed=f"{rail_cars_count} конт. платформы",
                ai_recommendation_label="Стратегический коридор: прямой выход в Индию без риска санкционных задержек в ЕС",
                pros=["Выигрыш 14 дней по сравнению с маршрутом через Суэцкий канал", "Прямое соглашение о транзите с РФ и Ираном"],
                cons=["Каспийская перевалка на суда река-море"]
            ))

            # Через Черное море и Новороссийск
            sea_novo_rate = 3900.0 * rail_cars_count
            sea_novo_thc = 1200.0 * rail_cars_count
            total_novo = sea_novo_rate + sea_novo_thc

            routes.append(RouteOption(
                route_id="IN_SEA_NOVO",
                route_name="Мультимодальный (Ж/Д до Новороссийска + Суэц в Мумбаи)",
                mode="MULTIMODAL_SEA",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["PORT_NOVOROSSIYSK"],
                base_transit_days=26.0,
                queue_delay_days=round(q_novo + 2.0, 1),
                lead_time_days=3.0,
                total_days=round(31.0 + q_novo, 1),
                cost_per_unit_usd=round(total_novo / batch_units, 2),
                total_cost_usd=round(total_novo, 2),
                base_freight_usd=round(sea_novo_rate, 2),
                terminal_thc_usd=round(sea_novo_thc, 2),
                demurrage_cost_usd=0.0,
                risk_score=40.0,
                co2_per_unit_kg=1300,
                vehicles_needed=f"{rail_cars_count} морских слота",
                ai_recommendation_label="Альтернативный морской путь через Новороссийск",
                pros=["Регулярные контейнерные линии Черного моря"],
                cons=["Риски прохода через Босфор и Суэцкий канал (+10 дней)"]
            ))

        # =========================================================================
        # 5. ДРУГИЕ НАПРАВЛЕНИЯ (Универсальный расчет)
        # =========================================================================
        else:
            q_kg = self.get_node_queue_delay("KRASNAYA_GORKA")
            q_br = self.get_node_queue_delay("PORT_BRONKA")

            # Автовариант
            truck_rate = 2600.0 * trucks_count
            demurrage = trucks_count * 80.0 * q_kg
            total_road = truck_rate + demurrage
            routes.append(RouteOption(
                route_id="GEN_ROAD",
                route_name=f"Прямой автомаршрут до склада в {dest_info['name']}",
                mode="ROAD",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["KRASNAYA_GORKA"],
                base_transit_days=4.0,
                queue_delay_days=round(q_kg, 1),
                lead_time_days=0.5,
                total_days=round(4.5 + q_kg, 1),
                cost_per_unit_usd=round(total_road / batch_units, 2),
                total_cost_usd=round(total_road, 2),
                base_freight_usd=round(truck_rate, 2),
                terminal_thc_usd=0.0,
                demurrage_cost_usd=round(demurrage, 2),
                risk_score=18.0,
                co2_per_unit_kg=650,
                vehicles_needed=f"{trucks_count} фур(ы)",
                ai_recommendation_label="Оптимально для малых партий (1-3 ед.): подача без задержек",
                pros=["Быстрая подача", "Door-to-Door"],
                cons=["Очереди на трассе"]
            ))

            # ЖД / Мультимодал
            rail_rate = 1800.0 * rail_cars_count
            thc = 600.0 * rail_cars_count
            total_rail = rail_rate + thc
            routes.append(RouteOption(
                route_id="GEN_RAIL",
                route_name=f"Контейнерный Ж/Д маршрут через хаб «Великий Камень» в {dest_info['name']}",
                mode="RAIL_EXPRESS",
                origin=orig_info["name"],
                destination=dest_info["name"],
                via_nodes=["GREAT_STONE"],
                base_transit_days=5.5,
                queue_delay_days=0.3,
                lead_time_days=2.0,
                total_days=7.8,
                cost_per_unit_usd=round(total_rail / batch_units, 2),
                total_cost_usd=round(total_rail, 2),
                base_freight_usd=round(rail_rate, 2),
                terminal_thc_usd=round(thc, 2),
                demurrage_cost_usd=0.0,
                risk_score=12.0,
                co2_per_unit_kg=280,
                vehicles_needed=f"{rail_cars_count} платформ(ы)",
                ai_recommendation_label="Экономично при объемах от 5+ единиц",
                pros=["Низкая стоимость", "Надежность"],
                cons=["Плата за целый вагон"]
            ))

        # =========================================================================
        # КОМПЛЕКСНЫЙ МНОГОКРИТЕРИАЛЬНЫЙ СКОРИНГ МАРШРУТОВ
        # =========================================================================
        costs = np.array([r.total_cost_usd for r in routes])
        days = np.array([r.total_days for r in routes])
        risks = np.array([r.risk_score for r in routes])

        norm_cost = costs / (np.max(costs) + 1e-5)
        norm_days = days / (np.max(days) + 1e-5)
        norm_risk = risks / (np.max(risks) + 1e-5)

        if priority == "COST":
            w_c, w_t, w_r = 0.65, 0.20, 0.15
        elif priority == "TIME":
            w_c, w_t, w_r = 0.15, 0.70, 0.15
        elif priority == "RISK":
            w_c, w_t, w_r = 0.15, 0.20, 0.65
        else: # BALANCE
            w_c, w_t, w_r = 0.45, 0.35, 0.20

        # Корректировка: если партия маленькая (<= 2 ед.) и разница во времени велика,
        # логистическая гибкость автотранспорта получает дополнительный бонус
        scores = w_c * norm_cost + w_t * norm_days + w_r * norm_risk

        for idx, r in enumerate(routes):
            # Бонус прямому авто для малых партий
            if batch_units <= 2 and r.mode == "ROAD":
                scores[idx] -= 0.12 # бонус гибкости
            # Штраф авто для огромных партий
            elif batch_units >= 15 and r.mode == "ROAD":
                scores[idx] += 0.20 # штраф за необходимость фрахтовать 15 отдельных фур
            r.score = round(float(scores[idx]), 3)

        # Сортировка: наименьший score = 1 место
        routes.sort(key=lambda x: x.score)

        # Обновляем AI-лейбл для победителя
        if routes:
            win = routes[0]
            if win.mode == "ROAD" and batch_units <= 3:
                win.ai_recommendation_label = f"🏆 ВЫБОР ИИ: Для партии {batch_units} ед. автотранспорт (FTL) дешевле и быстрее: нет платы за бронирование вагона и перевалку"
            elif win.mode == "RAIL_EXPRESS":
                win.ai_recommendation_label = f"🏆 ВЫБОР ИИ: Для партии {batch_units} ед. контейнерный поезд дает максимальную экономию на масштабе и стабильный срок"
            elif win.mode == "MULTIMODAL_SEA":
                win.ai_recommendation_label = f"🏆 ВЫБОР ИИ: Оптимальная стоимость за тонну для тяжеловесных трансконтинентальных поставок"

        return routes
