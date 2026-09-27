"""
Централизованная конфигурация цифрового двойника экспорта Республики Беларусь (Машиностроение).
Включает расширенные евразийские коридоры и кластер «Великий Камень».
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# Базовые пути
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "export_twin.duckdb"

# Создание каталогов при необходимости
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Предприятия и кластеры продукции
ENTERPRISES = {
    "MTZ": {
        "name": "ОАО «МТЗ» (Минский тракторный завод)",
        "hs_code": "8701",
        "product": "Тракторы колесные (Беларус)",
        "base_monthly_volume": 3200,  # шт
        "avg_unit_price_usd": 24000,
        "key_markets": ["РФ", "Казахстан", "Узбекистан", "Пакистан", "Вьетнам", "Египет"],
        "critical_materials": ["steel_hrc", "rubber", "electronics"]
    },
    "BELAZ": {
        "name": "ОАО «БЕЛАЗ» (Карьерная техника)",
        "hs_code": "8704_HEAVY",
        "product": "Карьерные самосвалы (45-360 тонн)",
        "base_monthly_volume": 65,  # шт
        "avg_unit_price_usd": 950000,
        "key_markets": ["РФ", "Казахстан", "Узбекистан", "Китай", "Индия", "ЮАР", "Зимбабве"],
        "critical_materials": ["steel_alloy", "heavy_tires", "hydraulics"]
    },
    "MAZ": {
        "name": "ОАО «МАЗ» (Минский автомобильный завод)",
        "hs_code": "8704",
        "product": "Грузовые автомобили, тягачи, спецшасси",
        "base_monthly_volume": 900,  # шт
        "avg_unit_price_usd": 75000,
        "key_markets": ["РФ", "Казахстан", "Азербайджан", "Грузия", "Монголия"],
        "critical_materials": ["steel_hrc", "diesel_engines", "electronics"]
    },
    "AMKODOR": {
        "name": "ОАО «Амкодор»",
        "hs_code": "8429",
        "product": "Погрузчики, лесозаготовительная и дорожная техника",
        "base_monthly_volume": 420,  # шт
        "avg_unit_price_usd": 85000,
        "key_markets": ["РФ", "Казахстан", "Узбекистан", "Армения"],
        "critical_materials": ["steel_hrc", "hydraulics", "engines"]
    },
    "GREAT_STONE": {
        "name": "Индустриальный парк «Великий Камень» (Кластер автокомпонентов)",
        "hs_code": "8408_8708",
        "product": "Двигатели Weichai, КПП Fast Gear, мехатроника и суперконденсаторы",
        "base_monthly_volume": 1200,  # шт
        "avg_unit_price_usd": 18000,
        "key_markets": ["РФ", "Казахстан", "Китай", "Узбекистан"],
        "critical_materials": ["steel_alloy", "electronics", "aluminum"]
    }
}

# Ключевые логистические узлы и евразийские коридоры
LOGISTICS_NODES = {
    # Западные погранпереходы
    "KOZLOVICHI": {"name": "ПП Козловичи (BY-PL)", "type": "checkpoint", "lat": 52.128, "lon": 23.585, "capacity_trucks_day": 1200},
    "KAMENNY_LOG": {"name": "ПП Каменный Лог (BY-LT)", "type": "checkpoint", "lat": 54.542, "lon": 25.955, "capacity_trucks_day": 600},
    "GRIGOROVSHCHINA": {"name": "ПП Григоровщина (BY-LV)", "type": "checkpoint", "lat": 55.828, "lon": 27.915, "capacity_trucks_day": 300},
    
    # Восточный автокоридор (М1)
    "KRASNAYA_GORKA": {"name": "Трасса М1 Редьки/Красная Горка (BY-RU)", "type": "checkpoint_open", "lat": 54.693, "lon": 30.985, "capacity_trucks_day": 4000},
    
    # Белорусские внутренние мультимодальные хабы
    "KOLYADICHI": {"name": "ТЛЦ Колядичи (Ж/Д контейнерный терминал)", "type": "rail_hub", "lat": 53.805, "lon": 27.568, "capacity_teu_day": 1500},
    "GREAT_STONE": {"name": "Мультимодальный бондовый хаб «Великий Камень» (Сухой порт)", "type": "multimodal_hub", "lat": 53.905, "lon": 27.978, "capacity_teu_day": 2000},
    "MINSK2_AIR_CARGO": {"name": "Авиагрузовой терминал Минск-2 / Великий Камень", "type": "air_cargo", "lat": 53.882, "lon": 28.030, "capacity_teu_day": 400},
    
    # Морские порты РФ
    "PORT_BRONKA": {"name": "Морской порт Бронка / СПб (Балтика)", "type": "seaport", "lat": 59.932, "lon": 29.702, "capacity_teu_day": 2500},
    "PORT_NOVOROSSIYSK": {"name": "Порт Новороссийск (Черное море)", "type": "seaport", "lat": 44.724, "lon": 37.768, "capacity_teu_day": 1800},
    "PORT_VLADIVOSTOK": {"name": "Порт Владивосток (Тихий океан / Азия)", "type": "seaport", "lat": 43.105, "lon": 131.874, "capacity_teu_day": 3000},
    
    # Евразийские трансграничные шлюзы
    "DOSTYK_ALASHANKOU": {"name": "Ж/Д шлюз Достык — Алашанькоу (KZ-CN)", "type": "rail_border_hub", "lat": 45.253, "lon": 82.482, "capacity_teu_day": 3500},
    "ASTRAKHAN_OLYA": {"name": "Порт Оля / Астрахань (МТК «Север — Юг» / Каспий)", "type": "caspian_port", "lat": 45.783, "lon": 47.550, "capacity_teu_day": 1200},
    "ZABAIKALSK": {"name": "ПП Забайкальск — Маньчжурия (RU-CN)", "type": "rail_border_hub", "lat": 49.650, "lon": 117.320, "capacity_teu_day": 2800}
}

# Ключевые внешние целевые рынки и центры потребления
DESTINATION_CITIES = {
    "MOSCOW": {"name": "Москва (РФ / Центр)", "lat": 55.755, "lon": 37.617, "country": "РФ", "distance_km": 720},
    "YEKATERINBURG": {"name": "Екатеринбург (РФ / Урал)", "lat": 56.838, "lon": 60.597, "country": "РФ", "distance_km": 2500},
    "ALMATY": {"name": "Алматы (Казахстан)", "lat": 43.238, "lon": 76.945, "country": "Казахстан", "distance_km": 4400},
    "TASHKENT": {"name": "Ташкент (Узбекистан)", "lat": 41.311, "lon": 69.279, "country": "Узбекистан", "distance_km": 4200},
    "SHANGHAI": {"name": "Шанхай (Китай)", "lat": 31.230, "lon": 121.473, "country": "Китай", "distance_km": 8300},
    "MUMBAI": {"name": "Мумбаи (Индия / МТК «Север — Юг»)", "lat": 19.076, "lon": 72.877, "country": "Индия", "distance_km": 6200},
    "ALEXANDRIA": {"name": "Александрия / Каир (Египет)", "lat": 31.200, "lon": 29.918, "country": "Египет", "distance_km": 3600}
}

# Временной диапазон исторической генерации
START_DATE = "2021-01-01"
END_DATE = "2026-06-30"
FORECAST_HORIZON_MONTHS = 12
