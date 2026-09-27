"""
Главный запускающий скрипт полного цикла цифрового двойника экспорта Республики Беларусь.
"""
import sys
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"

def run_all():
    print("=" * 70)
    print("🚜 ЦИФРОВОЙ ДВОЙНИК ЭКСПОРТА РБ (МАШИНОСТРОЕНИЕ) — ЗАПУСК СИСТЕМЫ")
    print("=" * 70)

    # 1. Ingestion
    print("\n>>> ЭТАП 1: Уровень сбора и интеграции данных (Data Ingestion Layer)...")
    res1 = subprocess.run([str(PYTHON_EXE), "-m", "src.ingestion.pipeline"], cwd=str(BASE_DIR))
    if res1.returncode != 0:
        print("Ошибка на этапе Ingestion!")
        return

    # 2. Predictive
    print("\n>>> ЭТАП 2: Аналитическое предиктивное ядро (Predictive Core Layer)...")
    res2 = subprocess.run([str(PYTHON_EXE), "-m", "src.predictive.trainer"], cwd=str(BASE_DIR))
    if res2.returncode != 0:
        print("Ошибка на этапе Predictive Core!")
        return

    # 3. Simulation
    print("\n>>> ЭТАП 3: Сценарный симулятор стресс-тестирования (Simulation Layer)...")
    res3 = subprocess.run([str(PYTHON_EXE), "-m", "src.simulation.impact_calculator"], cwd=str(BASE_DIR))
    if res3.returncode != 0:
        print("Ошибка на этапе Simulation!")
        return

    # 4. Tests
    print("\n>>> ВЕРИФИКАЦИЯ: Запуск сквозного набора тестов (pytest)...")
    res4 = subprocess.run([str(BASE_DIR / ".venv" / "Scripts" / "pytest.exe"), "tests/"], cwd=str(BASE_DIR))
    if res4.returncode != 0:
        print("Ошибка при прохождении тестов!")
        return

    print("\n" + "=" * 70)
    print(" Все 4 уровня цифрового двойника успешно развернуты и протестированы!")
    print(" Для запуска интерактивного дашборда выполните:")
    print(f'   & "{PYTHON_EXE}" -m streamlit run src/ui/dashboard.py')
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", action="store_true", help="Сразу запустить Streamlit веб-интерфейс")
    args = parser.parse_args()

    if args.ui:
        print(">>> Запуск интерактивного дашборда на Streamlit...")
        subprocess.run([str(PYTHON_EXE), "-m", "streamlit", "run", "src/ui/dashboard.py"], cwd=str(BASE_DIR))
    else:
        run_all()
