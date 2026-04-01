import requests
import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Настройки
API_URL = "http://127.0.0.1:8000/analyze"
DATA_FILE = "tests/test_responses.json"  # Путь к твоему файлу


def run_auto_robot():
    y_true = []
    y_pred = []

    # БЛОК ЧТЕНИЯ: Открываем твой внешний JSON
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Файл {DATA_FILE} не найден!")
        return

    # БЛОК ЗАПРОСОВ: Имитируем Swagger для каждого примера
    for case in test_cases:
        try:
            response = requests.post(API_URL, json=case["input"])

            if response.status_code == 200:
                server_answer = response.json()

                # Сопоставляем результат (True -> 1, False -> 0)
                prediction = 1 if server_answer.get("make_avert") else 0
                actual_expected = case["expected"]

                y_true.append(actual_expected)
                y_pred.append(prediction)

                # Лог в консоль для наглядности
                icon = "✅" if prediction == actual_expected else "❌"
                print(f"ID {case['input']['itemId']}: {icon} (Ждали: {actual_expected}, Получили: {prediction})")
            else:
                print(f"Ошибка сервера на ID {case['input']['itemId']}: {response.status_code}")

        except Exception as e:
            print(f"Критическая ошибка при запросе: {e}")

    # БЛОК МАТЕМАТИКИ: sklearn считает итог
    if y_true:
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)


        print(f"\nPrecision (Точность): {p:.2f}")
        print(f"Recall (Полнота):    {r:.2f}")
        print(f"F1-Score (Баланс):   {f1:.2f}")


if __name__ == "__main__":
    run_auto_robot()