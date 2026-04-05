"""
Проверка ответа POST /analyze по файлу tests/test_responses.json.

Красивый вывод в терминале (строки по каждому объявлению + сводка метрик).
Исходный mcId в метриках по микрокатегориям не учитывается.

Запуск (сервер должен быть поднят: uvicorn main:app --reload):
    python tests/validate.py

Юнит-тесты функций метрик:
    python tests/validate.py --unit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES_FILE = ROOT / "tests" / "test_responses.json"
API_URL = "http://127.0.0.1:8000/analyze"

_METRIC_OUT = sys.stderr


def _ensure_utf8_stdio() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            except (AttributeError, OSError, ValueError):
                pass


def load_json_file(path: Path) -> Any:
    """Читает JSON; при ошибке парсера убирает //-комментарии и пробует снова."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    lines: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("//"):
            continue
        if "//" in line:
            in_string = False
            escape = False
            cut = len(line)
            for i, ch in enumerate(line):
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string and line[i : i + 2] == "//":
                    cut = i
                    break
            line = line[:cut].rstrip()
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
    return json.loads(cleaned)


def default_expected() -> Dict[str, Any]:
    return {"detectedMcIds": [], "shouldSplit": False, "drafts": []}


def normalize_expected(raw: Optional[dict]) -> Dict[str, Any]:
    if not raw:
        return default_expected()
    return {
        "detectedMcIds": list(raw.get("detectedMcIds") or []),
        "shouldSplit": bool(raw.get("shouldSplit", False)),
        "drafts": list(raw.get("drafts") or []),
    }


def normalize_case(raw: dict) -> Dict[str, Any]:
    """Поддержка формата { input, expected } и плоского { itemId, ..., expected }."""
    if "input" in raw:
        inp = raw["input"]
        exp = normalize_expected(raw.get("expected"))
    else:
        d = dict(raw)
        exp = normalize_expected(d.pop("expected", None))
        inp = d
    return {"input": inp, "expected": exp}


def load_cases(path: Path) -> List[Dict[str, Any]]:
    data = load_json_file(path)
    if not isinstance(data, list):
        raise ValueError("Ожидается JSON-массив кейсов")
    return [normalize_case(item) for item in data]


def additional_mc_ids(original_mc_id: int, detected: Iterable[int]) -> Set[int]:
    return {x for x in detected if x != original_mc_id}


def micro_precision_recall_f1(
    pairs: Sequence[Tuple[Set[int], Set[int]]],
) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for truth, pred in pairs:
        tp += len(truth & pred)
        fp += len(pred - truth)
        fn += len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def micro_confusion_totals(
    pairs: Sequence[Tuple[Set[int], Set[int]]],
) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for truth, pred in pairs:
        tp += len(truth & pred)
        fp += len(pred - truth)
        fn += len(truth - pred)
    return tp, fp, fn


def should_split_accuracy(expected: Sequence[bool], predicted: Sequence[bool]) -> float:
    if len(expected) != len(predicted):
        raise ValueError("expected и predicted должны быть одинаковой длины")
    if not expected:
        return 1.0
    return sum(a == b for a, b in zip(expected, predicted)) / len(expected)


def build_pairs_from_cases(
    cases: Sequence[dict],
    responses: Sequence[dict],
) -> Tuple[List[Tuple[Set[int], Set[int]]], List[bool], List[bool]]:
    pairs: List[Tuple[Set[int], Set[int]]] = []
    exp_split: List[bool] = []
    pred_split: List[bool] = []
    for case, resp in zip(cases, responses):
        inp = case["input"]
        orig = inp["mcId"]
        exp_ids = case["expected"]["detectedMcIds"]
        truth = additional_mc_ids(orig, exp_ids)
        pred_ids = resp.get("detectedMcIds") or []
        pred = additional_mc_ids(orig, pred_ids)
        pairs.append((truth, pred))
        exp_split.append(bool(case["expected"]["shouldSplit"]))
        pred_split.append(bool(resp.get("shouldSplit")))
    return pairs, exp_split, pred_split


def binary_metrics_from_should_split(
    y_true: List[int], y_pred: List[int]
) -> Tuple[float, float, float]:
    """Precision / Recall / F1 для метки shouldSplit (0 и 1)."""
    if not y_true:
        return 1.0, 1.0, 1.0
    return (
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(recall_score(y_true, y_pred, zero_division=0)),
        float(f1_score(y_true, y_pred, zero_division=0)),
    )


def print_pretty_report(
    cases: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    *,
    out: Any = sys.stdout,
) -> None:
    """Вывод по образцу: строки ID + блок Precision / Recall / F1 (Точность / Полнота / Баланс)."""
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []

    for case, resp in zip(cases, responses):
        inp = case["input"]
        item_id = inp.get("itemId", "?")
        exp_s = 1 if case["expected"]["shouldSplit"] else 0
        got_s = 1 if resp.get("shouldSplit") else 0
        y_true_bin.append(exp_s)
        y_pred_bin.append(got_s)
        print(f"ID {item_id}: (Ждали: {exp_s}, Получили: {got_s})", file=out)

    print(file=out)

    p_b, r_b, f_b = binary_metrics_from_should_split(y_true_bin, y_pred_bin)
    print(f"Precision (Точность): {p_b:.2f}", file=out)
    print(f"Recall (Полнота): {r_b:.2f}", file=out)
    print(f"F1-Score (Баланс): {f_b:.2f}", file=out)

    pairs, exp_sp, pred_sp = build_pairs_from_cases(cases, responses)
    tp, fp, fn = micro_confusion_totals(pairs)
    p_m, r_m, f_m = micro_precision_recall_f1(pairs)
    acc = should_split_accuracy(exp_sp, pred_sp)

    print(file=out)
    print("Дополнительные микрокатегории (без исходного mcId):", file=out)
    print(f"  TP / FP / FN (сумма): {tp} / {fp} / {fn}", file=out)
    print(f"  Precision (micro): {p_m:.4f}", file=out)
    print(f"  Recall (micro): {r_m:.4f}", file=out)
    print(f"  F1-score (micro): {f_m:.4f}", file=out)
    print(f"  Accuracy (shouldSplit): {acc:.4f}", file=out)


def run_api_evaluation(cases_file: Path, api_url: str) -> int:
    try:
        cases = load_cases(cases_file)
    except FileNotFoundError:
        print(f"Файл не найден: {cases_file}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Ошибка разбора JSON: {e}", file=sys.stderr)
        return 1

    responses: List[dict] = []
    for case in cases:
        try:
            r = requests.post(api_url, json=case["input"], timeout=120)
        except requests.RequestException as e:
            print(f"Ошибка запроса itemId={case['input'].get('itemId')}: {e}", file=sys.stderr)
            return 1
        if r.status_code != 200:
            print(
                f"HTTP {r.status_code} для itemId={case['input'].get('itemId')}: {r.text[:500]}",
                file=sys.stderr,
            )
            return 1
        responses.append(r.json())

    print_pretty_report(cases, responses, out=sys.stdout)
    return 0


# --- Юнит-тесты (только с флагом --unit) --------------------------------------


class TestAdditionalMcIds(unittest.TestCase):
    def test_filters_original(self) -> None:
        self.assertEqual(additional_mc_ids(201, [201, 101, 102]), {101, 102})

    def test_empty(self) -> None:
        self.assertEqual(additional_mc_ids(201, [201]), set())


class TestMicroPrecisionRecallF1(unittest.TestCase):
    def test_perfect_two_labels(self) -> None:
        pairs = [({101, 102}, {101, 102})]
        p, r, f1 = micro_precision_recall_f1(pairs)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_one_wrong_one_missing(self) -> None:
        pairs = [({101, 102}, {101, 103})]
        p, r, f1 = micro_precision_recall_f1(pairs)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(f1, 0.5)


def _print_test_run_summary(result: unittest.TestResult) -> None:
    out = _METRIC_OUT
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(getattr(result, "skipped", []) or [])
    passed = result.testsRun - failures - errors - skipped
    print("\n" + "=" * 50, file=out)
    print("Сводка unit-тестов", file=out)
    print("=" * 50, file=out)
    print(f"  Всего: {result.testsRun}, пройдено: {passed}", file=out)
    if failures or errors:
        print(f"  Провалов: {failures}, ошибок: {errors}", file=out)
    print(
        "  Итог: "
        + ("OK" if result.wasSuccessful() else "FAILED"),
        file=out,
    )
    print("=" * 50 + "\n", file=out)


def run_unit_tests(unittest_argv: List[str]) -> int:
    verbosity = 0 if ("-q" in unittest_argv or "--quiet" in unittest_argv) else 2
    print("Unit-тесты метрик\n", file=_METRIC_OUT, flush=True)
    program = unittest.main(
        argv=[sys.argv[0]] + unittest_argv,
        exit=False,
        verbosity=verbosity,
    )
    result = program.result
    _print_test_run_summary(result)
    return 0 if result.wasSuccessful() else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка API по tests/test_responses.json или unit-тесты метрик."
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Запустить только юнит-тесты функций метрик",
    )
    parser.add_argument("--url", default=API_URL, help="URL POST /analyze")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_FILE, help="JSON с кейсами")
    args, rest = parser.parse_known_args()

    _ensure_utf8_stdio()

    if args.unit:
        sys.exit(run_unit_tests(rest))

    sys.exit(run_api_evaluation(args.cases, args.url))


if __name__ == "__main__":
    main()
