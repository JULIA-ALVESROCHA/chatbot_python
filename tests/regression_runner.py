import json
import os

from chatbot.run_chatbot import run_chatbot
from tests.evaluator import evaluate_response
from reports.reports import generate_report, save_reports

TEST_FILE = "tests/hallucination_tests.json"


def main():
    print(">>> REGRESSION TEST STARTED <<<")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        tests = json.load(f)

    results = []
    total = len(tests)

    for i, test in enumerate(tests, 1):
        prompt = test["prompt"]
        category = test.get("category", "uncategorized")

        print(f"[{i}/{total}] {test.get('id')} — {category}")

        try:
            response = run_chatbot(prompt)
        except Exception as e:
            response = f"[ERROR DURING CHATBOT EXECUTION] {e}"
            print(f"  ERROR: {e}")

        score, reason = evaluate_response(
            prompt=prompt,
            response=response,
            expected_behavior=test.get("expected_behavior"),
            forbidden_keywords=test.get("forbidden_keywords"),
            gold_standard=test.get("gold_standard"),
        )

        print(f"  score={score} | {reason}")

        results.append({
            "id": test.get("id"),
            "category": category,
            "prompt": prompt,
            "response": response,
            "score": score,
            "reason": reason,
        })

    report = generate_report(results)
    save_reports(results, report)

    print(f"\n>>> REGRESSION TEST FINISHED <<<")
    print(f"    Hallucination rate : {report['summary']['hallucination_rate']*100:.1f}%")
    print(f"    Severe rate        : {report['summary']['severe_hallucination_rate']*100:.1f}%")
    print(f"    Total tests        : {report['summary']['total_tests']}")


if __name__ == "__main__":
    main()