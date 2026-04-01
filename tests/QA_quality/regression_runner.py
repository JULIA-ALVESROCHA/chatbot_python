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

    for test in tests:
        prompt = test["prompt"]
        category = test.get("category", "uncategorized")

        try:
            response = run_chatbot(prompt)
        except Exception as e:
            response = f"[ERROR DURING CHATBOT EXECUTION] {e}"

        score, reason = evaluate_response(
            prompt=prompt,
            response=response,
            expected_behavior=test.get("expected_behavior"),
            forbidden_keywords=test.get("forbidden_keywords"),
        )

        results.append({
            "id": test.get("id"),
            "category": category,
            "prompt": prompt,
            "response": response,
            "score": score,
            "reason": reason,
        })

    # 🔬 Geração de relatório científico
    report = generate_report(results)
    save_reports(results, report)

    print(">>> REGRESSION TEST FINISHED <<<")


if __name__ == "__main__":
    main()
