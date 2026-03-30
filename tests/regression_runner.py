import json
import os
import time

from chatbot.run_chatbot import run_chatbot
from tests.evaluator import evaluate_response
from reports.reports import generate_report, save_reports

TEST_FILE = "tests/hallucination_tests.json"
CHECKPOINT_FILE = "tests/checkpoint.json"
SLEEP_BETWEEN_CALLS = 1.5  # seconds — avoids OpenAI rate limiting
MAX_RETRIES = 3


def load_checkpoint():
    """Load previously completed results to resume interrupted runs."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_checkpoint(results):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def evaluate_with_retry(prompt, response, expected_behavior, forbidden_keywords, gold_standard):
    """Call evaluator with retry logic on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return evaluate_response(
                prompt=prompt,
                response=response,
                expected_behavior=expected_behavior,
                forbidden_keywords=forbidden_keywords,
                gold_standard=gold_standard,
            )
        except KeyboardInterrupt:
            raise  # always let the user interrupt cleanly
        except Exception as e:
            print(f"  [Evaluator attempt {attempt}/{MAX_RETRIES} failed]: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)  # exponential backoff: 5s, 10s
            else:
                return 0, f"Evaluator failed after {MAX_RETRIES} attempts: {e}"


def main():
    print(">>> REGRESSION TEST STARTED <<<")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        tests = json.load(f)

    # Resume from checkpoint if available
    results = load_checkpoint()
    completed_ids = {r["id"] for r in results}

    if completed_ids:
        print(f"  Resuming from checkpoint — {len(completed_ids)} tests already done.")

    total = len(tests)

    try:
        for i, test in enumerate(tests, 1):
            test_id = test.get("id")
            category = test.get("category", "uncategorized")

            # Skip already completed tests
            if test_id in completed_ids:
                print(f"[{i}/{total}] {test_id} — skipped (already done)")
                continue

            prompt = test["prompt"]
            print(f"[{i}/{total}] {test_id} — {category}")

            try:
                response = run_chatbot(prompt)
            except Exception as e:
                response = f"[ERROR DURING CHATBOT EXECUTION] {e}"
                print(f"  ERROR in chatbot: {e}")

            score, reason = evaluate_with_retry(
                prompt=prompt,
                response=response,
                expected_behavior=test.get("expected_behavior"),
                forbidden_keywords=test.get("forbidden_keywords"),
                gold_standard=test.get("gold_standard"),
            )

            print(f"  score={score} | {reason}")

            results.append({
                "id": test_id,
                "category": category,
                "prompt": prompt,
                "response": response,
                "score": score,
                "reason": reason,
            })

            # Save progress after every test
            save_checkpoint(results)
            time.sleep(SLEEP_BETWEEN_CALLS)

    except KeyboardInterrupt:
        print("\n  [Interrupted by user — progress saved to checkpoint]")
        return

    # Clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    report = generate_report(results)
    save_reports(results, report)

    print(f"\n>>> REGRESSION TEST FINISHED <<<")
    print(f"    Hallucination rate : {report['summary']['hallucination_rate']*100:.1f}%")
    print(f"    Severe rate        : {report['summary']['severe_hallucination_rate']*100:.1f}%")
    print(f"    Total tests        : {report['summary']['total_tests']}")


if __name__ == "__main__":
    main()