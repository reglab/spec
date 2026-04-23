#!/usr/bin/env python3
"""
Benchmark: Evaluate SPEC on the Colorado UI adjudication dataset.

Usage:
    python run_benchmark.py                           # Run all questions with default model
    python run_benchmark.py --provider anthropic      # Use Claude
    python run_benchmark.py --provider openai          # Use GPT
    python run_benchmark.py --provider gemini          # Use Gemini
    python run_benchmark.py --questions 1-10           # Run specific questions
    python run_benchmark.py --questions 1,5,10         # Run selected questions
"""

import os
import sys
import argparse
import openpyxl
from pathlib import Path
from dotenv import load_dotenv

from spec.pipeline import SPECPipeline

load_dotenv()

BENCHMARK_FILE = "data/benchmark.xlsx"


def parse_questions(question_str: str, max_q: int) -> list:
    """Parse question selection string like '1-10' or '1,5,10'."""
    if not question_str:
        return list(range(1, max_q + 1))

    questions = []
    for part in question_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            questions.extend(range(int(start), int(end) + 1))
        else:
            questions.append(int(part))
    return questions


def load_benchmark(path: str) -> list:
    """Load benchmark questions from Excel."""
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb['Benchmark']

    questions = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        no, question, answer = row[0], row[1], row[2]
        if question and answer:
            questions.append({"no": no, "question": question, "answer": answer})

    return questions


def evaluate_answer(predicted: str, expected: str) -> bool:
    """Check if predicted answer matches expected."""
    predicted = predicted.lower().strip()
    expected = expected.lower().strip()

    if expected in ["eligible", "ineligible", "inconclusive", "yes", "no"]:
        return expected in predicted
    return expected in predicted


def main():
    parser = argparse.ArgumentParser(description="SPEC Benchmark Evaluation")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "gemini"])
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--questions", default=None, help="Question selection (e.g., '1-10' or '1,5,10')")
    parser.add_argument("--docs-dir", default="docs", help="Directory containing adjudication guides")
    args = parser.parse_args()

    # Load benchmark
    benchmark = load_benchmark(BENCHMARK_FILE)
    question_nums = parse_questions(args.questions, len(benchmark))

    print(f"SPEC Benchmark Evaluation")
    print(f"Provider: {args.provider}")
    print(f"Questions: {len(question_nums)}")
    print("=" * 60)

    # Initialize pipeline
    pipeline = SPECPipeline(
        llm_provider=args.provider,
        model=args.model,
        docs_dir=args.docs_dir,
    )

    correct, total = 0, 0
    correct_complete, total_complete = 0, 0
    correct_inconclusive, total_inconclusive = 0, 0

    for q in benchmark:
        if q["no"] not in question_nums:
            continue

        total += 1
        expected = q["answer"]
        is_inconclusive = expected.lower() == "inconclusive"

        print(f"\n[{q['no']}/{len(benchmark)}] {q['question'][:80]}...")

        try:
            result = pipeline.process_question(q["question"])
            predicted = result.get("determination", "ERROR")
        except Exception as e:
            predicted = f"ERROR: {e}"

        is_correct = evaluate_answer(predicted, expected)
        if is_correct:
            correct += 1

        if is_inconclusive:
            total_inconclusive += 1
            if is_correct:
                correct_inconclusive += 1
        else:
            total_complete += 1
            if is_correct:
                correct_complete += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  Expected: {expected} | Got: {predicted} | {status}")

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS")
    print(f"=" * 60)
    print(f"Overall:      {correct}/{total} ({correct/total*100:.1f}%)" if total else "No questions run")
    if total_complete:
        print(f"Complete:     {correct_complete}/{total_complete} ({correct_complete/total_complete*100:.1f}%)")
    if total_inconclusive:
        print(f"Inconclusive: {correct_inconclusive}/{total_inconclusive} ({correct_inconclusive/total_inconclusive*100:.1f}%)")


if __name__ == "__main__":
    main()
