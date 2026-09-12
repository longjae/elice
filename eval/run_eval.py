import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from eval.metrics import answerable_accuracy, citation_accuracy, criteria_coverage, recall_at_k, reciprocal_rank
from rag.qa import answer_question, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Kubernetes RAG Gold Set.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "baseline.yaml")
    parser.add_argument("--output-name", default="baseline")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    missing_config = sorted({"top_k", "seed", "minimum_similarity", "retrieval_mode"} - config.keys())
    if missing_config:
        raise SystemExit(f"Missing config keys: {', '.join(missing_config)}")

    gold_path = PROJECT_ROOT / "data" / "gold_set.jsonl"
    gold_items = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line]
    if args.limit is not None:
        gold_items = gold_items[: args.limit]
    if len(gold_items) < 20 and args.limit is None:
        raise SystemExit("Gold Set must contain at least 20 questions")
    required_gold = {"id", "question", "question_type", "expected_doc_ids", "acceptance_criteria", "answerable"}
    for item in gold_items:
        missing_gold = sorted(required_gold - item.keys())
        if missing_gold:
            raise SystemExit(f"Gold item {item.get('id', '<unknown>')} is missing: {', '.join(missing_gold)}")

    rows: list[dict[str, object]] = []
    for position, item in enumerate(gold_items, start=1):
        try:
            if args.retrieval_only:
                retrieval = retrieve(item["question"], config_path=args.config)
                predicted_answerable = bool(retrieval and retrieval[0]["score"] >= config["minimum_similarity"])
                result = {"answer": "", "answerable": predicted_answerable, "citations": [], "retrieval": retrieval, "usage": {}}
            else:
                result = answer_question(item["question"], config_path=args.config)
            retrieved_doc_ids = [entry["doc_id"] for entry in result["retrieval"]]
            cited_doc_ids = [entry["doc_id"] for entry in result["citations"]]
            row = {
                "id": item["id"], "question": item["question"], "question_type": item["question_type"],
                "llm_model": result.get("model"),
                "expected_answerable": item["answerable"], "predicted_answerable": result["answerable"],
                "expected_doc_ids": item["expected_doc_ids"], "retrieved_doc_ids": retrieved_doc_ids,
                "cited_doc_ids": cited_doc_ids, "answer": result["answer"],
                "recall_at_k": recall_at_k(item["expected_doc_ids"], retrieved_doc_ids) if item["answerable"] else None,
                "mrr": reciprocal_rank(item["expected_doc_ids"], retrieved_doc_ids) if item["answerable"] else None,
                "answerable_accuracy": answerable_accuracy(item["answerable"], bool(result["answerable"])),
                "citation_accuracy": None if args.retrieval_only else citation_accuracy(item["expected_doc_ids"], cited_doc_ids, item["answerable"]),
                "criteria_coverage": None if args.retrieval_only else criteria_coverage(item.get("required_terms", []), str(result["answer"])),
                "retrieval": result["retrieval"], "usage": result["usage"], "error": None,
            }
        except Exception as error:
            error_message = str(error).splitlines()[0][:300]
            row = {"id": item["id"], "question": item["question"], "question_type": item["question_type"], "llm_model": None, "expected_answerable": item["answerable"], "predicted_answerable": None, "expected_doc_ids": item["expected_doc_ids"], "retrieved_doc_ids": [], "cited_doc_ids": [], "answer": "", "recall_at_k": None, "mrr": None, "answerable_accuracy": 0.0, "citation_accuracy": None, "criteria_coverage": None, "retrieval": [], "usage": {}, "error": f"{type(error).__name__}: {error_message}"}
        rows.append(row)
        print(f"[{position}/{len(gold_items)}] {item['id']} error={row['error'] is not None}")

    def average(field: str) -> float | None:
        values = [float(row[field]) for row in rows if row[field] is not None]
        return sum(values) / len(values) if values else None

    summary = {"question_count": len(rows), "error_count": sum(row["error"] is not None for row in rows), "recall_at_k": average("recall_at_k"), "mrr": average("mrr"), "answerable_accuracy": average("answerable_accuracy"), "citation_accuracy": average("citation_accuracy"), "criteria_coverage": average("criteria_coverage")}
    configured_llm = os.getenv("ELICE_LLM_MODEL", "gpt-5.6-luna")
    if configured_llm == "gpt-5.6-luna":
        configured_llm = "openai/gpt-5.6-luna"
    report = {"run": {"created_at": datetime.now(timezone.utc).isoformat(), "output_name": args.output_name, "retrieval_only": args.retrieval_only, "config_path": str(args.config), "config": config, "gold_set": str(gold_path), "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), "llm_model": configured_llm, "local_seed": config["seed"]}, "summary": summary, "items": rows}

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    json_path = reports_dir / f"{args.output_name}.json"
    csv_path = reports_dir / f"{args.output_name}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["id", "question_type", "recall_at_k", "mrr", "answerable_accuracy", "citation_accuracy", "criteria_coverage", "error"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"json={json_path} csv={csv_path}")
    if summary["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
