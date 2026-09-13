import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


parser = argparse.ArgumentParser(description="Compare two evaluation reports.")
parser.add_argument("baseline", type=Path)
parser.add_argument("improved", type=Path)
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "hybrid-comparison.md")
args = parser.parse_args()

baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
improved = json.loads(args.improved.read_text(encoding="utf-8"))
if baseline["run"]["gold_set"] != improved["run"]["gold_set"]:
    raise SystemExit("Reports use different Gold Sets")
if baseline["summary"]["question_count"] != improved["summary"]["question_count"]:
    raise SystemExit("Reports contain different question counts")

metrics = ["recall_at_k", "mrr", "answerable_accuracy", "citation_accuracy", "criteria_coverage"]
lines = [
    "# Dense baseline vs Hybrid RRF",
    "",
    "## Hypothesis",
    "",
    "Dense retrieval에 BM25 exact-term 신호를 약하게 결합하면 Kubernetes 고유 명칭이 포함된 질문의 순위가 개선된다.",
    "",
    "## Result",
    "",
    "| Metric | Baseline | Hybrid | Delta |",
    "|---|---:|---:|---:|",
    f"| error_count | {baseline['summary']['error_count']} | {improved['summary']['error_count']} | {improved['summary']['error_count'] - baseline['summary']['error_count']:+d} |",
]
for metric in metrics:
    before = baseline["summary"][metric]
    after = improved["summary"][metric]
    if before is None or after is None:
        lines.append(f"| {metric} | n/a | n/a | n/a |")
    else:
        lines.append(f"| {metric} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")

baseline_items = {item["id"]: item for item in baseline["items"]}
improved_items = {item["id"]: item for item in improved["items"]}
improved_ids = [item_id for item_id in baseline_items if (improved_items[item_id]["mrr"] or 0) > (baseline_items[item_id]["mrr"] or 0)]
regressed_ids = [item_id for item_id in baseline_items if (improved_items[item_id]["mrr"] or 0) < (baseline_items[item_id]["mrr"] or 0)]
lines.extend([
    "",
    "## Analysis",
    "",
    f"- MRR 개선 문항: {', '.join(improved_ids) or '없음'}",
    f"- MRR 하락 문항: {', '.join(regressed_ids) or '없음'}",
    "- 한국어 질문과 영문 코퍼스의 언어 불일치 때문에 BM25는 영문 Kubernetes 용어가 질문에 포함된 경우에만 기여한다.",
    "- 두 실행의 error_count가 다르면 오류 문항을 제외하는 생성 지표의 분모도 달라지므로 해당 수치는 직접 비교에 주의해야 한다.",
    "",
    "## Next Steps",
    "",
    "- 하락 문항에서 BM25 일치 토큰과 상위 청크를 확인한다.",
    "- 필요하면 query translation 또는 multilingual sparse retrieval을 별도 실험한다.",
    "- 전체 생성 평가 후 citation과 답변 수용 기준의 변화를 함께 비교한다.",
    "",
])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("\n".join(lines), encoding="utf-8")
print(args.output)
