# Dense baseline vs Hybrid RRF

## Hypothesis

Dense retrieval에 BM25 exact-term 신호를 약하게 결합하면 Kubernetes 고유 명칭이 포함된 질문의 순위가 개선된다.

## Result

| Metric | Baseline | Hybrid | Delta |
|---|---:|---:|---:|
| error_count | 1 | 0 | -1 |
| recall_at_k | 0.9487 | 0.9295 | -0.0192 |
| mrr | 0.8462 | 0.9038 | +0.0577 |
| answerable_accuracy | 0.9286 | 0.9643 | +0.0357 |
| citation_accuracy | 0.9259 | 0.9286 | +0.0026 |
| criteria_coverage | 0.8974 | 0.8859 | -0.0115 |

## Analysis

- MRR 개선 문항: q004, q005, q025
- MRR 하락 문항: 없음
- 한국어 질문과 영문 코퍼스의 언어 불일치 때문에 BM25는 영문 Kubernetes 용어가 질문에 포함된 경우에만 기여한다.
- 두 실행의 error_count가 다르면 오류 문항을 제외하는 생성 지표의 분모도 달라지므로 해당 수치는 직접 비교에 주의해야 한다.

## Next Steps

- 하락 문항에서 BM25 일치 토큰과 상위 청크를 확인한다.
- 필요하면 query translation 또는 multilingual sparse retrieval을 별도 실험한다.
- 전체 생성 평가 후 citation과 답변 수용 기준의 변화를 함께 비교한다.
