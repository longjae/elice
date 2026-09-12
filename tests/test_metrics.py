from eval.metrics import answerable_accuracy, citation_accuracy, criteria_coverage, recall_at_k, reciprocal_rank


def test_retrieval_metrics() -> None:
    assert recall_at_k(["a", "b"], ["x", "b", "c"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["x", "b", "a"]) == 0.5
    assert reciprocal_rank(["a"], ["x", "y"]) == 0.0


def test_generation_metrics() -> None:
    assert answerable_accuracy(True, True) == 1.0
    assert citation_accuracy(["probes"], ["probes"], True) == 1.0
    assert citation_accuracy([], [], False) == 1.0
    assert criteria_coverage([["readiness"], ["restart", "재시작"]], "Readiness는 재시작과 다르다") == 1.0
