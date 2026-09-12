def recall_at_k(expected_doc_ids: list[str], retrieved_doc_ids: list[str]) -> float:
    if not expected_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    return len(expected.intersection(retrieved_doc_ids)) / len(expected)


def reciprocal_rank(expected_doc_ids: list[str], retrieved_doc_ids: list[str]) -> float:
    expected = set(expected_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def answerable_accuracy(expected: bool, predicted: bool) -> float:
    return float(expected == predicted)


def citation_accuracy(expected_doc_ids: list[str], cited_doc_ids: list[str], expected_answerable: bool) -> float:
    if not expected_answerable:
        return float(not cited_doc_ids)
    if not expected_doc_ids:
        return 0.0
    return float(bool(set(expected_doc_ids).intersection(cited_doc_ids)))


def criteria_coverage(required_terms: list[list[str]], answer: str) -> float | None:
    if not required_terms:
        return None
    normalized = answer.casefold()
    matched = sum(any(term.casefold() in normalized for term in alternatives) for alternatives in required_terms)
    return matched / len(required_terms)
