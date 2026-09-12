from rag.ingest import extract_sections, split_text


def test_extract_sections_keeps_headings_and_text() -> None:
    html = "<html><main><h1>Pods</h1><p>First paragraph.</p><h2>Lifecycle</h2><p>Second paragraph.</p></main></html>"
    assert extract_sections(html, "Fallback") == [("Pods", "First paragraph."), ("Lifecycle", "Second paragraph.")]


def test_split_text_respects_size_and_overlap() -> None:
    chunks = split_text("one two three four five six", chunk_size=14, overlap=4)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 14 for chunk in chunks)
