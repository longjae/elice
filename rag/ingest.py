import json
import re
from pathlib import Path

import yaml
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_sections(html: str, fallback_title: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find("article")
    if main is None:
        raise ValueError("document has no main or article element")
    for unwanted in main.select("nav, script, style, form, .feedback--prompt, .td-page-meta"):
        unwanted.decompose()

    sections: list[tuple[str, str]] = []
    heading = fallback_title
    parts: list[str] = []
    for element in main.find_all(["h1", "h2", "h3", "p", "li", "pre"]):
        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if element.name in {"h1", "h2", "h3"}:
            if parts:
                sections.append((heading, "\n".join(parts)))
                parts = []
            heading = text
        else:
            parts.append(text)
    if parts:
        sections.append((heading, "\n".join(parts)))
    return sections


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def main() -> None:
    with (PROJECT_ROOT / "config" / "baseline.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "corpus.yaml").open(encoding="utf-8") as file:
        documents = yaml.safe_load(file)["documents"]

    chunks: list[dict[str, object]] = []
    missing: list[str] = []
    for document in documents:
        raw_path = PROJECT_ROOT / "data" / "raw" / f"{document['doc_id']}.html"
        if not raw_path.exists():
            missing.append(document["doc_id"])
            continue
        sections = extract_sections(raw_path.read_text(encoding="utf-8"), document["title"])
        chunk_number = 0
        for section, text in sections:
            for chunk_text in split_text(text, config["chunk_size"], config["chunk_overlap"]):
                chunks.append({"chunk_id": f"{document['doc_id']}__{chunk_number:04d}", "doc_id": document["doc_id"], "title": document["title"], "section": section, "text": chunk_text, "url": document["url"], "category": document["category"]})
                chunk_number += 1

    if missing:
        raise SystemExit(f"Missing raw documents: {', '.join(missing)}. Run scripts/download_corpus.py first.")
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"documents={len(documents)} chunks={len(chunks)} output={output_dir / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
