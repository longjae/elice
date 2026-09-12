import argparse
import json
from pathlib import Path

import httpx
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the reviewed Kubernetes corpus allowlist.")
    parser.add_argument("--force", action="store_true", help="Download files that already exist.")
    args = parser.parse_args()

    with (PROJECT_ROOT / "config" / "corpus.yaml").open(encoding="utf-8") as file:
        documents = yaml.safe_load(file)["documents"]

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failures: list[dict[str, str]] = []

    with httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": "elice-kubernetes-rag/0.1"}) as client:
        for document in documents:
            output = raw_dir / f"{document['doc_id']}.html"
            if output.exists() and not args.force:
                skipped += 1
                continue
            try:
                response = client.get(document["url"])
                response.raise_for_status()
                if response.url.host != "kubernetes.io":
                    raise ValueError(f"unexpected redirect host: {response.url.host}")
                output.write_text(response.text, encoding="utf-8")
                downloaded += 1
                print(f"downloaded {document['doc_id']}")
            except (httpx.HTTPError, ValueError) as error:
                failures.append({"doc_id": document["doc_id"], "error": str(error)})
                print(f"failed {document['doc_id']}: {error}")

    (raw_dir / "manifest.json").write_text(
        json.dumps({"document_count": len(documents), "downloaded": downloaded, "skipped": skipped, "failures": failures, "documents": documents}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"documents={len(documents)} downloaded={downloaded} skipped={skipped} failures={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
