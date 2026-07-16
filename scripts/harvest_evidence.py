"""Harvest the evidence library from PubMed.

Every document in `data/evidence_library.json` is a real PubMed record: real PMID,
real title, real journal, real year, real abstract text. Nothing here is written by
an LLM. Re-run this script to rebuild or refresh the library.

Usage:
    python scripts/harvest_evidence.py [--out data/evidence_library.json]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# One query per formulation strategy. "free full text[filter]" restricts to records
# with an openly accessible full text, so a reader can always follow the citation.
TOPICS: list[tuple[str, str]] = [
    (
        "amorphous solid dispersion",
        "amorphous solid dispersion AND (poorly soluble OR bioavailability)",
    ),
    (
        "lipid-based formulation",
        "(lipid-based drug delivery OR self-emulsifying drug delivery system) AND oral bioavailability",
    ),
    ("salt formation", "salt formation AND (solubility OR dissolution) AND pharmaceutical"),
    ("cocrystal", "pharmaceutical cocrystal AND (solubility OR dissolution)"),
    ("cyclodextrin", "cyclodextrin complexation AND (solubility OR bioavailability) AND drug"),
    ("nanosuspension", "nanosuspension AND (poorly water soluble OR dissolution)"),
    (
        "particle size reduction",
        "(micronization OR nanocrystal OR particle size reduction) AND dissolution rate AND drug",
    ),
    (
        "supersaturating formulation",
        "supersaturating drug delivery system AND (precipitation OR bioavailability)",
    ),
    (
        "precipitation inhibition",
        "precipitation inhibitor AND (polymer OR supersaturation) AND oral absorption",
    ),
    (
        "poorly soluble oral drugs",
        "(BCS class II OR BCS class IV) AND (oral bioavailability OR solubility enhancement)",
    ),
]

PER_TOPIC = 5
MIN_YEAR = 2010
MIN_ABSTRACT_CHARS = 400


@dataclass
class Document:
    id: str
    title: str
    source: str
    year: int
    url: str
    doi: str
    pmid: str
    text: str
    tags: list[str]


def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - network flakiness, retry
            if attempt == 3:
                raise
            print(f"    retry {attempt + 1} after {exc}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def esearch(query: str, retmax: int) -> list[str]:
    params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": f"{query} AND free full text[filter] AND {MIN_YEAR}:3000[dp]",
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
        }
    )
    data = json.loads(_get(f"{EUTILS}/esearch.fcgi?{params}"))
    return data["esearchresult"]["idlist"]


def _abstract_text(article: ET.Element) -> str:
    """Join structured abstract sections, preserving their labels."""
    parts: list[str] = []
    for node in article.iter("AbstractText"):
        chunk = "".join(node.itertext()).strip()
        if not chunk:
            continue
        label = node.get("Label")
        parts.append(f"{label.strip().title()}: {chunk}" if label else chunk)
    return " ".join(parts).strip()


def _year(article: ET.Element) -> int | None:
    for path in (".//JournalIssue/PubDate/Year", ".//PubDate/Year", ".//ArticleDate/Year"):
        node = article.find(path)
        if node is not None and node.text and node.text[:4].isdigit():
            return int(node.text[:4])
    medline = article.find(".//JournalIssue/PubDate/MedlineDate")
    if medline is not None and medline.text and medline.text[:4].isdigit():
        return int(medline.text[:4])
    return None


def efetch(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    params = urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    )
    root = ET.fromstring(_get(f"{EUTILS}/efetch.fcgi?{params}"))

    records = []
    for article in root.iter("PubmedArticle"):
        pmid_node = article.find(".//PMID")
        title_node = article.find(".//ArticleTitle")
        journal_node = article.find(".//Journal/Title")
        if pmid_node is None or title_node is None:
            continue

        doi = ""
        for eid in article.iter("ArticleId"):
            if eid.get("IdType") == "doi" and eid.text:
                doi = eid.text.strip()
                break

        records.append(
            {
                "pmid": pmid_node.text.strip(),
                "title": "".join(title_node.itertext()).strip().rstrip("."),
                "source": (journal_node.text or "").strip() if journal_node is not None else "",
                "year": _year(article),
                "doi": doi,
                "text": _abstract_text(article),
            }
        )
    return records


def harvest() -> list[Document]:
    by_pmid: dict[str, Document] = {}

    for tag, query in TOPICS:
        print(f"[{tag}]")
        pmids = esearch(query, PER_TOPIC * 3)
        time.sleep(0.4)
        records = efetch(pmids)
        time.sleep(0.4)

        kept = 0
        for rec in records:
            if kept >= PER_TOPIC:
                break
            # Reject anything we cannot cite honestly: no abstract, no year, no journal.
            if not rec["text"] or len(rec["text"]) < MIN_ABSTRACT_CHARS:
                continue
            if not rec["year"] or not rec["source"]:
                continue

            pmid = rec["pmid"]
            if pmid in by_pmid:
                if tag not in by_pmid[pmid].tags:
                    by_pmid[pmid].tags.append(tag)
                continue

            by_pmid[pmid] = Document(
                id="",  # assigned after sorting, so ids are stable and readable
                title=rec["title"],
                source=rec["source"],
                year=rec["year"],
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                doi=rec["doi"],
                pmid=pmid,
                text=rec["text"],
                tags=[tag],
            )
            kept += 1
            print(f"    + {rec['year']}  {rec['title'][:72]}")

        if kept < PER_TOPIC:
            print(f"    ! only {kept}/{PER_TOPIC} usable records for '{tag}'")

    docs = sorted(by_pmid.values(), key=lambda d: (d.tags[0], -d.year))
    for i, doc in enumerate(docs, start=1):
        doc.id = f"S{i:02d}"
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/evidence_library.json")
    args = parser.parse_args()

    docs = harvest()
    payload = [asdict(d) for d in docs]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\n{len(docs)} documents -> {args.out}")
    tags = sorted({t for d in docs for t in d.tags})
    for tag in tags:
        print(f"  {sum(tag in d.tags for d in docs):2d}  {tag}")


if __name__ == "__main__":
    main()
