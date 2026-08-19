from .vector_store import VectorStore
from .bm25_index import IndexSearch
from .db import get_client, DEFAULT_DB, DEFAULT_COLLECTION
from .pipeline import DocumentChunk
from pathlib import Path
import os
import re
from dotenv import load_dotenv
load_dotenv()

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./chroma/bm25_index")
FULL_VECTOR_K = 100  # vector candidates pulled from full corpus
BM25_TOP_K = 100     # BM25 candidates pulled from full corpus
FINAL_K = 5


ARTICLE_PATTERN = re.compile(r"\barticle\s*(\d+)\b", re.IGNORECASE)
SECTION_LEADER = re.compile(r"^\s*(\d+)\.\s+[A-Z]", re.MULTILINE)


def _extract_article_number(text: str) -> int | None:
    head = text[:400]
    m = ARTICLE_PATTERN.search(head)
    if m:
        return int(m.group(1))
    m = SECTION_LEADER.search(head)
    if m:
        return int(m.group(1))
    return None


def _title_boost(query: str, doc: DocumentChunk, base_score: float) -> float:
    q = query.lower().strip()
    m = ARTICLE_PATTERN.search(q)
    if not m:
        return base_score
    wanted = int(m.group(1))
    got = _extract_article_number(doc.content)
    if got == wanted:
        return base_score + 50.0
    return base_score


def _lexical_article_match(query: str, docs: list[DocumentChunk]) -> DocumentChunk | None:
    """Force-include the actual article chunk for "article N" queries.

    Vector search and BM25 both miss chunks that begin with "N. Title..."
    because they don't contain the literal word "article". This scans the
    corpus for the chunk whose section number == wanted and returns it.
    """
    q = query.lower().strip()
    m = ARTICLE_PATTERN.search(q)
    if not m:
        return None
    wanted = int(m.group(1))
    matches: list[DocumentChunk] = []
    for d in docs:
        if _extract_article_number(d.content) != wanted:
            continue
        head = d.content[:400].lower()
        other = ARTICLE_PATTERN.findall(head)
        other_nums = {int(x) for x in other if int(x) != wanted}
        if len(other_nums) > 2:
            continue
        # Constitutional articles live in the body (PART I-XXII, ~ pages
        # 33-300 of coi.pdf). Skip Sixth Schedule, Third Schedule etc.
        # which number their paragraphs with the same scheme.
        if "schedule" in head and "part" not in head.split("schedule")[0]:
            continue
        if "amendment act" in head[:80]:
            continue
        matches.append(d)
    if not matches:
        return None
    matches.sort(key=lambda d: d.source.page)
    return matches[0]


def _load_corpus() -> tuple[list[DocumentChunk], dict[str, DocumentChunk]]:
    """Read all chunks from Mongo, return list + id-index."""
    client = get_client()
    collection = client[DEFAULT_DB][DEFAULT_COLLECTION]
    docs = [DocumentChunk.model_validate(d) for d in collection.find({})]
    return docs, {d.id: d for d in docs}


def _load_or_build_bm25(docs: list[DocumentChunk]) -> IndexSearch:
    index_path = Path(BM25_INDEX_PATH)
    if index_path.exists():
        try:
            return IndexSearch.load(str(index_path))
        except Exception:
            pass
    texts = [d.content for d in docs]
    index = IndexSearch()
    index.index(texts)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(index_path))
    return index


def search(query: str, k: int = FINAL_K, filename: str | None = None):
    vectorstore = VectorStore()

    collection_name = os.environ.get(
        "CHROMA_COLLECTION",
        DEFAULT_COLLECTION
    )

    where = {"source": filename} if filename else None

    # 1) Vector recall over full corpus
    vec_results = vectorstore.search(
        collection_name=collection_name,
        query=query,
        k=FULL_VECTOR_K,
        where=where
    )
    vec_ids = vec_results["ids"][0]
    # Vector distance -> similarity (cosine: smaller dist = more similar)
    vec_dists = vec_results.get("distances", [[]])[0]
    if len(list(vec_dists)) > 0:
        vec_sims = {
            _id: 1.0 - float(d)
            for _id, d in zip(vec_ids, vec_dists)
        }
    else:
        vec_sims = {_id: 0.0 for _id in vec_ids}

    # 2) Full corpus BM25
    docs, docs_by_id = _load_corpus()
    if not docs:
        return []

    if filename:
        docs = [d for d in docs if d.source.filename == filename]
        if not docs:
            return []
        docs_by_id = {d.id: d for d in docs}

    bm25_index = _load_or_build_bm25(docs)
    bm25_hits = bm25_index.search(query, k=min(BM25_TOP_K, len(docs)))
    bm25_indices = bm25_hits.documents[0]
    bm25_scores_raw = list(bm25_hits.scores[0]) if hasattr(bm25_hits, "scores") else [0.0] * len(bm25_indices)

    # Normalize BM25 scores per query (min-max to [0, 1])
    if len(bm25_scores_raw) > 0:
        lo = min(bm25_scores_raw)
        hi = max(bm25_scores_raw)
        span = hi - lo if hi != lo else 1.0
        bm25_norm = {docs[i].id: (s - lo) / span for i, s in zip(bm25_indices, bm25_scores_raw)}
    else:
        bm25_norm = {}

    # 3) Fuse: union of candidates, weighted sum
    # Vector embeddings on this corpus confuse "article 1" because every
    # page that mentions "article 19" or "article 30" also contains "1".
    # BM25 alone retrieves Article 1 reliably.
    fused: dict[str, float] = {}
    for cid, sim in vec_sims.items():
        fused[cid] = max(fused.get(cid, 0.0), 0.6 * sim)
    for cid, score in bm25_norm.items():
        fused[cid] = fused.get(cid, 0.0) + 0.8 * score

    # 4) Rank + title boost
    ranked_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    results: list[DocumentChunk] = []
    for cid, score in ranked_ids:
        if cid not in docs_by_id:
            continue
        doc = docs_by_id[cid]
        boosted = _title_boost(query, doc, score)
        results.append((boosted, doc))

    # 5) Force-include the actual article chunk for "article N" queries.
    # Vector search + BM25 miss page-33-style chunks that start with "1.
    # Name and territory of the Union" because they don't contain the
    # literal word "article".
    forced = _lexical_article_match(query, docs)
    if forced is not None:
        results.append((1e9, forced))
        # Dedup
        seen: set[str] = set()
        deduped = []
        for score, doc in sorted(results, key=lambda x: x[0], reverse=True):
            if doc.id in seen:
                continue
            seen.add(doc.id)
            deduped.append(doc)
        return deduped[:k]

    results.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in results[:k]]
