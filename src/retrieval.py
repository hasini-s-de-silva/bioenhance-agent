"""Evidence retrieval over the curated PubMed library.

Dense retrieval with sentence-transformers + FAISS, with a TF-IDF fallback so the
repository still runs end-to-end on a machine that cannot download model weights.
The backend in use is always reported, because a reviewer should never have to guess
which retriever produced a result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .schemas import EvidenceDoc, MolecularDescriptors, RetrievedEvidence, SolubilityPrediction

ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = ROOT / "data" / "evidence_library.json"
EMBED_CACHE = ROOT / "data" / "evidence_embeddings.npy"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_library(path: Path = LIBRARY_PATH) -> list[EvidenceDoc]:
    docs = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvidenceDoc(**d) for d in docs]


class EvidenceIndex:
    """Vector index over the evidence library.

    Embeddings are cached to disk keyed on library size, so Streamlit reruns and the
    evaluation harness don't re-embed 50 abstracts every time.
    """

    def __init__(self, docs: list[EvidenceDoc] | None = None, prefer_dense: bool | None = None):
        self.docs = docs if docs is not None else load_library()
        self.backend = "none"
        self._model = None
        self._vectorizer = None
        self._matrix: np.ndarray | None = None

        # BIOENHANCE_RETRIEVER=dense|tfidf|auto. Explicit beats guessing: CI and
        # offline demos pin tfidf so a Hugging Face outage cannot fail the run.
        if prefer_dense is None:
            setting = os.environ.get("BIOENHANCE_RETRIEVER", "auto").lower()
            prefer_dense = setting in {"auto", "dense"}

        if prefer_dense and self._try_dense():
            self.backend = "sentence-transformers/all-MiniLM-L6-v2 + faiss"
        else:
            self._build_sparse()
            self.backend = "tf-idf (sparse fallback)"

    # -- corpus text -------------------------------------------------------

    def _corpus(self) -> list[str]:
        # Title and tags carry a lot of signal for short queries, so weight them in
        # by repeating them alongside the abstract.
        return [
            f"{d.title}. {' '.join(d.tags)}. {d.title}. {d.text}" for d in self.docs
        ]

    # -- dense backend -----------------------------------------------------

    def _try_dense(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:  # noqa: BLE001 - fall back to TF-IDF
            return False

        try:
            self._model = SentenceTransformer(EMBED_MODEL)
            vectors = self._load_or_embed()
            self._matrix = vectors
            self._build_faiss(vectors)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[retrieval] dense backend unavailable ({exc}); using TF-IDF")
            return False

    def _load_or_embed(self) -> np.ndarray:
        if EMBED_CACHE.exists():
            cached = np.load(EMBED_CACHE)
            if cached.shape[0] == len(self.docs):
                return cached
        vectors = self._model.encode(
            self._corpus(), normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        np.save(EMBED_CACHE, vectors)
        return vectors

    def _build_faiss(self, vectors: np.ndarray) -> None:
        try:
            import faiss

            index = faiss.IndexFlatIP(vectors.shape[1])  # cosine, vectors are normalised
            index.add(vectors)
            self._faiss = index
        except Exception:  # noqa: BLE001 - brute force is exact and fine at this size
            self._faiss = None

    # -- sparse backend ----------------------------------------------------

    def _build_sparse(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = self._vectorizer.fit_transform(self._corpus()).toarray().astype("float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / np.clip(norms, 1e-9, None)

    # -- query -------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        if self._model is not None:
            return self._model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")
        vec = self._vectorizer.transform([query]).toarray().astype("float32")
        return vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9, None)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        q = self._embed_query(query)

        if getattr(self, "_faiss", None) is not None:
            scores, idx = self._faiss.search(q, min(top_k, len(self.docs)))
            pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
        else:
            sims = (self._matrix @ q[0]).tolist()
            order = np.argsort(sims)[::-1][:top_k]
            pairs = [(int(i), float(sims[i])) for i in order]

        return [
            RetrievedEvidence(doc=self.docs[i], score=round(float(s), 4))
            for i, s in pairs
            if i >= 0
        ]


_INDEX: EvidenceIndex | None = None


def get_index() -> EvidenceIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = EvidenceIndex()
    return _INDEX


def retrieve_relevant_evidence(query: str, top_k: int = 5) -> list[RetrievedEvidence]:
    """Retrieve the top_k most relevant evidence documents for a free-text query."""
    return get_index().search(query, top_k=top_k)


def build_query(
    desc: MolecularDescriptors,
    sol: SolubilityPrediction,
    constraints: str | None = None,
    dosage_form: str | None = None,
) -> str:
    """Turn the calculated molecular profile into a retrieval query.

    The query is written in the vocabulary of the formulation literature rather than
    as raw numbers, since the abstracts talk about "high lipophilicity", not "clogp=5.6".
    """
    parts: list[str] = []

    if sol.risk.value == "high":
        parts.append("poorly water soluble drug, low aqueous solubility, BCS class II")
    elif sol.risk.value == "moderate":
        parts.append("moderately soluble drug, dissolution rate limited absorption")
    else:
        parts.append("aqueous soluble drug oral formulation")

    if desc.clogp >= 5:
        parts.append("very high lipophilicity, high logP, grease ball molecule")
    elif desc.clogp >= 3:
        parts.append("lipophilic drug, solubilisation in lipid excipients")
    elif desc.clogp < 1:
        parts.append("low lipophilicity, brick dust molecule, crystal lattice limited")

    if desc.molecular_weight > 500:
        parts.append("high molecular weight compound")
    if desc.h_bond_donors >= 3:
        parts.append("hydrogen bonding, strong crystal lattice, high melting point")
    if desc.aromatic_rings >= 3 and desc.fraction_csp3 < 0.3:
        parts.append("planar aromatic rigid molecule, crystal packing")
    if desc.rotatable_bonds > 10:
        parts.append("flexible molecule")

    parts.append(
        "bioavailability enhancement strategy, amorphous solid dispersion, "
        "lipid-based formulation, cocrystal, salt formation, cyclodextrin, "
        "nanosuspension, particle size reduction, supersaturation, precipitation inhibition"
    )

    if dosage_form:
        parts.append(f"{dosage_form} dosage form")
    if constraints:
        parts.append(constraints)

    return ". ".join(parts)
