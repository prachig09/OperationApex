"""Semantic data-drift monitor for document embeddings.

Compares new embeddings against a baseline built from the Stage 1 Chroma store:

* Cosine distance (hand-implemented): distance between the new batch's centroid and the baseline
  centroid, plus the mean per-vector distance to the baseline centroid.
* KL divergence (hand-implemented, KL(new || baseline)) between binned distributions of
    - each vector's cosine distance to the baseline centroid (radial feature), and
    - projections onto the baseline's top principal components (scikit-learn PCA).
  The reported KL is the mean over those features. Bins are baseline quantiles with open-ended
  outer bins and Laplace smoothing.

Thresholds are calibrated by a holdout bootstrap on the baseline, per batch size, unless set
explicitly. ``DataDriftWarning`` is raised (or emitted with ``mode="warn"``) when either metric
exceeds its threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

from config import ApexConfig
from security.network_guard import local_processing_guarantee

LOG = logging.getLogger("apex.drift")
EPS = 1e-12


class DataDriftWarning(UserWarning):
    """Raised (or warned) when embedding drift exceeds a threshold."""

    def __init__(self, message: str, report: "DriftReport") -> None:
        """Carry the full drift report."""
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------- metric primitives
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance 1 - (a.b)/(|a||b|), clipped to [0, 2]."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < EPS:
        raise ValueError("cosine distance undefined for a zero-norm vector")
    return float(np.clip(1.0 - float(a @ b) / denom, 0.0, 2.0))


def cosine_distances_to(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Vectorized cosine distance of every row of ``matrix`` to ``vector``."""
    matrix, vector = np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64)
    vnorm = float(np.linalg.norm(vector))
    if vnorm < EPS:
        raise ValueError("reference vector has zero norm")
    sims = (matrix @ vector) / (np.maximum(np.linalg.norm(matrix, axis=1), EPS) * vnorm)
    return np.clip(1.0 - sims, 0.0, 2.0)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) = sum_i p_i ln(p_i / q_i) in nats; inputs are normalized, q is floored at EPS."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be 1-D arrays of equal length")
    if (p < 0).any() or (q < 0).any() or p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("p and q must be non-negative with positive mass")
    p, q = p / p.sum(), np.maximum(q / q.sum(), EPS)
    mask = p > 0
    return float(max(0.0, np.sum(p[mask] * np.log(p[mask] / q[mask]))))


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior bin edges at baseline quantiles (outer bins are open-ended)."""
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))


def binned_distribution(values: np.ndarray, edges: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Laplace-smoothed probability mass over bins defined by ``edges``."""
    idx = np.searchsorted(edges, np.asarray(values, dtype=np.float64), side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(np.float64)
    return (counts + alpha) / (counts.sum() + alpha * len(counts))


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows, rejecting NaNs and zero vectors."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2-D array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (norms < EPS).any():
        raise ValueError("embeddings contain NaN/inf or zero vectors")
    return x / norms


# ------------------------------------------------------------- reference model
@dataclass
class _Reference:
    centroid: np.ndarray
    mean_doc_distance: float
    radial_edges: np.ndarray
    radial_probs: np.ndarray
    pca: Optional[PCA]
    comp_edges: List[np.ndarray]
    comp_probs: List[np.ndarray]


def _build_reference(x: np.ndarray, n_components: int, max_bins: int, alpha: float, seed: int) -> _Reference:
    n, d = x.shape
    centroid = x.mean(axis=0)
    dists = cosine_distances_to(x, centroid)
    bins = int(np.clip(n // 10, 4, max_bins))
    r_edges = quantile_edges(dists, bins)
    pca, c_edges, c_probs = None, [], []
    k = min(n_components, n - 1, d)
    if k >= 1:
        pca = PCA(n_components=k, svd_solver="full", random_state=seed).fit(x)
        proj = pca.transform(x)
        for j in range(k):
            edges = quantile_edges(proj[:, j], bins)
            c_edges.append(edges)
            c_probs.append(binned_distribution(proj[:, j], edges, alpha))
    return _Reference(centroid, float(dists.mean()), r_edges, binned_distribution(dists, r_edges, alpha), pca, c_edges, c_probs)


def _score(ref: _Reference, x: np.ndarray, alpha: float) -> Dict[str, Any]:
    dists = cosine_distances_to(x, ref.centroid)
    kl_radial = kl_divergence(binned_distribution(dists, ref.radial_edges, alpha), ref.radial_probs)
    kl_components: List[float] = []
    if ref.pca is not None:
        proj = ref.pca.transform(x)
        for j, (edges, probs) in enumerate(zip(ref.comp_edges, ref.comp_probs)):
            kl_components.append(kl_divergence(binned_distribution(proj[:, j], edges, alpha), probs))
    return {
        "centroid_cosine_distance": cosine_distance(x.mean(axis=0), ref.centroid),
        "mean_doc_cosine_distance": float(dists.mean()),
        "kl_radial": kl_radial,
        "kl_components": kl_components,
        "kl_divergence": float(np.mean([kl_radial] + kl_components)),
    }


@dataclass
class DriftReport:
    """Result of one drift evaluation."""

    n_baseline: int
    n_new: int
    centroid_cosine_distance: float
    mean_doc_cosine_distance: float
    baseline_mean_doc_distance: float
    kl_radial: float
    kl_components: List[float]
    kl_divergence: float
    cosine_threshold: float
    kl_threshold: float
    cosine_exceeded: bool
    kl_exceeded: bool
    drifted: bool

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view."""
        return asdict(self)


# --------------------------------------------------------------------- monitor
class SemanticDriftMonitor:
    """Baseline-centroid drift detector over embedding batches."""

    def __init__(
        self, cosine_threshold: Optional[float] = None, kl_threshold: Optional[float] = None,
        n_components: int = 8, max_bins: int = 10, n_boot: int = 30, alpha: float = 0.5,
        min_cosine_threshold: float = 0.01, min_kl_threshold: float = 0.05, seed: int = 7,
    ) -> None:
        """Configure thresholds (None = auto-calibrate by bootstrap) and binning."""
        self.cosine_threshold, self.kl_threshold = cosine_threshold, kl_threshold
        self.n_components, self.max_bins, self.n_boot, self.alpha = n_components, max_bins, n_boot, alpha
        self.min_cosine_threshold, self.min_kl_threshold, self.seed = min_cosine_threshold, min_kl_threshold, seed
        self._x: Optional[np.ndarray] = None
        self._ref: Optional[_Reference] = None
        self._cache: Dict[int, Tuple[float, float]] = {}

    # --- fitting / persistence ---
    def fit(self, embeddings: np.ndarray) -> "SemanticDriftMonitor":
        """Fit the baseline centroid, radial distribution and PCA reference."""
        x = unit_rows(embeddings)
        if x.shape[0] < 20:
            raise ValueError("baseline needs at least 20 embeddings; ingest more documents or pass explicit thresholds with a larger set")
        self._x = x
        self._ref = _build_reference(x, self.n_components, self.max_bins, self.alpha, self.seed)
        self._cache.clear()
        return self

    @classmethod
    def from_chroma(cls, store: Any, exclude_doc_ids: Sequence[str] = (), **kwargs: Any) -> "SemanticDriftMonitor":
        """Build a baseline from a Stage 1 ``ChromaStore`` (optionally excluding documents by ``doc_id``)."""
        query: Dict[str, Any] = {"include": ["embeddings"]}
        if exclude_doc_ids:
            query["where"] = {"doc_id": {"$nin": list(exclude_doc_ids)}}
        result = store.collection.get(**query)
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Chroma collection has no embeddings; run Stage 1 ingestion first")
        return cls(**kwargs).fit(np.asarray(embeddings, dtype=np.float64))

    def save(self, path: Path) -> None:
        """Persist the baseline embeddings and parameters (local file; the reference is refit on load)."""
        self._require()
        path.parent.mkdir(parents=True, exist_ok=True)
        params = dict(
            cosine_threshold=self.cosine_threshold, kl_threshold=self.kl_threshold, n_components=self.n_components,
            max_bins=self.max_bins, n_boot=self.n_boot, alpha=self.alpha,
            min_cosine_threshold=self.min_cosine_threshold, min_kl_threshold=self.min_kl_threshold, seed=self.seed,
        )
        np.savez_compressed(path, embeddings=self._x, params=np.array(json.dumps(params)))

    @classmethod
    def load(cls, path: Path) -> "SemanticDriftMonitor":
        """Restore a monitor saved with :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        return cls(**json.loads(str(data["params"]))).fit(data["embeddings"])

    def _require(self) -> Tuple[np.ndarray, _Reference]:
        if self._x is None or self._ref is None:
            raise RuntimeError("monitor has no baseline; call fit(), from_chroma() or load() first")
        return self._x, self._ref

    # --- thresholds ---
    def _thresholds(self, m: int) -> Tuple[float, float]:
        if self.cosine_threshold is not None and self.kl_threshold is not None:
            return self.cosine_threshold, self.kl_threshold
        if m not in self._cache:
            x, _ = self._require()
            rng = np.random.default_rng(self.seed + m)
            n = len(x)
            n_hold = max(5, n // 5)
            cos_vals, kl_vals = [], []
            for _ in range(self.n_boot):
                perm = rng.permutation(n)
                hold, ref_idx = perm[:n_hold], perm[n_hold:]
                ref = _build_reference(x[ref_idx], self.n_components, self.max_bins, self.alpha, self.seed)
                sample = x[rng.choice(hold, size=m, replace=m > n_hold)]
                s = _score(ref, sample, self.alpha)
                cos_vals.append(s["centroid_cosine_distance"])
                kl_vals.append(s["kl_divergence"])
            c, k = np.asarray(cos_vals), np.asarray(kl_vals)
            self._cache[m] = (
                max(float(c.mean() + 3 * c.std()), float(c.max()), self.min_cosine_threshold),
                max(float(k.mean() + 3 * k.std()), float(k.max()), self.min_kl_threshold),
            )
        auto_c, auto_k = self._cache[m]
        return (
            self.cosine_threshold if self.cosine_threshold is not None else auto_c,
            self.kl_threshold if self.kl_threshold is not None else auto_k,
        )

    # --- evaluation ---
    def evaluate(self, new_embeddings: np.ndarray) -> DriftReport:
        """Compute cosine and KL drift of a new batch against the baseline."""
        x, ref = self._require()
        new = unit_rows(new_embeddings)
        if new.shape[1] != x.shape[1]:
            raise ValueError(f"dimension mismatch: baseline {x.shape[1]} vs new {new.shape[1]}")
        s = _score(ref, new, self.alpha)
        c_thr, k_thr = self._thresholds(len(new))
        c_hit, k_hit = s["centroid_cosine_distance"] > c_thr, s["kl_divergence"] > k_thr
        return DriftReport(
            n_baseline=len(x), n_new=len(new), centroid_cosine_distance=s["centroid_cosine_distance"],
            mean_doc_cosine_distance=s["mean_doc_cosine_distance"], baseline_mean_doc_distance=ref.mean_doc_distance,
            kl_radial=s["kl_radial"], kl_components=s["kl_components"], kl_divergence=s["kl_divergence"],
            cosine_threshold=c_thr, kl_threshold=k_thr, cosine_exceeded=c_hit, kl_exceeded=k_hit, drifted=c_hit or k_hit,
        )

    def check(self, new_embeddings: np.ndarray, mode: str = "raise") -> DriftReport:
        """Evaluate and raise ``DataDriftWarning`` (mode='raise') or emit it via ``warnings`` (mode='warn')."""
        if mode not in ("raise", "warn"):
            raise ValueError("mode must be 'raise' or 'warn'")
        report = self.evaluate(new_embeddings)
        if report.drifted:
            parts = []
            if report.cosine_exceeded:
                parts.append(f"cosine distance {report.centroid_cosine_distance:.4f} > {report.cosine_threshold:.4f}")
            if report.kl_exceeded:
                parts.append(f"KL divergence {report.kl_divergence:.4f} > {report.kl_threshold:.4f}")
            drift = DataDriftWarning("Semantic drift detected: " + "; ".join(parts), report)
            if mode == "raise":
                raise drift
            warnings.warn(drift, stacklevel=2)
        return report

    def check_chunks(self, chunks: Sequence[Any], embedder: Any, mode: str = "raise") -> DriftReport:
        """Embed Stage 1 ``Chunk`` objects locally (zero sockets) and run :meth:`check`."""
        with local_processing_guarantee(allow_loopback=False):
            vectors = embedder.embed([c.text for c in chunks])
        return self.check(np.asarray(vectors, dtype=np.float64), mode)


# ------------------------------------------------------------------------- CLI
def run_selftest(seed: int = 11) -> Dict[str, Any]:
    """Verify the hand-written math against SciPy and prove drift detection on synthetic vectors."""
    from scipy.spatial.distance import cosine as scipy_cosine
    from scipy.stats import entropy

    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=64), rng.normal(size=64)
    assert np.isclose(cosine_distance(a, b), scipy_cosine(a, b), atol=1e-12), "cosine mismatch vs SciPy"
    p, q = rng.dirichlet(np.ones(12)), rng.dirichlet(np.ones(12))
    assert np.isclose(kl_divergence(p, q), entropy(p, q), atol=1e-9), "KL mismatch vs SciPy"
    assert kl_divergence(p, p) < 1e-12 and cosine_distance(a, a) < 1e-12

    d = 64
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)

    def cloud(center: np.ndarray, n: int) -> np.ndarray:
        return center + 0.6 * rng.normal(size=(n, d)) / np.sqrt(d)

    monitor = SemanticDriftMonitor(seed=seed).fit(cloud(u, 300))
    in_dist = monitor.evaluate(cloud(u, 40))
    w = rng.normal(size=d)
    v = u + w / np.linalg.norm(w)
    shifted = monitor.evaluate(cloud(v / np.linalg.norm(v), 40))
    assert shifted.drifted, "synthetic drift was not detected"
    return {"math_matches_scipy": True, "in_distribution": in_dist.to_dict(), "shifted": shifted.to_dict()}


def _embed_pdfs(paths: Sequence[Path], cfg: ApexConfig) -> Dict[str, np.ndarray]:
    """Parse, chunk and embed PDFs with the Stage 1 components (zero sockets)."""
    from pipeline.ingest import LocalEmbedder, SemanticChunker, group_sections, parse_document

    out: Dict[str, np.ndarray] = {}
    with local_processing_guarantee(allow_loopback=False):
        embedder = LocalEmbedder(cfg.embedding_model_path, cfg.embed_batch_size)
        chunker = SemanticChunker(embedder.tokenizer, cfg.chunk_tokens, cfg.chunk_overlap)
    for path in paths:
        doc_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        sections = group_sections(parse_document(path), fallback_title=path.stem)
        chunks = chunker.chunk_document(doc_id, path.name, sections)
        if not chunks:
            LOG.warning("no chunks produced for %s", path.name)
            continue
        with local_processing_guarantee(allow_loopback=False):
            out[path.name] = np.asarray(embedder.embed([c.text for c in chunks]), dtype=np.float64)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``--selftest`` | ``--fit`` | ``--check PATH...``."""
    parser = argparse.ArgumentParser(description="Apex semantic drift monitor.")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--fit", action="store_true", help="Build the baseline from the Chroma store.")
    parser.add_argument("--exclude-doc-id", action="append", default=[], help="doc_id(s) to leave out of the baseline.")
    parser.add_argument("--check", nargs="+", type=Path, help="PDF files or directories to test against the baseline.")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--mode", choices=["raise", "warn"], default="raise")
    parser.add_argument("--cosine-threshold", type=float, default=None)
    parser.add_argument("--kl-threshold", type=float, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = ApexConfig()
    baseline_path = args.baseline or cfg.data_dir / "observability" / "baseline.npz"

    if args.selftest:
        print(json.dumps(run_selftest(), indent=2))
        return 0
    if args.fit:
        from pipeline.ingest import ChromaStore

        with local_processing_guarantee(cfg.allowed_hosts):
            store = ChromaStore(cfg.chroma_dir, cfg.chroma_collection)
            monitor = SemanticDriftMonitor.from_chroma(
                store, args.exclude_doc_id, cosine_threshold=args.cosine_threshold, kl_threshold=args.kl_threshold
            )
        monitor.save(baseline_path)
        print(f"baseline saved: {baseline_path} ({len(monitor._x)} embeddings)")  # noqa: SLF001
        return 0
    if args.check:
        monitor = SemanticDriftMonitor.load(baseline_path)
        if args.cosine_threshold is not None:
            monitor.cosine_threshold = args.cosine_threshold
        if args.kl_threshold is not None:
            monitor.kl_threshold = args.kl_threshold
        files: List[Path] = []
        for p in args.check:
            files += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
        drifted = False
        for name, vectors in _embed_pdfs(files, cfg).items():
            try:
                report = monitor.check(vectors, mode=args.mode)
            except DataDriftWarning as exc:
                report, drifted = exc.report, True
            drifted = drifted or report.drifted
            print(json.dumps({"file": name, **report.to_dict()}))
        return 4 if drifted else 0
    parser.error("choose --selftest, --fit or --check")
    return 1


if __name__ == "__main__":
    sys.exit(main())