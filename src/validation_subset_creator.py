from __future__ import annotations
import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from collections import defaultdict


class _DefaultLogger:
    def log(self, msg: str) -> None:
        print(msg)


class ValidationSubsetCreator:
    """
    Create validation folds that are (a) as uniform in size as possible and
    (b) stratified by label when labels are sufficiently frequent; otherwise,
    fall back to semantic clustering.

    Key improvements vs. the original:
      - Preserve stratification (no global reshuffle that destroys label balance).
      - Single RNG for determinism without per-class re-seeding artifacts.
      - Guards for n_folds > n_samples and KMeans sample/cluster counts.
      - Faster/leaner embeddings path (convert_to_numpy=True).
      - Option to include/exclude 'output' text from embeddings during clustering.
    """

    def __init__(
        self,
        logger: Any = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_folds: int = 5,
        seed: int = 42,
        include_output_in_text_for_clustering: bool = False,
        normalize_embeddings: bool = False,
        kmeans_n_init: int = 10,
    ):
        self.logger = logger or _DefaultLogger()
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.include_output_in_text_for_clustering = bool(include_output_in_text_for_clustering)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.kmeans_n_init = int(kmeans_n_init)

        # Initialize the embedding model lazily to avoid heavy imports if unused
        self._embedding_model_name = embedding_model
        self._embedding_model: SentenceTransformer | None = None

        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    # ---------- Public API ----------

    def create_subsets(self, val_examples: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Creates folds with uniform size and balanced class distribution (when possible).
        If labels are mostly unique/sparse, falls back to semantic clustering.

        Each example in val_examples is expected to be a dict with keys:
            - 'input'  (str)
            - 'output' (label or str)
        """
        val_examples = list(val_examples)
        total_examples = len(val_examples)
        if total_examples == 0:
            self.logger.log("No validation examples provided.")
            return [[] for _ in range(self.n_folds)]

        # Guard: cannot have more folds than examples (reduce or raise)
        if self.n_folds > total_examples:
            self.logger.log(
                f"Reducing n_folds from {self.n_folds} to {total_examples} (not enough samples)."
            )
            self.n_folds = total_examples

        # Build label buckets
        class_to_examples: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for ex in val_examples:
            class_to_examples[ex["output"]].append(ex)

        unique_labels = len(class_to_examples)
        # Heuristic for "labels are sparse or mostly unique"
        labels_mostly_unique = (
            unique_labels > 0
            and ((unique_labels / total_examples) > 0.8 or (total_examples / unique_labels) < 2)
        )

        # Compute per-fold target sizes (uniform)
        base = total_examples // self.n_folds
        rem = total_examples % self.n_folds
        target_sizes = [base + (1 if i < rem else 0) for i in range(self.n_folds)]
        self.logger.log(
            f"Creating {self.n_folds} folds with target sizes {target_sizes}"
        )

        if labels_mostly_unique:
            self.logger.log(
                "Most labels are unique or have few samples. Using uniform global clusters."
            )
            return self._create_uniform_global_clusters(val_examples, target_sizes)

        # Otherwise, stratified allocation preserving target_sizes
        self.logger.log(
            f"Stratifying by {unique_labels} classes with total {total_examples} examples."
        )
        rng = np.random.default_rng(self.seed)
        folds = self._stratified_uniform_allocation(class_to_examples, target_sizes, rng)

        self.logger.log(f"Final fold sizes: {[len(f) for f in folds]}")
        return folds

    # ---------- Internal helpers ----------

    def _create_uniform_global_clusters(
        self,
        val_examples: List[Dict[str, Any]],
        target_sizes: List[int],
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster examples semantically into n_folds buckets, then redistribute to match target_sizes.
        """
        n_samples = len(val_examples)
        n_clusters = min(self.n_folds, n_samples)

        # Texts to embed (optionally include the output label text)
        if self.include_output_in_text_for_clustering:
            texts = [f"{ex['input']} [SEP] {ex['output']}" for ex in val_examples]
        else:
            texts = [ex["input"] for ex in val_examples]

        # Encode to numpy for efficiency; optional normalization
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        if n_clusters <= 1:
            # Degenerate case: just slice to targets
            self.logger.log("n_clusters <= 1, skipping KMeans; slicing to targets.")
            return self._slice_to_targets(val_examples, target_sizes)

        # KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.seed,
            n_init=self.kmeans_n_init,
        )
        cluster_labels = kmeans.fit_predict(embeddings)

        # Map cluster -> examples
        cluster_to_examples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for idx, ex in enumerate(val_examples):
            cluster_to_examples[cluster_labels[idx]].append(ex)

        # Within each cluster, sort by example id BEFORE shuffling so the
        # input order to rng.shuffle is itself deterministic. KMeans cluster
        # assignment can subtly depend on input order (e.g. ties broken by
        # insertion order), and any earlier non-deterministic step in the
        # pipeline could otherwise leak into validation subset selection.
        rng = np.random.default_rng(self.seed)
        all_examples = []
        for cid in sorted(cluster_to_examples.keys()):
            lst = sorted(cluster_to_examples[cid],
                         key=lambda ex: str(ex.get('id', '')))
            rng.shuffle(lst)
            all_examples.extend(lst)

        # Now place into folds honoring target_sizes
        folds = [[] for _ in range(self.n_folds)]
        cursor = 0
        for f_idx, target in enumerate(target_sizes):
            if target > 0:
                chunk = all_examples[cursor : cursor + target]
                folds[f_idx].extend(chunk)
                cursor += target

        self.logger.log(f"Final fold sizes: {[len(f) for f in folds]}")
        return folds

    def _slice_to_targets(
        self, examples: List[Dict[str, Any]], target_sizes: List[int]
    ) -> List[List[Dict[str, Any]]]:
        """Helper: simple slicing into folds by target sizes (used for degenerate cases)."""
        rng = np.random.default_rng(self.seed)
        examples = examples.copy()
        rng.shuffle(examples)

        folds = [[] for _ in range(len(target_sizes))]
        cursor = 0
        for i, t in enumerate(target_sizes):
            folds[i] = examples[cursor : cursor + t]
            cursor += t
        return folds

    def _stratified_uniform_allocation(
        self,
        class_to_examples: Dict[Any, List[Dict[str, Any]]],
        target_sizes: List[int],
        rng: np.random.Generator,
    ) -> List[List[Dict[str, Any]]]:
        """
        Allocate each class across folds using largest-remainder quotas so that:
          - The total per fold matches target_sizes exactly
          - Class proportions are approximately preserved (stratified)

        Steps:
          1) Compute fractional quotas q_{c,f} = target_sizes[f] * (n_c / N).
          2) Take floors, then distribute remaining items for each class to folds with
             largest fractional parts.
          3) Shuffle within each class once and slice according to allocations.
        """
        n_folds = len(target_sizes)
        total = sum(len(v) for v in class_to_examples.values())
        if total == 0:
            return [[] for _ in range(n_folds)]

        # Initialize allocations per class per fold
        allocations: Dict[Any, List[int]] = {c: [0] * n_folds for c in class_to_examples}

        # First pass: per-class quotas
        for c, exs in class_to_examples.items():
            n_c = len(exs)
            # fractional quotas per fold for this class
            frac = [target_sizes[f] * (n_c / total) for f in range(n_folds)]
            floors = [int(math.floor(x)) for x in frac]
            alloc = floors[:]
            leftover = n_c - sum(floors)

            if leftover > 0:
                # Indices sorted by descending fractional remainder
                remainders = [x - math.floor(x) for x in frac]
                order = np.argsort([-r for r in remainders])
                for f in order[:leftover]:
                    alloc[f] += 1

            allocations[c] = alloc

        # Sanity: make sure each fold's sum across classes equals target_sizes[f]
        per_fold_totals = [0] * n_folds
        for c in allocations:
            for f in range(n_folds):
                per_fold_totals[f] += allocations[c][f]

        # Mild corrective pass if rounding drifted (should be rare)
        drift = [per_fold_totals[f] - target_sizes[f] for f in range(n_folds)]
        if any(d != 0 for d in drift):
            # Try to fix by moving 1 item of some class from overfull folds to underfull folds
            over = [i for i, d in enumerate(drift) if d > 0]
            under = [i for i, d in enumerate(drift) if d < 0]
            for f_over in over:
                for f_under in under:
                    need = -drift[f_under]
                    give = drift[f_over]
                    if need <= 0 or give <= 0:
                        continue
                    moved = False
                    # find a class with alloc > 0 in f_over to move
                    for c in allocations:
                        if allocations[c][f_over] > 0:
                            allocations[c][f_over] -= 1
                            allocations[c][f_under] += 1
                            drift[f_over] -= 1
                            drift[f_under] += 1
                            moved = True
                            break
                    if not moved:
                        # couldn't move from this overfull fold; try next under
                        continue
                    if drift[f_over] == 0:
                        break

        # Build folds by slicing shuffled class lists according to allocations
        folds = [[] for _ in range(n_folds)]
        for c, exs in class_to_examples.items():
            exs = exs.copy()
            rng.shuffle(exs)
            idx = 0
            for f in range(n_folds):
                take = allocations[c][f]
                if take > 0:
                    folds[f].extend(exs[idx : idx + take])
                    idx += take

        return folds
        
    def reseed(self, new_seed):
        """Reseed the random number generator for this subset creator."""
        self.seed = int(new_seed)