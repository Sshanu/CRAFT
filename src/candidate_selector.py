import random
import numpy as np
import math
import numpy as np

import numpy as np
from collections import Counter

def _map_params(
    filter_factor: float,
    q_min: float = 0.70,
    q_max: float = 0.98,
    delta_min: float = 0.05,
    delta_max: float = 0.15):
    f = float(np.clip(filter_factor, 0.0, 1.0))
    q = q_min + (q_max - q_min) * f
    delta = delta_max - (delta_max - delta_min) * f
    return q, delta

def filter_bad_candidates(
    candidates,
    logger,
    score_attr: str = "eval_score",   # posterior mean (higher is better)
    std_attr: str  = "eval_std",      # posterior std (>=0); if missing, treated as 0
    filter_factor: float = 0.7,       # single strictness knob [0,1]
    min_points: int = 1,              # floor, not a cap
    # mapping ranges (tune if desired)
    q_min: float = 0.70,
    q_max: float = 0.98,
    delta_min: float = 0.05,
    delta_max: float = 0.15,
    # one knob for both UCB/LCB
    bound: str = "lcb",               # "lcb" | "ucb" | "mean"
    beta: float = 1.0,                # used for both LCB and UCB
):
    """
    Batch-relative, uncertainty-aware pruning.

    metric =
        - "lcb":  mu - beta*std
        - "ucb":  mu + beta*std
        - "mean": mu

    threshold = 0.5 * ( quantile_q(metric) + (max(metric) - δ) )
    """
    if not candidates:
        return candidates

    # map (q, δ)
    q, delta = _map_params(
        filter_factor, q_min=q_min, q_max=q_max,
        delta_min=delta_min, delta_max=delta_max,
    )

    mu = np.array([float(getattr(c, score_attr, 0.0)) for c in candidates], dtype=float)
    sd = np.array([float(getattr(c, std_attr, 0.0))  for c in candidates], dtype=float)
    sd = np.where(np.isfinite(sd), sd, 0.0)

    if bound == "lcb":
        metric = mu - beta * sd
        tag = f"LCB(beta={beta:.2f})"
    elif bound == "ucb":
        metric = mu + beta * sd
        tag = f"UCB(beta={beta:.2f})"
    elif bound == "mean":
        metric = mu
        tag = "MEAN"
    else:
        raise ValueError("bound must be 'lcb' | 'ucb' | 'mean'")

    n = len(metric)
    max_m = float(np.max(metric))
    avg_m = float(np.mean(metric))
    logger.log(f"Batch {tag} stats - Max: {max_m:.3f}, Mean: {avg_m:.3f}, n={n}; q={q:.3f}, δ={delta:.3f}")

    if n == 1 or np.allclose(metric, metric[0]):
        logger.log(f"{tag} identical or single candidate. No filtering applied.")
        return candidates

    anchor_q   = float(np.quantile(metric, q))
    thr_margin = max_m - float(delta)
    threshold  = 0.5 * (anchor_q + thr_margin)

    logger.log(f"Filtering on {tag} with anchor={anchor_q:.3f}; max−δ={thr_margin:.3f}; thr={threshold:.3f}")

    kept_idx = np.where(metric >= threshold)[0]
    kept = [candidates[i] for i in kept_idx]

    if len(kept) < min_points:
        logger.log(f"Only {len(kept)} passed; topping up to {min_points} by {tag}.")
        order = np.argsort(-metric)
        kept = [candidates[i] for i in order[:min_points]]

    kept_vals = [float(getattr(c, score_attr, 0.0)) for c in kept]
    kept_stds = [float(getattr(c, std_attr, 0.0)) for c in kept]
    if bound == "lcb":
        kept_metric = [m - beta*s for m, s in zip(kept_vals, kept_stds)]
    elif bound == "ucb":
        kept_metric = [m + beta*s for m, s in zip(kept_vals, kept_stds)]
    else:
        kept_metric = kept_vals

    logger.log(f"Filtered out {n - len(kept)} below {threshold:.3f} (except min_points safeguard)")
    logger.log(f"Kept {tag}: {[round(s, 3) for s in sorted(kept_metric, reverse=True)]}")
    return kept

def calculate_crowding_distance(points, front):
    """
    Calculate the crowding distance for a single front.

    Args:
        points (array): Array of points [score, token_length].
        front (list): Indices of points in the front.

    Returns:
        crowding_distances (array): Crowding distances for points in the front.
    """
    crowding_distances = np.zeros(len(points))

    if len(front) <= 2:  # Boundary points get maximum crowding distance
        for idx in front:
            crowding_distances[idx] = np.inf
        return crowding_distances

    front_points = points[front]
    for m in range(points.shape[1]):  # Iterate over each objective
        sorted_indices = np.argsort(front_points[:, m])
        sorted_front = np.array(front)[sorted_indices]

        # Assign infinite distance to boundary points
        crowding_distances[sorted_front[0]] = np.inf
        crowding_distances[sorted_front[-1]] = np.inf

        # Normalize objective values
        min_value = front_points[sorted_indices[0], m]
        max_value = front_points[sorted_indices[-1], m]
        if max_value > min_value:
            normalized_values = (
                front_points[sorted_indices, m] - min_value
            ) / (max_value - min_value)
        else:
            normalized_values = np.zeros(len(sorted_indices))

        # Compute crowding distances for internal points
        for i in range(1, len(sorted_indices) - 1):
            crowding_distances[sorted_front[i]] += (
                normalized_values[i + 1] - normalized_values[i - 1]
            )

    return crowding_distances

def non_dominated_sort_with_buffer(points: np.ndarray, accuracy_buffer: float = 0.0, token_buffer: float = 0.0):
    """
    Perform non-dominated sorting on `points` with buffers for score and length.

    Args:
        points (np.ndarray): shape (N, 2) where each row is (score, length).
        accuracy_buffer (float): buffer for score.
        token_buffer (float): buffer for length.

    Returns:
        List[List[int]]: fronts, where fronts[0] is the rank-1 Pareto front,
                        fronts[1] the next front, and so on (last may be empty).
    """
    num_points = points.shape[0]
    domination_count = [0] * num_points
    dominates = [set() for _ in range(num_points)]

    # Pairwise comparisons
    for i in range(num_points):
        si, li = points[i]
        for j in range(num_points):
            if i == j:
                continue
            sj, lj = points[j]

            # if j dominates i
            if (
                sj >= si + accuracy_buffer
                and lj <= li - token_buffer
                and (sj > si + accuracy_buffer or lj < li - token_buffer)
            ):
                domination_count[i] += 1

            # else if i dominates j
            elif (
                si >= sj + accuracy_buffer
                and li <= lj - token_buffer
                and (si > sj + accuracy_buffer or li < lj - token_buffer)
            ):
                dominates[i].add(j)

    # Build fronts
    fronts = []
    first = [i for i in range(num_points) if domination_count[i] == 0]
    fronts.append(first)

    while fronts[-1]:
        next_front = []
        for i in fronts[-1]:
            # Iterate deterministically over dominated indices to avoid set-order nondeterminism
            for j in sorted(dominates[i]):
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)

    return fronts


def get_pareto_front_points(points: np.ndarray):
    """
    Returns the points on the Pareto front.
    """
    fronts = non_dominated_sort_with_buffer(points, 0.0)
    if not fronts or not fronts[0]:
        return np.array([])
    return points[fronts[0]]


class CandidateSelector:
    def __init__(self, config, logger, random_seed=42):
        self.config = config
        self.logger = logger
        # Create seeded random generator for reproducible candidate selection
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed

    def _select_by_distance(self, points, initial_selected_indices, remaining_indices, target_points):
        """
        Iteratively selects points from remaining_indices based on maximum minimum distance
        to points in initial_selected_indices until target_points are selected.
        """
        selected_indices = list(initial_selected_indices)
        current_remaining_indices = list(remaining_indices)

        while len(selected_indices) < target_points and current_remaining_indices:
            max_min_distance = -np.inf
            best_remaining_idx = -1

            for remaining_idx in current_remaining_indices:
                min_distance_to_selected = np.inf
                for selected_idx in selected_indices:
                    distance = np.linalg.norm(points[remaining_idx] - points[selected_idx])
                    min_distance_to_selected = min(min_distance_to_selected, distance)

                if min_distance_to_selected > max_min_distance:
                    max_min_distance = min_distance_to_selected
                    best_remaining_idx = remaining_idx

            if best_remaining_idx != -1:
                selected_indices.append(best_remaining_idx)
                current_remaining_indices.remove(best_remaining_idx)
            else:
                # Should not happen if current_remaining_indices is not empty, but as a safeguard
                break
        
        return selected_indices
        
    def reseed(self, new_seed):
        """Reseed the random number generator for this candidate selector."""
        self.random_seed = new_seed
        self.rng = np.random.default_rng(new_seed)
    
    def _non_dominated_sort_with_buffer(self, points: np.ndarray, accuracy_buffer: float = 0.0, token_buffer: float = 0.0):
        return non_dominated_sort_with_buffer(points, accuracy_buffer, token_buffer)

    def select_by_score(self, candidates, n, score_attr="eval_score", descending=True):
        """
        Select candidates based on a specific score attribute.
        """
        sorted_candidates = sorted(
            candidates,
            key=lambda c: getattr(c, score_attr, 0),
            reverse=descending
        )
        return sorted_candidates[:n]

    def select_by_token_length(self, candidates, n, min_length=None, max_length=None):
        """
        Select candidates based on token length.
        """
        filtered = candidates
        if min_length is not None:
            filtered = [c for c in filtered if getattr(c, "token_length", 0) >= min_length]
        if max_length is not None:
            filtered = [c for c in filtered if getattr(c, "token_length", 0) <= max_length]
        if len(filtered) <= n:
            return filtered
        # Use seeded random generator for reproducible sampling
        indices = self.rng.choice(len(filtered), size=n, replace=False)
        return [filtered[i] for i in indices]

    def select_by_performance_and_token_length(self, candidates, n, score_attr="eval_score", score_weight=0.7, token_weight=0.3):
        """
        Select candidates based on a weighted combination of evaluation performance (eval_score) and token_length.
        Here, the eval_score is retrieved from the candidate's attribute (which we want to maximize)
        and token_length is also retrieved from the candidate (which we want to minimize). Both are normalized,
        and then a weighted sum is computed.
        """
        # Extract evaluation scores and token lengths from candidates.
        eval_scores = np.array([getattr(c, score_attr, 0) for c in candidates], dtype=float)
        token_lengths = np.array([getattr(c, "token_length", 0) for c in candidates], dtype=float)
        
        # Normalize eval_scores (to maximize).
        if np.max(eval_scores) > np.min(eval_scores):
            norm_perf = (eval_scores - np.min(eval_scores)) / (np.max(eval_scores) - np.min(eval_scores))
        else:
            norm_perf = np.ones_like(eval_scores)
        
        # Normalize token lengths (to minimize; higher normalized value means shorter token length).
        if np.max(token_lengths) > np.min(token_lengths):
            norm_token = (np.max(token_lengths) - token_lengths) / (np.max(token_lengths) - np.min(token_lengths))
        else:
            norm_token = np.ones_like(token_lengths)
        
        # Compute the combined score using the provided weights.
        combined_scores = score_weight * norm_perf + token_weight * norm_token
        
        # Sort candidates by combined score (descending) and select top n.
        sorted_indices = np.argsort(-combined_scores)[:n]
        return [candidates[i] for i in sorted_indices]
    
    def _nsga2_core(
        self,
        candidates,
        perf_values: np.ndarray,
        max_points: int | None,
        min_points: int,
        progression: float,
        accuracy_buffer: float,
        token_buffer: float,
        k: float,
        log_prefix: str = "[NSGA-II]",
    ):
        """
        Shared NSGA-II selection core:
        - Objectives: (perf_values maximize, token_length minimize)
        - Buffered dominance via non_dominated_sort_with_buffer
        - If rank-1 too large: keep extremes, then use _select_by_distance
        - If rank-1 too small: fill from next fronts via _select_by_distance
        """
        token_lengths = np.array([getattr(c, "token_length", 0.0) for c in candidates], dtype=float)
        points = np.column_stack([perf_values, token_lengths])

        # natural decay (kept for parity)
        _ = math.exp(-k * progression)
        # normalized decay (0 at progression=1)
        decay = (math.exp(-k * progression) - math.exp(-k)) / (1 - math.exp(-k))

        decayed_accuracy_buffer = float(accuracy_buffer) * decay
        decayed_token_buffer    = float(token_buffer) * decay

        fronts = self._non_dominated_sort_with_buffer(
            points,
            accuracy_buffer=decayed_accuracy_buffer,
            token_buffer=decayed_token_buffer,
        )

        pareto_optimal_indices = fronts[0] if fronts and fronts[0] else []
        self.logger.log(f"{log_prefix} Number of Pareto-optimal points: {len(pareto_optimal_indices)}")
        for idx in pareto_optimal_indices:
            self.logger.log(f"{log_prefix} Point {idx}: Perf={points[idx,0]:.6f}, Tokens={points[idx,1]}")

        # Too many in rank-1: keep extremes, then maximize minimum distance
        if max_points is not None and len(pareto_optimal_indices) > max_points:
            self.logger.log(f"{log_prefix} Reducing Pareto-optimal points to {max_points} using distance-based selection.")

            current_front_points_data = points[pareto_optimal_indices]
            current_front_original_indices = list(pareto_optimal_indices)
            selected_original_indices = []

            if current_front_points_data.shape[0] > 0:
                # extreme in perf (maximize)
                max_perf_idx_in_front_data = np.argmax(current_front_points_data[:, 0])
                max_perf_original_idx = current_front_original_indices[max_perf_idx_in_front_data]
                selected_original_indices.append(max_perf_original_idx)

                # extreme in token (minimize)
                min_token_idx_in_front_data = np.argmin(current_front_points_data[:, 1])
                min_token_original_idx = current_front_original_indices[min_token_idx_in_front_data]
                if min_token_original_idx not in selected_original_indices:
                    selected_original_indices.append(min_token_original_idx)

            remaining_original_indices = [idx for idx in current_front_original_indices if idx not in selected_original_indices]
            pareto_optimal_indices = self._select_by_distance(points, selected_original_indices, remaining_original_indices, max_points)

        # Too few in rank-1: fill from next fronts using distance to already selected
        elif len(pareto_optimal_indices) < min_points:
            self.logger.log(f"{log_prefix} Increasing Pareto-optimal points to {min_points} using distance-based fill from subsequent fronts.")

            selected_original_indices = list(pareto_optimal_indices)
            for rank in fronts[1:]:
                if len(selected_original_indices) >= min_points:
                    break
                if rank:
                    selected_original_indices = self._select_by_distance(points, selected_original_indices, rank, min_points)

            if len(selected_original_indices) < min_points:
                self.logger.log(f"{log_prefix} WARNING: Could not reach minimum points target. "
                                f"Only {len(selected_original_indices)} candidates available across all fronts (target: {min_points})")

            pareto_optimal_indices = selected_original_indices

        # Stable final ordering: (perf desc, token asc)
        pareto_optimal_indices = sorted(
            set(pareto_optimal_indices),
            key=lambda idx: (points[idx, 0], -points[idx, 1]),
            reverse=True
        )
        return [candidates[i] for i in pareto_optimal_indices]

    def select_by_nsga2(
        self,
        candidates,
        score_attr="eval_score",
        max_points=None,
        min_points=1,
        progression=1,
        accuracy_buffer=0.0,
        token_buffer=0.0,
        k=1,
    ):
        """
        NSGA-II Pareto selection (score vs tokens) with distance-based truncation/fill.
        Behavior preserved; implementation deduplicated via _nsga2_core.
        """
        eval_scores = np.array([getattr(c, score_attr, 0) for c in candidates], dtype=float)
        return self._nsga2_core(
            candidates=candidates,
            perf_values=eval_scores,                 # maximize raw score
            max_points=max_points,
            min_points=min_points,
            progression=progression,
            accuracy_buffer=accuracy_buffer,
            token_buffer=token_buffer,
            k=k,
            log_prefix="[NSGA-II]",
        )

    def select_by_nsga2_lcb(
        self,
        candidates,
        score_attr: str = "eval_score",
        std_attr: str = "eval_std",
        beta: float = 1.0,             # LCB = score - beta*std
        max_points=None,
        min_points=1,
        progression=1.0,
        accuracy_buffer=0.0,
        token_buffer=0.0,
        k=1,
    ):
        """
        NSGA-II Pareto selection with LCB (score - beta*std) vs tokens,
        sharing the same non-dominated sorting and distance-based truncation/fill.
        """
        means = np.array([float(getattr(c, score_attr, 0.0)) for c in candidates], dtype=float)
        stds  = np.array([float(getattr(c, std_attr, 0.0))  for c in candidates], dtype=float)
        stds  = np.where(np.isfinite(stds), stds, 0.0)
        lcbs  = means - beta * stds

        return self._nsga2_core(
            candidates=candidates,
            perf_values=lcbs,
            max_points=max_points,
            min_points=min_points,
            progression=progression,
            accuracy_buffer=accuracy_buffer,
            token_buffer=token_buffer,
            k=k,
            log_prefix=f"[LCB-NSGA-II β={beta:.2f}]",
        )    
    
    def filter_bad_candidates(self, candidates, score_attr="eval_score", std_attr="eval_std", filter_factor=0.5, progression=0.0, min_points=1):
        """
        Remove candidates with a score by checking max and average scores using adaptive thresholding.
        """
        return filter_bad_candidates(candidates, self.logger, score_attr=score_attr, std_attr=std_attr, filter_factor=filter_factor, min_points=min_points, beta=self.config.beta, bound="lcb")

    def select_candidates(self, input_candidates, progression=1):
        """
        Select candidates from the provided input list according to the selector's method.
        The input list of candidates is used for selection, self.candidates is updated
        with the selected subset, and that subset is returned.
        
        Additional keyword arguments can be used to adjust selection parameters.
        
        Args:
            input_candidates (list): List of candidate prompt objects.
            n (int): Number of candidates to select.
            **kwargs: Extra parameters for the specific selection method. For example:
                - For "score": score_attr (default "test_score"), descending (default True)
                - For "token": min_length, max_length.
                - For "performance": eval_scores (list), performance_weight, token_weight.
        
        Returns:
            List of selected candidate prompt objects.
        """
        adjusted_beam_size = int(self.config.beam_size * (self.config.lambda_factor + (1 - self.config.lambda_factor) * progression))
        self.logger.log(f"Adjusted beam size based on progression {progression}: {adjusted_beam_size}")

        if self.config.method == "score":
            self.logger.log(f"Selecting top {self.config.beam_size} candidates by score.")
            selected = self.select_by_score(
            input_candidates,
            self.config.beam_size,
            score_attr="eval_score",
            descending=True
            )
        elif self.config.method == "token":
            self.logger.log(f"Selecting {self.config.beam_size} candidates by token length (min: {self.config.min_length}, max: {self.config.max_length}).")
            selected = self.select_by_token_length(
            input_candidates,
            self.config.beam_size,
            min_length=self.config.min_length,
            max_length=self.config.max_length
            )
        elif self.config.method == "weighted":
            self.logger.log(f"Selecting {self.config.beam_size} candidates by weighted performance (score_weight: {self.config.score_weight}, token_weight: {self.config.token_weight}).")
            selected = self.select_by_performance_and_token_length(
            input_candidates,
            self.config.beam_size,
            score_attr="eval_score",
            score_weight=self.config.score_weight,
            token_weight=self.config.token_weight
            )
        elif self.config.method == "nsga2":
            self.logger.log(f"Selecting {adjusted_beam_size} candidates using NSGA-II.")
            selected = self.select_by_nsga2(
                input_candidates,
                score_attr="eval_score",
                max_points=adjusted_beam_size,
                min_points=adjusted_beam_size,
                progression=progression,
                accuracy_buffer=self.config.accuracy_buffer,
                token_buffer=self.config.token_buffer,
                k=self.config.k
            )
        elif self.config.method == "nsga2-lcb":
            self.logger.log(f"Selecting {adjusted_beam_size} candidates using LCB-NSGA-II (beta: {self.config.beta}).")
            selected = self.select_by_nsga2_lcb(
                input_candidates,
                score_attr="eval_score",
                std_attr="eval_std",
                beta=self.config.beta,
                max_points=adjusted_beam_size,
                min_points=adjusted_beam_size,
                progression=progression,
                accuracy_buffer=self.config.accuracy_buffer,
                token_buffer=self.config.token_buffer,
                k=self.config.k
            )
        else:
            self.logger.warning(f"Unknown selection method '{self.config.method}'. Returning all input candidates.")
            selected = input_candidates

        self.logger.log(f"Selected {len(selected)} candidates.")
        return selected
    
