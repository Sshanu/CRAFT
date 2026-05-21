from tqdm import tqdm
import random
import numpy as np
import math
import os
import utils
from candidate_selector import get_pareto_front_points, filter_bad_candidates

class SimpleCandidate:
    """Simple candidate object for compatibility with filter_bad_candidates function"""
    def __init__(self, eval_score, token_length, index, eval_std=0.0):
        self.eval_score = eval_score
        self.token_length = token_length
        self.index = index
        self.eval_std = eval_std

def compute_mo_ucb_priority(
    perf_mean,
    perf_std,
    prompt_lengths,
    pareto_front_points,
    metric: str = "ucb",   # "ucb" | "lcb" | "mean"
    beta: float = 1.0,     # used for both UCB and LCB
):
    """
    Gap-to-Pareto score in a consistent metric space.
    Lower scores = better (closer to PF).
    """
    if metric == "ucb":
        perf_metric = perf_mean + beta * perf_std
    elif metric == "lcb":
        perf_metric = perf_mean - beta * perf_std
    elif metric == "mean":
        perf_metric = perf_mean
    else:
        raise ValueError("metric must be 'ucb' | 'lcb' | 'mean'")

    if pareto_front_points.size == 0:
        return np.zeros(len(perf_metric))

    # normalize + clip in the same metric
    min_perf, max_perf = float(np.min(perf_metric)), float(np.max(perf_metric))
    min_len,  max_len  = float(np.min(prompt_lengths)), float(np.max(prompt_lengths))
    perf_range = max(max_perf - min_perf, 1e-9)
    len_range  = max(max_len - min_len, 1e-9)

    norm_perf = np.clip((perf_metric - min_perf) / perf_range, 0.0, 1.0)
    norm_len  = np.clip((prompt_lengths - min_len) / len_range, 0.0, 1.0)
    norm_pf   = [(
        np.clip((p[0] - min_perf) / perf_range, 0.0, 1.0),
        np.clip((p[1] - min_len)  / len_range,  0.0, 1.0),
    ) for p in pareto_front_points]

    scores = []
    for i in range(len(norm_perf)):
        pt = (norm_perf[i], norm_len[i])
        best = float("inf")
        for pf in norm_pf:
            perf_gap = max(0.0, pf[0] - pt[0])  # want perf ≥ PF
            len_gap  = max(0.0, pt[1] - pf[1])  # want length ≤ PF
            gap = perf_gap + len_gap
            if gap < best:
                best = gap
        scores.append(best)
    return np.array(scores)

class UCBBandits:
    """Upper Confidence Bound Bandits."""

    def __init__(self, logger, num_prompts, token_lengths, num_samples=5, c=1.0, mode='ucb', random_seed=42, score_weight=0.7, token_weight=0.3):
        self.logger = logger
        self.c = c
        assert mode in {'ucb', 'ucb-e', 'ucb-rs', 'ucb-mo', 'ucb-ws'}, f"Unknown mode: {mode}"
        self.mode = mode
        self.num_prompts = num_prompts
        self.num_samples = num_samples
        self.token_lengths = np.array(token_lengths, dtype=float) if token_lengths is not None else None
        # Create seeded random generator for reproducible behavior
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.score_weight = score_weight
        self.token_weight = token_weight
        self.reset()

    def update(self, chosen, scores):
        for i, score in zip(chosen, scores):
            self.prompt_scores[i].append(score)

    def reset(self):
        self.prompt_scores = [[] for _ in range(self.num_prompts)]

    def get_scores(self):
        means = [np.mean(s) if s else 0 for s in self.prompt_scores]
        return np.array(means)

    def get_std(self):
        stds = []
        for s in self.prompt_scores:
            if len(s) > 1:
                # Multiple samples: use actual standard deviation
                stds.append(np.std(s))
            elif len(s) == 1:
                # Single sample: use high uncertainty to encourage exploration
                # Use the absolute value of the score as a proxy for uncertainty
                # This encourages exploration when we have limited data
                stds.append(max(abs(s[0]), 1.0))  # Minimum uncertainty of 1.0
            else:
                # No samples: maximum uncertainty
                stds.append(1.0)
        return np.array(stds)
    
    def _normalize(self, data):
        """Normalizes data to a [0, 1] range."""
        min_val, max_val = np.min(data), np.max(data)
        range_val = max_val - min_val
        if range_val == 0:
            return np.zeros_like(data, dtype=float)
        return (data - min_val) / range_val

    def choose(self, n, t, max_t=None):
        counts = np.array([len(s) for s in self.prompt_scores])
        if np.sum(counts) == 0:
            return list(range(self.num_prompts))

        scores = self.get_scores()
        exploration_counts = counts + 1e-3
        exploration_bonus = self.c * np.sqrt(np.log(t) / exploration_counts)

        if self.mode == 'ucb':
            ucb_scores = scores + exploration_bonus
            return np.argsort(ucb_scores)[::-1][:n]

        elif self.mode == 'ucb-e':
            ucb_scores = scores + self.c * np.sqrt(self.c / exploration_counts)
            return np.argsort(ucb_scores)[::-1][:n]

        elif self.mode == 'ucb-rs':
            if self.token_lengths is None:
                raise ValueError("token_lengths must be provided for ucb-rs mode")

            norm_scores = self._normalize(scores)
            norm_token_lengths = self._normalize(self.token_lengths)
            
            # Use seeded random generator instead of global random.uniform
            weight = self.rng.uniform(-0.5, 0.5)
            adjusted_weight = 0.5 + weight * max(0, 1 - t / max_t)
            self.logger.log(f"UCB-RS: t={t}, max_t={max_t}, weight={weight:.3f}, adjusted_weight={adjusted_weight:.3f}")

            combined_scores = norm_scores - adjusted_weight * norm_token_lengths + exploration_bonus
            return np.argsort(combined_scores)[::-1][:n]
        
        elif self.mode == 'ucb-mo':
            if self.token_lengths is None:
                raise ValueError("token_lengths must be provided for ucb-mo mode")

            # beta for upper confidence bound
            assert max_t is not None, "max_t must be provided for ucb-mo mode"
            beta_val = 0.5 + 0.5 * max(0, 1 - t / max_t)   # ∈ [0.5,1.0]

            perf_mean = scores
            perf_std  = self.get_std()

            temp_candidates = [
                SimpleCandidate(eval_score=perf_mean[i], token_length=self.token_lengths[i], index=i, eval_std=perf_std[i])
                for i in range(len(perf_mean))
            ]

            # Choose the bound used by the filter; here: optimistic prefilter on UCB
            filtered_candidates = filter_bad_candidates(
                temp_candidates, self.logger,
                score_attr="eval_score", std_attr="eval_std",
                filter_factor=0.5, min_points=n,
                bound="ucb", beta=beta_val  # <-- one knob
            )
            if len(filtered_candidates) == 0:
                self.logger.log("UCB-MO: No candidates left after filtering, falling back to all candidates")
                filtered_candidates = temp_candidates

            f_mu  = np.array([c.eval_score for c in filtered_candidates], dtype=float)
            f_sd  = np.array([c.eval_std   for c in filtered_candidates], dtype=float)
            f_len = np.array([c.token_length for c in filtered_candidates], dtype=float)

            # Using UCB for Pareto Front construction to encourage exploration
            filtered_perf_metric = f_mu + beta_val * f_sd
            filtered_points = np.column_stack([filtered_perf_metric, f_len])
            pareto_front_points = get_pareto_front_points(filtered_points)

            self.logger.log(
                f"UCB-MO: t={t}, max_t={max_t}, beta={beta_val:.3f}, "
                f"num_pareto_points={len(pareto_front_points)}, candidates_after_filter={len(f_mu)}"
            )

            # Score ALL candidates in the same metric space
            mo_scores = compute_mo_ucb_priority(
                perf_mean=perf_mean,
                perf_std=perf_std,
                prompt_lengths=self.token_lengths,
                pareto_front_points=pareto_front_points,
                metric="ucb",   # or "lcb" / "mean"
                beta=beta_val,
            )

            # Restrict exploration to filtered set
            allowed = np.zeros(len(perf_mean), dtype=bool)
            allowed[[c.index for c in filtered_candidates]] = True
            mo_scores[~allowed] = np.inf

            # Lower is better
            return np.argsort(mo_scores)[:n]


        elif self.mode == 'ucb-ws':
            """
            Weighted-sum objective over performance (to maximize) and token length (to minimize).
            Uses UCB-style performance (mean + exploration_bonus) as the 'eval_score'.
            """
            if self.token_lengths is None:
                raise ValueError("token_lengths must be provided for ucb-ws mode")

            # UCB-style performance term
            eval_scores = scores + exploration_bonus
            token_lengths = self.token_lengths.astype(float)

            # Normalize eval_scores (maximize)
            max_eval, min_eval = float(np.max(eval_scores)), float(np.min(eval_scores))
            if max_eval > min_eval:
                norm_perf = (eval_scores - min_eval) / (max_eval - min_eval)
            else:
                norm_perf = np.ones_like(eval_scores)

            # Normalize token_lengths (minimize; larger normalized value = shorter prompt)
            max_len, min_len = float(np.max(token_lengths)), float(np.min(token_lengths))
            if max_len > min_len:
                norm_token = (max_len - token_lengths) / (max_len - min_len)
            else:
                norm_token = np.ones_like(token_lengths)

            combined_scores = self.score_weight * norm_perf + self.token_weight * norm_token

            # Higher combined_scores are better
            return np.argsort(combined_scores)[::-1][:n]
        
        else:
            # Should not reach here due to assert in __init__
            return np.argsort(scores)[::-1][:n]

    def reseed(self, new_seed):
        """Reseed the random number generator for this bandit."""
        self.random_seed = new_seed
        self.rng = np.random.default_rng(new_seed)
    
    def get_infos(self):
        return np.array([len(s) for s in self.prompt_scores])