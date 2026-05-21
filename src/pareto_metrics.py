import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import random

class ParetoMetrics:
    """
    Tracks multi-objective trade-off metrics (accuracy ↑, tokens ↓) across rounds.
    
    Methods:
    - calculate(candidates, score_attr='score', length_attr='token_length'):
        Compute and store metrics for one round of candidates.
    - metrics_df():
        Return a pandas DataFrame summarizing all rounds.
    - plot(metric_names=None, save_prefix=None):
        Line-plot stored metrics over rounds, optionally saving each figure.
    """
    def __init__(self, output_dir, ndigits: int = 3, wandb_logger=None, random_seed: int = None):
        self.output_dir = output_dir
        self.ndigits = ndigits
        self.records = []
        self.history = {}        
        self.output_dir = output_dir
        self.wandb_logger = wandb_logger
        # Initialize seeded RNG for deterministic behavior
        self.rng = np.random.default_rng(random_seed)
        os.makedirs(self.output_dir, exist_ok=True)

    def reseed(self, new_seed):
        """Reseed the random number generator for deterministic behavior."""
        self.rng = np.random.default_rng(new_seed)

    @staticmethod
    def _is_pareto_efficient(costs: np.ndarray, maximize: list):
        """
        Determine Pareto-efficient points.
        
        Parameters:
        - costs: 2D array of shape (N, 2), columns [accuracy, tokens]
        - maximize: [bool, bool] indicating which objectives to maximize
        
        Returns:
        - boolean mask of length N, True if that row is non-dominated
        """
        # Handle empty array case
        if costs.shape[0] == 0:
            return np.array([], dtype=bool)
            
        costs = costs.copy().astype(float)
        for j, m in enumerate(maximize):
            if m:
                costs[:, j] *= -1.0
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if not is_efficient[i]:
                continue
            # any other row strictly better in all objectives?
            dominates = np.all(costs <= c, axis=1) & np.any(costs < c, axis=1)
            is_efficient[dominates] = False
        return is_efficient

    @staticmethod
    def _hypervolume_2d(scores: np.ndarray, lengths: np.ndarray, ref_point: tuple = None):
        """
        Compute 2D hypervolume (area under the Pareto front).
        
        Parameters:
        - scores: 1D array of accuracies
        - lengths: 1D array of token counts
        - ref_point: (accuracy_ref, token_ref); defaults to (0.0, 1.1*max_length)
        
        Returns:
        - float hypervolume
        """
        if ref_point is None:
            ref_point = (0.0, 100000)
        acc_ref, len_ref = ref_point
        pts = np.vstack([scores, lengths]).T
        # sort by accuracy ascending
        idx = np.argsort(pts[:, 0])
        sorted_pts = pts[idx]
        hv = 0.0
        prev_acc = acc_ref
        for acc, ln in sorted_pts:
            dx = acc - prev_acc
            dy = len_ref - ln
            if dx > 0 and dy > 0:
                hv += dx * dy
            prev_acc = acc
        return hv
    
    @staticmethod
    def _convex_hull_area(scores: np.ndarray, lengths: np.ndarray) -> float:
        """
        Compute area of the convex hull enclosing the Pareto points.
        If fewer than 3 points, returns 0.0.
        """
        pts = np.vstack([scores, lengths]).T
        if pts.shape[0] < 3:
            return 0.0
        hull = ConvexHull(pts)
        return float(hull.volume)  # in 2D, volume == area

    def calculate(self, current_round: int, candidates: list, score_attr: str = 'test_score', token_attr: str = 'token_length'):
        """
        Compute and store a suite of metrics for one round.
        
        Parameters:
        - candidates: iterable of objects with attributes for accuracy and token_length
        - score_attr: name of the accuracy attribute on each candidate
        - length_attr: name of the token count attribute
        """
        # Handle empty candidates case
        if not candidates:
            metrics = {
                'round': current_round,
                'hypervolume': 0.0,
                'convex_hull_area': 0.0,            
                'front_size': 0,
                'max_accuracy': 0.0,
                'min_tokens': 0.0,
                'accuracy_range': 0.0,
                'token_range': 0.0,
                'spacing': 0.0,
                'spread': 0.0,
                'best_efficiency': 0.0
            }
            self.records.append(metrics)
            
            # Log to wandb if logger is available
            if self.wandb_logger:
                self.wandb_logger.log_pareto_metrics(current_round, metrics, candidates, score_attr)
            
            # Save empty arrays for plotting
            self.history[current_round] = {
                'accuracy': np.array([]),
                'tokens': np.array([])
            }
            return
        
        # Extract arrays
        eval_scores = np.array([getattr(c, score_attr, 0) for c in candidates], dtype=float)
        token_lengths = np.array([getattr(c, token_attr, 0) for c in candidates], dtype=float)
        
        # Pareto mask
        costs = np.vstack([eval_scores, token_lengths]).T
        mask = self._is_pareto_efficient(costs, maximize=[True, False])
        
        # Core metrics
        hv = self._hypervolume_2d(eval_scores[mask], token_lengths[mask])
        hull_area = self._convex_hull_area(eval_scores[mask], token_lengths[mask])
        front_size = int(mask.sum())
        max_acc = float(eval_scores.max()) if eval_scores.size > 0 else 0.0
        min_tok = float(token_lengths.min()) if token_lengths.size > 0 else 0.0
        acc_range = float(np.ptp(eval_scores)) if eval_scores.size > 0 else 0.0
        tok_range = float(np.ptp(token_lengths)) if token_lengths.size > 0 else 0.0
        
        # Spacing & Spread
        idx_sorted = np.argsort(eval_scores)
        diffs = np.vstack([
            np.diff(eval_scores[idx_sorted]),
            np.diff(token_lengths[idx_sorted])
        ]).T
        dists = np.linalg.norm(diffs, axis=1)
        spacing = float(dists.std()) if len(dists) > 1 else 0.0
        d_mean = float(dists.mean()) if len(dists) > 0 else 0.0
        spread = float((np.abs(dists - d_mean).sum() / (len(dists) * d_mean))) if d_mean > 0 else 0.0
        
        # Efficiency (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiencies = np.divide(eval_scores, token_lengths, out=np.zeros_like(eval_scores), where=token_lengths!=0) * 100
        best_eff = float(efficiencies.max()) if efficiencies.size > 0 and not np.isnan(efficiencies).all() else 0.0
        
        # Round, collect, and store
        metrics = {
            'round': current_round,
            'hypervolume': round(hv, self.ndigits),
            'convex_hull_area': round(hull_area, self.ndigits),            
            'front_size': front_size,
            'max_accuracy': round(max_acc, self.ndigits),
            'min_tokens': round(min_tok, self.ndigits),
            'accuracy_range': round(acc_range, self.ndigits),
            'token_range': round(tok_range, self.ndigits),
            'spacing': round(spacing, self.ndigits),
            'spread': round(spread, self.ndigits),
            'best_efficiency': round(best_eff, self.ndigits)
        }
        self.records.append(metrics)
        
        # Log to wandb if logger is available
        if self.wandb_logger:
            self.wandb_logger.log_pareto_metrics(current_round, metrics, candidates, score_attr)
        
      # Save Pareto points for plotting
        self.history[current_round] = {
            'accuracy': eval_scores,
            'tokens': token_lengths
        }

    def metrics_df(self) -> pd.DataFrame:
        """Return a DataFrame of all stored metrics by round."""
        return pd.DataFrame(self.records)

    def add_jitter(
        self,
        x,
        y,
        strength: float = 0.05,
        *,
        x_range_override: float | None = None,
        y_range_override: float | None = None,
        rng=None,
    ):
        """
        Add small random jitter to handle overlapping points.

        Jitter scales with the (global) data range, not per-round range.
        With default strength=0.05 and typical ranges, this yields roughly:
        - tokens: ±5 (1σ ≈ 5)
        - accuracy: ±0.0003 (1σ ≈ 3e-4)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) == 0:
            return x, y

        # Use provided global ranges if available (recommended),
        # else fall back to per-array range.
        x_rng = x_range_override if x_range_override is not None else (np.ptp(x) if np.ptp(x) > 0 else 1.0)
        y_rng = y_range_override if y_range_override is not None else (np.ptp(y) if np.ptp(y) > 0 else 1.0)

        # Coefficients tuned so that strength=0.05 ≈ 1.5% of token range (≈ ±5 on a ~360 range),
        # and ≈ 0.6% of accuracy range (≈ ±3e-4 on a ~0.055 range).
        x_sigma = 0.30 * strength * x_rng   # 0.30 * 0.05 = 0.015 of range
        y_sigma = 0.12 * strength * y_rng   # 0.12 * 0.05 = 0.006 of range

        # Use provided rng or fall back to instance rng
        rng_to_use = rng if rng is not None else self.rng
        jitter_x = rng_to_use.normal(0, x_sigma, len(x))
        jitter_y = rng_to_use.normal(0, y_sigma, len(y))
        return x + jitter_x, y + jitter_y
    
    def plot(self, metric_names: list = None, max_rounds_display: int = 20,
         use_distinct_colors: bool = True, jitter_strength: float = 0.05,
         font_scale: float = 1.5):
        """
        Advanced plotting method with bold, high-visibility markers and a marker-cycling policy:
        • First full cycle uses markers in a fixed order (no randomization).
        • Each subsequent full cycle randomizes the marker order once, then uses sequentially.
        """
        # Font size configuration (base sizes that can be scaled)
        base_title_size = 16
        base_label_size = 14
        base_tick_size = 12
        base_legend_size = 10

        # Apply font scale
        title_fontsize = int(base_title_size * font_scale)
        label_fontsize = int(base_label_size * font_scale)
        tick_fontsize = int(base_tick_size * font_scale)
        legend_fontsize = int(base_legend_size * font_scale)

        df = self.metrics_df()
        if df.empty:
            raise ValueError("No metrics available. Call `calculate(...)` first.")

        # Save detailed report
        df.to_csv(os.path.join(self.output_dir, "pareto_report.tsv"), sep='\t', index=False)

        if metric_names is None:
            metric_names = [c for c in df.columns if c != 'round']

        # Per-metric line plots over rounds
        for name in metric_names:
            plt.figure(figsize=(10, 5))
            plt.plot(df['round'], df[name], marker='o', linewidth=2, markersize=6)
            plt.xlabel('Round', fontsize=label_fontsize)
            plt.ylabel(name.replace('_', ' ').title(), fontsize=label_fontsize)
            plt.title(f'{name.replace("_", " ").title()} Over Rounds', fontsize=title_fontsize)
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            plt.savefig(f'{self.output_dir}/{name}.png', bbox_inches='tight', dpi=300)
            plt.close()

        # Combined scatter of accuracy vs tokens across rounds
        num_rounds = len(self.history)
        if num_rounds == 0:
            return

        base_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '8', 'd', 'P', 'X']

        class MarkerCycler:
            def __init__(self, markers, rng=random):
                self._markers = list(markers)
                self._pool = list(markers)
                self._first_cycle = True
                self._rng = rng
            def next(self):
                if not self._pool:
                    self._pool = list(self._markers)
                    if self._first_cycle:
                        self._first_cycle = False
                    else:
                        self._rng.shuffle(self._pool)
                return self._pool.pop(0)

        marker_cycler = MarkerCycler(base_markers, rng=self.rng)

        def get_distinct_colors(n_colors: int):
            if n_colors == 7:
                palette = ['#808080','#FFBE0B','#FA7921','#D7263D','#B5179E','#3A86FF','#02C39A']
            elif n_colors == 9:
                palette = ['#808080','#FFBE0B','#FA7921','#E55934','#D7263D','#B5179E','#3A86FF','#118AB2','#02C39A']
            elif n_colors == 13:
                palette = ['#808080','#FFBE0B','#F6AE2D','#FA7921','#E55934','#D7263D','#FF006E','#B5179E','#8338EC','#3A86FF','#118AB2','#06D6A0','#9BC53D']
            elif n_colors == 20:
                palette = ['#808080','#FFBE0B','#F6AE2D','#FF9F1C','#FA7921','#E55934','#D7263D','#FF006E','#B5179E','#8338EC','#6A4C93','#3A86FF','#1982C4','#5BC0EB','#6FFFE9','#06D6A0','#02C39A','#00A896','#8AC926','#9BC53D']
            elif n_colors == 25:
                palette = ['#808080','#FFBE0B','#F6AE2D','#FF9F1C','#FA7921','#E55934','#D7263D','#FF4D6D','#FF006E','#F72585','#B5179E','#7209B7','#8338EC','#6A4C93','#3A0CA3','#3A86FF','#1982C4','#118AB2','#5BC0EB','#6FFFE9','#06D6A0','#02C39A','#00A896','#8AC926','#9BC53D']
            else:
                import colorsys
                palette = ['#808080']
                for i in range(1, n_colors):
                    hue = i / (n_colors - 1)
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
                    palette.append('#{:02X}{:02X}{:02X}'.format(int(r*255), int(g*255), int(b*255)))
            return palette

        def scatter_bold(x, y, label, marker, color=None):
            kw = dict(s=220, linewidths=1.8, alpha=0.95, zorder=3, edgecolor='k')
            if color is not None:
                kw['color'] = color
            return plt.scatter(x, y, marker=marker, label=label, **kw)

        # Collect raw rows (already present) and new jittered rows for auditing
        scatter_rows = []
        jittered_rows = []

        # Compute GLOBAL raw ranges across all rounds (to stabilize jitter magnitude)
        all_tokens_raw = []
        all_scores_raw = []
        for pts in self.history.values():
            if len(pts['tokens']) > 0:
                all_tokens_raw.extend(pts['tokens'])
                all_scores_raw.extend(pts['accuracy'])

        global_x_range = (max(all_tokens_raw) - min(all_tokens_raw)) if all_tokens_raw else 1.0
        global_y_range = (max(all_scores_raw) - min(all_scores_raw)) if all_scores_raw else 1.0

        # For axis limits, compute from jittered values so nothing is clipped
        all_j_tokens = []
        all_j_scores = []

        plt.figure(figsize=(12, 8))
        plt.grid(True, alpha=0.25, zorder=0)

        if num_rounds <= max_rounds_display:
            colors = get_distinct_colors(num_rounds) if use_distinct_colors else None

            for i, (rnd, pts) in enumerate(self.history.items()):
                scores = pts['accuracy']
                tokens = pts['tokens']
                if len(scores) == 0:
                    continue

                for tok, sc in zip(tokens, scores):
                    scatter_rows.append({"round": rnd, "token": tok, "accuracy": sc})

                marker = marker_cycler.next()
                color = colors[i] if (use_distinct_colors and i < len(colors)) else None

                # Jitter with corrected scaling and GLOBAL ranges
                if jitter_strength > 0:
                    jittered_tokens, jittered_scores = self.add_jitter(
                        tokens, scores, jitter_strength,
                        x_range_override=global_x_range,
                        y_range_override=global_y_range,
                    )
                else:
                    jittered_tokens, jittered_scores = tokens, scores

                # Track jittered for axis + auditing TSV
                all_j_tokens.extend(jittered_tokens.tolist())
                all_j_scores.extend(jittered_scores.tolist())
                for tok_raw, sc_raw, tok_j, sc_j in zip(tokens, scores, jittered_tokens, jittered_scores):
                    jittered_rows.append({
                        "round": rnd,
                        "token_raw": float(tok_raw),
                        "accuracy_raw": float(sc_raw),
                        "token_jittered": float(tok_j),
                        "accuracy_jittered": float(sc_j),
                    })

                scatter_bold(jittered_tokens, jittered_scores, label=f'Round {rnd}', marker=marker, color=color)

            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=legend_fontsize)

        else:
            step = max(1, num_rounds // max_rounds_display)
            displayed_rounds = list(self.history.keys())[::step]
            colors = get_distinct_colors(len(displayed_rounds)) if use_distinct_colors else None

            for i, rnd in enumerate(displayed_rounds):
                pts = self.history[rnd]
                scores = pts['accuracy']
                tokens = pts['tokens']
                if len(scores) == 0:
                    continue

                for tok, sc in zip(tokens, scores):
                    scatter_rows.append({"round": rnd, "token": tok, "accuracy": sc})

                marker = marker_cycler.next()
                color = colors[i] if (use_distinct_colors and i < len(colors)) else None

                # Jitter with corrected scaling and GLOBAL ranges (fixed here too)
                if jitter_strength > 0:
                    jittered_tokens, jittered_scores = self.add_jitter(
                        tokens, scores, jitter_strength,
                        x_range_override=global_x_range,
                        y_range_override=global_y_range,
                    )
                else:
                    jittered_tokens, jittered_scores = tokens, scores

                # Track jittered for axis + auditing TSV
                all_j_tokens.extend(jittered_tokens.tolist())
                all_j_scores.extend(jittered_scores.tolist())
                for tok_raw, sc_raw, tok_j, sc_j in zip(tokens, scores, jittered_tokens, jittered_scores):
                    jittered_rows.append({
                        "round": rnd,
                        "token_raw": float(tok_raw),
                        "accuracy_raw": float(sc_raw),
                        "token_jittered": float(tok_j),
                        "accuracy_jittered": float(sc_j),
                    })

                scatter_bold(jittered_tokens, jittered_scores, label=f'Round {rnd}', marker=marker, color=color)

            plt.legend(title=f'Showing every {step} rounds', bbox_to_anchor=(1.05, 1),
                    loc='upper left', fontsize=legend_fontsize)

        # Axis limits based on jittered data so jittered points aren't clipped
        if all_j_tokens:
            token_min, token_max = min(all_j_tokens), max(all_j_tokens)
            score_min, score_max = min(all_j_scores), max(all_j_scores)

            token_range = token_max - token_min
            score_range = score_max - score_min
            token_padding = max(token_range * 0.05, 10)   # At least 10 tokens padding
            score_padding = max(score_range * 0.05, 0.01)

            # Inverted x-axis (high→low) preserved
            plt.xlim(token_max + token_padding, token_min - token_padding)
            plt.ylim(score_min - score_padding, score_max + score_padding)

            # Ticks across jittered range
            num_token_ticks = 6
            token_ticks = np.linspace(token_min, token_max, num_token_ticks)
            token_ticks = [int(round(t)) for t in token_ticks]
            plt.xticks(token_ticks)

            num_score_ticks = 6
            score_ticks = np.linspace(score_min, score_max, num_score_ticks)
            score_ticks = [round(s, 3) for s in score_ticks]
            plt.yticks(score_ticks)

        plt.xlabel('Token Count', fontsize=label_fontsize)
        plt.ylabel('Accuracy', fontsize=label_fontsize)
        plt.title(f'Pareto Front Evolution Across {num_rounds} Rounds', fontsize=title_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/combined_pareto.png', bbox_inches='tight', dpi=300)
        plt.close()

        # RAW points TSV (unchanged behavior)
        points_path = os.path.join(self.output_dir, "combined_pareto_points.tsv")
        with open(points_path, "w", encoding="utf-8") as f:
            f.write("round\ttoken\taccuracy\n")
            for r in scatter_rows:
                f.write(f"{r['round']}\t{r['token']}\t{r['accuracy']}\n")

        # Add jittered TSV for auditing what was actually plotted
        jittered_points_path = os.path.join(self.output_dir, "combined_pareto_points_jittered.tsv")
        with open(jittered_points_path, "w", encoding="utf-8") as f:
            f.write("round\ttoken_raw\taccuracy_raw\ttoken_jittered\taccuracy_jittered\n")
            for r in jittered_rows:
                f.write(f"{r['round']}\t{r['token_raw']:.4f}\t{r['accuracy_raw']:.4f}\t{r['token_jittered']:.4f}\t{r['accuracy_jittered']:.4f}\n")