"""Lightweight memory diagnostics — call once per round to surface where
memory is going.

Usage:
    from memory_diagnostic import log_round_memory
    log_round_memory(orchestrator, current_round, logger)

The helper:
1. Reports process RSS (and peak via tracemalloc if started).
2. Walks specific orchestrator-owned data structures and prints their size:
   - expansion_graph (nodes dict + edges list)
   - top_candidates list
   - per-candidate error_evaluations cumulative size
   - LLMCache _write_buffer size
3. Captures a tracemalloc snapshot diff vs the previous round so we can see
   per-round allocation deltas grouped by file:line.

Cost is small (~50 ms per round) and disabled if tracemalloc is not started.
"""
from __future__ import annotations

import gc
import os
import sys
import tracemalloc
from collections import defaultdict
from typing import Any


_PREV_SNAPSHOT: tracemalloc.Snapshot | None = None


def _rss_mb() -> float:
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: ru_maxrss is bytes, Linux: kilobytes. Heuristic: > 1e9 → bytes.
        rss = ru.ru_maxrss
        return rss / (1024 * 1024) if rss > 10**8 else rss / 1024
    except Exception:
        try:
            import psutil  # type: ignore
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            return -1.0


def _deep_size(obj: Any, seen: set[int] | None = None) -> int:
    """Recursive sizeof — handles dicts, lists, sets, tuples, dataclasses."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(_deep_size(k, seen) + _deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_deep_size(item, seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        size += _deep_size(vars(obj), seen)
    elif hasattr(obj, "__slots__"):
        size += sum(_deep_size(getattr(obj, s, None), seen) for s in obj.__slots__)
    return size


def _safe_attr_size(obj: Any, attr: str) -> int:
    try:
        return _deep_size(getattr(obj, attr, None))
    except Exception:
        return -1


def log_round_memory(orchestrator: Any, current_round: int, logger: Any) -> None:
    """Print a one-block memory report for the round."""
    global _PREV_SNAPSHOT

    gc.collect()  # reduce noise from stale references
    rss = _rss_mb()
    line = lambda s: logger.log(f"[MEM R{current_round}] {s}")
    line(f"RSS: {rss:.1f} MB")

    # Specific structures of interest
    try:
        eg = getattr(orchestrator, "expansion_graph", None)
        if eg is not None:
            n_nodes = len(getattr(eg, "nodes", {}) or {})
            n_edges = len(getattr(eg, "edges", []) or [])
            sz = _deep_size(eg)
            line(f"expansion_graph: {n_nodes} nodes, {n_edges} edges, "
                 f"~{sz/(1024*1024):.2f} MB deep")
    except Exception as e:
        line(f"expansion_graph sizing failed: {e}")

    try:
        tc = getattr(orchestrator, "top_candidates", None) or []
        n_tc = len(tc)
        sz_tc = _deep_size(tc)
        line(f"top_candidates: {n_tc} items, ~{sz_tc/(1024*1024):.2f} MB deep")
    except Exception as e:
        line(f"top_candidates sizing failed: {e}")

    # Sum error_evaluations across all known candidates (top + graph nodes)
    try:
        known = []
        if hasattr(orchestrator, "top_candidates"):
            known += list(orchestrator.top_candidates or [])
        eg = getattr(orchestrator, "expansion_graph", None)
        if eg is not None:
            known += list(getattr(eg, "nodes", {}).values() or [])
        ee_total = 0
        ee_count = 0
        for c in known:
            ee = getattr(c, "error_evaluations", None)
            if ee is not None:
                ee_total += _deep_size(ee)
                ee_count += len(ee) if hasattr(ee, "__len__") else 1
        line(f"error_evaluations across all known candidates: "
             f"{ee_count} entries, ~{ee_total/(1024*1024):.2f} MB deep")
    except Exception as e:
        line(f"error_evaluations sizing failed: {e}")

    # LLMCache _write_buffer
    try:
        from llm_cache import get_cache
        cache = get_cache()
        wb = getattr(cache, "_write_buffer", None) if cache else None
        if wb is not None:
            line(f"LLMCache._write_buffer: {len(wb)} entries, "
                 f"~{_deep_size(wb)/(1024*1024):.2f} MB deep")
    except Exception as e:
        line(f"LLMCache sizing failed: {e}")

    # Validation-related collections held on the eval_manager
    try:
        em = getattr(orchestrator, "eval_manager", None)
        if em is not None:
            for attr in ("global_df", "validation_subsets", "validation_subsets_metadata"):
                v = getattr(em, attr, None)
                if v is not None:
                    line(f"eval_manager.{attr}: ~{_deep_size(v)/(1024*1024):.2f} MB deep")
    except Exception as e:
        line(f"eval_manager sizing failed: {e}")

    # Tracemalloc diff
    if tracemalloc.is_tracing():
        try:
            snap = tracemalloc.take_snapshot()
            if _PREV_SNAPSHOT is not None:
                stats = snap.compare_to(_PREV_SNAPSHOT, "lineno")
                line("--- tracemalloc top 8 lines (this round delta) ---")
                for s in stats[:8]:
                    line(f"  {s.size_diff/1024:+.1f} KB  ({s.count_diff:+d} blocks)  "
                         f"{s.traceback.format()[-1].strip()}")
            _PREV_SNAPSHOT = snap
        except Exception as e:
            line(f"tracemalloc diff failed: {e}")
