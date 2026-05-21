"""High-performance persistent cache for LLM inputs and outputs.

This module implements a SQLite-backed cache that can comfortably handle
millions of prompt/response pairs across multiple runs while staying
thread-safe and providing detailed usage visibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import utils

CacheEntry = Tuple[str, str, Any]


class _ShardInfo:
    def __init__(self, key: str, provider: str, path: Path, conn: sqlite3.Connection, fts_available: bool) -> None:
        self.key = key
        self.provider = provider
        self.path = path
        self.conn = conn
        self.fts_available = fts_available

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


class LLMCache:
    """SQLite-backed cache for LLM requests and responses."""

    def __init__(
        self,
        cache_dir: str = ".llm_cache",
        enabled: bool = True,
        random_seed: Optional[int] = None,
        refresh_cache: bool = False,
        task_name: Optional[str] = None,
        max_cache_size_mb: int = 5120,
        max_response_size_kb: int = 2048,
        max_file_locks: int = 1000,
        max_memory_entries: int = 10000,
        shard_by_seed: bool = True,
        shard_by_provider: bool = True,
        logger: Optional[Any] = None,
    ) -> None:
        self.enabled = enabled
        self.random_seed = random_seed
        self.refresh_cache = refresh_cache
        self.task_name = task_name or "default"
        self.max_cache_size_bytes = int(max_cache_size_mb) * 1024 * 1024
        self.max_response_size_bytes = int(max_response_size_kb) * 1024
        self.max_memory_entries = max(0, int(max_memory_entries))
        self.max_file_locks = max_file_locks  # kept for backward compatibility
        self.shard_by_seed = shard_by_seed
        self.shard_by_provider = shard_by_provider
        self.logger = self._init_logger(logger)
        self.stats_log_every = 200  # emit stats every N cache calls

        self.cache_dir = Path(cache_dir) / self.task_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._seed_bucket = self._seed_directory_name()

        self._lock = threading.RLock()
        self._memory_lock = threading.RLock()
        self._db_lock = threading.RLock()
        self._memory_cache: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "total_calls": 0, "memory_hits": 0, "disk_hits": 0}

        self._shards: Dict[str, "_ShardInfo"] = {}

        self._write_buffer: List[Tuple[str, tuple]] = []
        self._flush_lock = threading.RLock()
        self._last_flush_time = time.time()
        # 15s instead of 60s: under high concurrency (12 EVAL_THREADS issuing
        # thousands of writes between flushes) the buffer can grow to 10K+
        # entries, adding GB-scale transient memory pressure. 15s keeps the
        # buffer bounded without thrashing the on-disk SQLite shards.
        self._flush_interval = 15.0
        
        # Start background flush thread
        self._stop_flush_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

        self._log_cache_overview()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_logger(self, logger: Optional[Any]) -> Any:
        if isinstance(logger, utils.Logger):
            return logger
        if logger is not None:
            return logger
        py_logger = logging.getLogger("llm_cache")
        if not py_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            py_logger.addHandler(handler)
        py_logger.setLevel(logging.INFO)
        return py_logger

    def _log(self, message: str, level: str = "info", print_screen: Optional[bool] = None) -> None:
        if not message:
            return

        formatted = f"[LLMCache] {message}"
        level = level.lower()

        if isinstance(self.logger, utils.Logger):
            # utils.Logger has a simple log(message, print_screen=True) API
            self.logger.log(formatted, print_screen=True if print_screen is None else print_screen)
            return

        # Fallback to Python logging or simple print if needed
        try:
            log_method = getattr(self.logger, level)
        except AttributeError:
            log_method = getattr(self.logger, "info", None)

        if callable(log_method):
            log_method(formatted)  # type: ignore[misc]
        else:
            print(formatted)

    def _initialise_database(self, db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache_entries (
                    cache_key           TEXT PRIMARY KEY,
                    system_prompt       TEXT,
                    user_message        TEXT,
                    history_json        TEXT,
                    model               TEXT,
                    provider            TEXT,
                    parameters_json     TEXT,
                    response            TEXT,
                    token_usage_json    TEXT,
                    created_at          REAL NOT NULL,
                    updated_at          REAL NOT NULL,
                    last_accessed_at    REAL,
                    hit_count           INTEGER NOT NULL DEFAULT 0
                ) WITHOUT ROWID
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_provider ON llm_cache_entries(model, provider)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_last_access ON llm_cache_entries(last_accessed_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache_metadata (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
        return conn

    def _ensure_fts(self, conn: sqlite3.Connection) -> bool:
        try:
            with conn:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS llm_cache_entries_fts USING fts5("
                    "cache_key, system_prompt, user_message, response)"
                )
            return True
        except sqlite3.OperationalError:
            self._log("SQLite FTS5 not available; falling back to LIKE-based search", level="warning")
            return False

    def _log_cache_overview(self) -> None:
        if not self.enabled:
            self._log("Cache initialised but disabled", level="info")
            return

        sqlite_files = list(self.cache_dir.glob("**/llm_cache.sqlite3"))
        size_mb = 0.0
        for path in sqlite_files:
            try:
                size_mb += os.path.getsize(path)
            except OSError:
                continue
        self._log(
            f"initialised cache root at {self.cache_dir} | shards={len(sqlite_files)} | total_size={size_mb:.2f} MB | "
            f"shard_by_seed={self.shard_by_seed} shard_by_provider={self.shard_by_provider}",
            level="info",
        )

    def _normalise_history(self, history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if not history:
            return []
        normalised: List[Dict[str, str]] = []
        for turn in history:
            if not isinstance(turn, dict):
                continue
            normalised.append({
                "User": str(turn.get("User", "")),
                "Assistant": str(turn.get("Assistant", "")),
            })
        return normalised

    def _parameter_payload(
        self,
        model: str,
        provider: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> Dict[str, Any]:
        return {
            "model": str(model),
            "provider": str(provider),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "frequency_penalty": float(frequency_penalty),
            "presence_penalty": float(presence_penalty),
            "random_seed": self.random_seed,
        }

    def _hash_request(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]],
        model: str,
        provider: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> str:
        payload = {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "history": self._normalise_history(history),
            "parameters": self._parameter_payload(
                model,
                provider,
                max_tokens,
                temperature,
                top_p,
                frequency_penalty,
                presence_penalty,
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _seed_directory_name(self) -> Optional[str]:
        if not self.shard_by_seed:
            return None
        seed_value = self.random_seed if self.random_seed is not None else "none"
        return f"seed_{seed_value}"

    def _provider_slug(self, provider: str) -> str:
        if not self.shard_by_provider:
            return "provider_all"
        provider = provider or "unknown"
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in provider)
        slug = "-".join(filter(None, slug.split("-"))) or "provider"
        digest = hashlib.sha1(provider.encode("utf-8")).hexdigest()[:8]
        return f"provider_{slug}-{digest}"

    def _shard_key(self, provider: str) -> str:
        seed_part = self._seed_bucket_identifier()
        provider_part = self._provider_slug(provider) if self.shard_by_provider else "provider_all"
        return f"{seed_part}|{provider_part}"

    def _seed_bucket_identifier(self) -> str:
        if not self.shard_by_seed:
            return "seed_all"
        return f"seed_{self.random_seed if self.random_seed is not None else 'none'}"

    def _db_path_for(self, provider: str) -> Path:
        parts = [self.cache_dir]
        if self._seed_bucket:
            parts.append(self._seed_bucket)
        if self.shard_by_provider:
            parts.append(self._provider_slug(provider))
        return Path(*parts) / "llm_cache.sqlite3"

    def _get_shard(self, provider: str) -> _ShardInfo:
        key = self._shard_key(provider)
        if key in self._shards:
            return self._shards[key]

        db_path = self._db_path_for(provider)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.refresh_cache and db_path.exists():
            self._log(f"refresh_cache=True -> clearing shard {db_path}", level="info")
            db_path.unlink()

        conn = self._initialise_database(db_path)
        fts_available = self._ensure_fts(conn)
        shard = _ShardInfo(key=key, provider=provider, path=db_path, conn=conn, fts_available=fts_available)
        self._ensure_metadata(shard, provider)
        self._shards[key] = shard
        return shard

    def _ensure_metadata(self, shard: _ShardInfo, provider: str) -> None:
        with self._db_lock:
            with shard.conn:
                existing_provider = shard.conn.execute(
                    "SELECT value FROM llm_cache_metadata WHERE key = 'provider'"
                ).fetchone()
                if not existing_provider:
                    shard.conn.execute(
                        "INSERT OR REPLACE INTO llm_cache_metadata(key, value) VALUES('provider', ?)",
                        (provider,),
                    )
                existing_seed = shard.conn.execute(
                    "SELECT value FROM llm_cache_metadata WHERE key = 'seed'"
                ).fetchone()
                if not existing_seed:
                    shard.conn.execute(
                        "INSERT OR REPLACE INTO llm_cache_metadata(key, value) VALUES('seed', ?)",
                        (self._seed_bucket_identifier(),),
                    )

    def _token_usage_to_dict(self, token_usage: Any) -> Optional[Dict[str, int]]:
        if token_usage is None:
            return None

        fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
        result: Dict[str, int] = {}

        if isinstance(token_usage, dict):
            for field in fields:
                value = token_usage.get(field)
                if value is not None:
                    result[field] = int(value)
        else:
            for field in fields:
                value = getattr(token_usage, field, None)
                if value is not None:
                    result[field] = int(value)

        if not result and isinstance(token_usage, utils.TokenUsage):
            result = {
                "prompt_tokens": int(getattr(token_usage, "prompt_tokens", 0)),
                "completion_tokens": int(getattr(token_usage, "completion_tokens", 0)),
                "total_tokens": int(getattr(token_usage, "total_tokens", 0)),
            }

        return result or None

    def _dict_to_token_usage(self, data: Optional[str]) -> utils.TokenUsage:
        if not data:
            return utils.TokenUsage()
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            payload = {}
        return utils.TokenUsage(payload)

    def _enforce_memory_limit(self) -> None:
        if self.max_memory_entries <= 0:
            self._memory_cache.clear()
            return
        while len(self._memory_cache) > self.max_memory_entries:
            self._memory_cache.popitem(last=False)

    def _update_fts(
        self,
        shard: "_ShardInfo",
        cache_key: str,
        system_prompt: str,
        user_message: str,
        response: str,
    ) -> None:
        if not shard.fts_available:
            return
        with self._db_lock:
            with shard.conn:
                shard.conn.execute("DELETE FROM llm_cache_entries_fts WHERE cache_key = ?", (cache_key,))
                shard.conn.execute(
                    "INSERT INTO llm_cache_entries_fts(cache_key, system_prompt, user_message, response) VALUES (?, ?, ?, ?)",
                    (cache_key, system_prompt, user_message, response),
                )

    def _prune_if_needed(self, shard: "_ShardInfo") -> None:
        if self.max_cache_size_bytes <= 0 or not shard.path.exists():
            return
        try:
            size = os.path.getsize(shard.path)
        except OSError:
            return
        if size <= self.max_cache_size_bytes:
            return

        target_bytes = int(self.max_cache_size_bytes * 0.9)
        self._log(
            f"database size {size / (1024 * 1024):.2f} MB exceeds limit -> pruning {shard.path}", level="warning"
        )

        while size > target_bytes:
            with self._db_lock:
                with shard.conn:
                    rows = shard.conn.execute(
                        "SELECT cache_key FROM llm_cache_entries ORDER BY "
                        "COALESCE(last_accessed_at, created_at) ASC LIMIT 500"
                    ).fetchall()
                    if not rows:
                        break
                    keys = [row["cache_key"] for row in rows]
                    shard.conn.executemany(
                        "DELETE FROM llm_cache_entries WHERE cache_key = ?",
                        ((key,) for key in keys),
                    )
                    if shard.fts_available:
                        shard.conn.executemany(
                            "DELETE FROM llm_cache_entries_fts WHERE cache_key = ?",
                            ((key,) for key in keys),
                        )
            try:
                size = os.path.getsize(shard.path)
            except OSError:
                break

    def _discover_shards_on_disk(self) -> List[_ShardInfo]:
        base = self.cache_dir
        if self._seed_bucket:
            base = base / self._seed_bucket
        if not base.exists():
            return []

        discovered: List[_ShardInfo] = []
        for db_path in base.glob("**/llm_cache.sqlite3"):
            # Skip if we already have an open shard pointing to this path
            if any(existing.path == db_path for existing in self._shards.values()):
                continue
            try:
                conn = self._initialise_database(db_path)
            except sqlite3.Error:
                continue
            fts_available = self._ensure_fts(conn)
            provider = self._read_provider_from_metadata(conn) or db_path.parent.name
            shard = _ShardInfo(
                key=self._shard_key(provider),
                provider=provider,
                path=db_path,
                conn=conn,
                fts_available=fts_available,
            )
            self._ensure_metadata(shard, provider)
            self._shards[shard.key] = shard
            discovered.append(shard)
        return discovered

    def _read_provider_from_metadata(self, conn: sqlite3.Connection) -> Optional[str]:
        try:
            row = conn.execute("SELECT value FROM llm_cache_metadata WHERE key = 'provider'").fetchone()
            if row and row[0]:
                return str(row[0])
        except sqlite3.Error:
            return None
        return None

    def stop(self) -> None:
        """Stops the background flush thread and ensures all data is written."""
        if not self.enabled:
            return
            
        self._log("Stopping flush thread and writing remaining cache items...", level="info")
        self._stop_flush_event.set()
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        
        # Final flush to catch anything left in the buffer
        self._flush_buffer()

    def _flush_loop(self) -> None:
        while not self._stop_flush_event.is_set():
            # Check every 1s, but use wait() to be responsive to stop event
            if self._stop_flush_event.wait(timeout=1.0):
                break
            
            if time.time() - self._last_flush_time >= self._flush_interval:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        # Drain the write buffer under the flush lock to avoid races with set()
        with self._flush_lock:
            if not self._write_buffer:
                return

            pending_writes = self._write_buffer
            self._write_buffer = []
            self._last_flush_time = time.time()

        if not pending_writes:
            return

        self._log(f"Flushing {len(pending_writes)} items to cache...", level="info")

        # Group by provider to minimize shard switching and DB context changes
        writes_by_provider: Dict[str, List[tuple]] = {}
        for provider, params in pending_writes:
            writes_by_provider.setdefault(provider, []).append(params)

        import random  # local import to avoid global dependency if unused elsewhere

        for provider, rows in writes_by_provider.items():
            shard = self._get_shard(provider)

            # Prepare FTS data: (cache_key, system_prompt, user_message, response)
            # params structure:
            #   (cache_key, sys, user, hist, model, prov, params, resp,
            #    usage, created, updated, access, hit)
            # indices: cache_key=0, system_prompt=1, user_message=2, response=7
            fts_rows: List[tuple] = []
            keys_to_delete: List[tuple] = []
            if shard.fts_available:
                for r in rows:
                    keys_to_delete.append((r[0],))
                    fts_rows.append((r[0], r[1], r[2], r[7]))

            # Retry loop for batch write
            max_retries = 6  # reduced from 20
            base_delay = 0.1
            max_delay = 5.0

            success = False
            for attempt in range(max_retries):
                try:
                    with self._db_lock:
                        with shard.conn:
                            # 1. Insert/Update main table
                            shard.conn.executemany(
                                """
                                INSERT INTO llm_cache_entries (
                                    cache_key,
                                    system_prompt,
                                    user_message,
                                    history_json,
                                    model,
                                    provider,
                                    parameters_json,
                                    response,
                                    token_usage_json,
                                    created_at,
                                    updated_at,
                                    last_accessed_at,
                                    hit_count
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                                ON CONFLICT(cache_key) DO UPDATE SET
                                    system_prompt    = excluded.system_prompt,
                                    user_message     = excluded.user_message,
                                    history_json     = excluded.history_json,
                                    model            = excluded.model,
                                    provider         = excluded.provider,
                                    parameters_json  = excluded.parameters_json,
                                    response         = excluded.response,
                                    token_usage_json = excluded.token_usage_json,
                                    updated_at       = excluded.updated_at,
                                    last_accessed_at = excluded.last_accessed_at
                                """,
                                rows,
                            )

                            # 2. Update FTS table if available
                            if shard.fts_available and fts_rows:
                                shard.conn.executemany(
                                    "DELETE FROM llm_cache_entries_fts WHERE cache_key = ?",
                                    keys_to_delete,
                                )
                                shard.conn.executemany(
                                    """
                                    INSERT INTO llm_cache_entries_fts (
                                        cache_key,
                                        system_prompt,
                                        user_message,
                                        response
                                    ) VALUES (?, ?, ?, ?)
                                    """,
                                    fts_rows,
                                )
                    success = True
                    break  # flushed this provider successfully, move to next one

                except sqlite3.OperationalError as e:
                    msg = str(e).lower()

                    # Typical lock/busy errors: "database is locked", "database is busy"
                    if ("locked" in msg) or ("busy" in msg):
                        if attempt < max_retries - 1:
                            delay = min(max_delay, base_delay * (2 ** attempt))
                            jitter = delay * 0.5 * random.random()
                            actual_delay = delay + jitter
                            self._log(
                                f"Database locked during batch flush for provider {provider}, "
                                f"retrying in {actual_delay:.2f}s "
                                f"(attempt {attempt + 1}/{max_retries})",
                                level="warning",
                            )
                            time.sleep(actual_delay)
                            continue
                        else:
                            # Give up after the last attempt – skip this batch
                            self._log(
                                f"Database still locked after {max_retries} attempts for provider {provider}; "
                                f"skipping batch of {len(rows)} entries",
                                level="warning",
                            )
                            break
                    else:
                        # Non-lock OperationalError: log once and skip this batch
                        self._log(
                            f"SQLite OperationalError during batch flush for provider {provider}: {e}; "
                            f"skipping batch of {len(rows)} entries",
                            level="error",
                        )
                        break

                except Exception as e:
                    # Any other unexpected error while flushing this provider
                    self._log(
                        f"Unexpected error during batch flush for provider {provider}: {e}; "
                        f"skipping batch of {len(rows)} entries",
                        level="error",
                    )
                    break

            if not success:
                # Intentional: this is a best-effort cache, so it's acceptable
                # to drop a batch if the DB keeps rejecting writes.
                continue

    def _collect_shards_for_search(self, provider: Optional[str]) -> List[_ShardInfo]:
        if provider is not None:
            return [self._get_shard(str(provider))]

        shards = list(self._shards.values())
        if not shards:
            shards = self._discover_shards_on_disk()
        else:
            self._discover_shards_on_disk()
            shards = list(self._shards.values())
        return shards

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]],
        model: str,
        provider: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> Optional[CacheEntry]:
        if not self.enabled:
            return None

        provider = str(provider)
        cache_key = self._hash_request(
            str(system_prompt).strip(),
            str(user_message).strip(),
            history,
            model,
            provider,
            max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
        )

        shard = self._get_shard(provider)

        with self._memory_lock:
            if cache_key in self._memory_cache:
                entry = self._memory_cache.pop(cache_key)
                self._memory_cache[cache_key] = entry
                with self._lock:
                    self.stats["hits"] += 1
                    self.stats["memory_hits"] += 1
                    self.stats["total_calls"] += 1
                if self.stats["total_calls"] % self.stats_log_every == 0:
                    self._log_stats()
                return entry

        with self._db_lock:
            with shard.conn:
                row = shard.conn.execute(
                    "SELECT user_message, response, token_usage_json, hit_count FROM llm_cache_entries WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()

        with self._lock:
            self.stats["total_calls"] += 1
            if row:
                self.stats["hits"] += 1
                self.stats["disk_hits"] += 1
            else:
                self.stats["misses"] += 1

        if not row:
            if self.stats["total_calls"] % self.stats_log_every == 0:
                self._log_stats()
            return None

        token_usage = self._dict_to_token_usage(row["token_usage_json"])
        entry = (row["user_message"], row["response"], token_usage)

        with self._db_lock:
            with shard.conn:
                now = time.time()
                shard.conn.execute(
                    "UPDATE llm_cache_entries SET last_accessed_at = ?, hit_count = hit_count + 1 WHERE cache_key = ?",
                    (now, cache_key),
                )

        with self._memory_lock:
            self._memory_cache[cache_key] = entry
            self._enforce_memory_limit()

        if self.stats["total_calls"] % self.stats_log_every == 0:
            self._log_stats()

        return entry

    def set(
        self,
        system_prompt: str,
        user_message: str,
        response: str,
        token_usage: Any,
        history: Optional[List[Dict[str, str]]],
        model: str,
        provider: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> bool:
        if not self.enabled:
            return False

        system_prompt = str(system_prompt).strip()
        user_message = str(user_message).strip()
        response = str(response)
        provider = str(provider)

        if self.max_response_size_bytes > 0 and len(response.encode("utf-8")) > self.max_response_size_bytes:
            self._log("response skipped due to size limit", level="warning")
            return False

        cache_key = self._hash_request(
            system_prompt,
            user_message,
            history,
            model,
            provider,
            max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
        )

        shard = self._get_shard(provider)

        history_json = json.dumps(self._normalise_history(history), ensure_ascii=False)
        parameters_json = json.dumps(
            self._parameter_payload(
                model, provider, max_tokens, temperature, top_p, frequency_penalty, presence_penalty
            ),
            separators=(",", ":"),
            sort_keys=True,
        )
        token_usage_dict = self._token_usage_to_dict(token_usage)
        token_usage_json = json.dumps(token_usage_dict, sort_keys=True) if token_usage_dict else None
        timestamp = time.time()

        # Add to write buffer instead of writing immediately
        with self._flush_lock:
            self._write_buffer.append((
                provider,
                (
                    cache_key,
                    system_prompt,
                    user_message,
                    history_json,
                    str(model),
                    str(provider),
                    parameters_json,
                    response,
                    token_usage_json,
                    timestamp,
                    timestamp,
                    timestamp,
                )
            ))
            
            # If buffer gets too big, flush immediately
            should_flush = len(self._write_buffer) >= 100
            
        if should_flush:
            self._flush_buffer()

        # We still update FTS and memory cache immediately for consistency
        # Note: FTS update might fail if the main entry isn't there yet, 
        # but for now we prioritize the main storage. 
        # Ideally FTS should also be batched, but let's keep it simple for now.
        # self._update_fts(shard, cache_key, system_prompt, user_message, response)

        with self._memory_lock:
            entry: CacheEntry = (user_message, response, utils.TokenUsage(token_usage_dict or {}))
            self._memory_cache[cache_key] = entry
            self._enforce_memory_limit()

        self._prune_if_needed(shard)
        return True

    def search(
        self,
        query: str,
        field: str = "user_message",
        limit: int = 20,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not query:
            return []

        field = field.lower()
        if field not in {"user_message", "system_prompt", "response"}:
            field = "user_message"

        shards = self._collect_shards_for_search(provider)

        results: List[Dict[str, Any]] = []
        for shard in shards:
            if shard.fts_available:
                pattern = query if ":" in query else f"{field}:{query}"
                statement = (
                    "SELECT e.cache_key, e.user_message, e.response, e.model, e.provider, e.hit_count, "
                    "e.last_accessed_at, e.updated_at "
                    "FROM llm_cache_entries AS e "
                    "JOIN llm_cache_entries_fts ON e.cache_key = llm_cache_entries_fts.cache_key "
                    "WHERE llm_cache_entries_fts MATCH ? "
                    "ORDER BY (e.last_accessed_at IS NULL), e.last_accessed_at DESC LIMIT ?"
                )
                params = (pattern, int(limit))
            else:
                statement = (
                    f"SELECT cache_key, user_message, response, model, provider, hit_count, last_accessed_at, updated_at "
                    f"FROM llm_cache_entries WHERE {field} LIKE ? "
                    "ORDER BY (last_accessed_at IS NULL), last_accessed_at DESC LIMIT ?"
                )
                params = (f"%{query}%", int(limit))

            with self._db_lock:
                with shard.conn:
                    rows = shard.conn.execute(statement, params).fetchall()

            for row in rows:
                results.append(
                    {
                        "cache_key": row["cache_key"],
                        "user_message": row["user_message"],
                        "response_snippet": row["response"][:200] if row["response"] else "",
                        "model": row["model"],
                        "provider": row["provider"],
                        "hit_count": row["hit_count"],
                        "last_accessed_at": row["last_accessed_at"],
                        "updated_at": row["updated_at"],
                    }
                )

        results.sort(
            key=lambda item: (
                item["last_accessed_at"] is None,
                item["last_accessed_at"] if item["last_accessed_at"] is not None else item["updated_at"],
            ),
            reverse=True,
        )

        return results[:limit]

    def delete(self, cache_key: str) -> None:
        if not cache_key:
            return
        shards = self._collect_shards_for_search(None)
        for shard in shards:
            with self._db_lock:
                with shard.conn:
                    cursor = shard.conn.execute(
                        "DELETE FROM llm_cache_entries WHERE cache_key = ?",
                        (cache_key,),
                    )
                    if cursor.rowcount:
                        if shard.fts_available:
                            shard.conn.execute(
                                "DELETE FROM llm_cache_entries_fts WHERE cache_key = ?",
                                (cache_key,),
                            )
                        break
        with self._memory_lock:
            self._memory_cache.pop(cache_key, None)

    def clear_cache(self) -> None:
        shards = self._collect_shards_for_search(None)
        for shard in shards:
            shard.close()
            try:
                if shard.path.exists():
                    os.remove(shard.path)
            except OSError:
                pass
        self._shards.clear()

        with self._memory_lock:
            self._memory_cache.clear()

        import shutil

        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
            except OSError:
                pass
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._log("cache cleared", level="info")

    def get_cache_stats(self) -> Dict[str, Any]:
        on_disk = 0
        size_mb = 0.0
        total_hits = 0
        shards = self._collect_shards_for_search(None)
        seen_paths = set()
        for shard in shards:
            with self._db_lock:
                with shard.conn:
                    row = shard.conn.execute(
                        "SELECT COUNT(*) AS c, SUM(hit_count) AS h FROM llm_cache_entries"
                    ).fetchone()
            if row:
                on_disk += int(row["c"] or 0)
                total_hits += int(row["h"] or 0)
            try:
                if shard.path not in seen_paths and shard.path.exists():
                    size_mb += os.path.getsize(shard.path) / (1024 * 1024)
                    seen_paths.add(shard.path)
            except OSError:
                continue
        return {
            "enabled": self.enabled,
            "task_name": self.task_name,
            "entries": on_disk,
            "db_size_mb": size_mb,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "total_calls": self.stats["total_calls"],
            "memory_hits": self.stats["memory_hits"],
            "disk_hits": self.stats["disk_hits"],
            "total_row_hits": total_hits,
            "memory_cache_entries": len(self._memory_cache),
            "max_memory_entries": self.max_memory_entries,
            "cache_root": str(self.cache_dir),
            "shard_count": len(seen_paths) if seen_paths else len(shards),
        }

    def _log_stats(self) -> None:
        stats = self.get_cache_stats()
        hit_rate = (
            stats["hits"] / stats["total_calls"] if stats["total_calls"] else 0
        )
        self._log(
            f"stats -> calls={stats['total_calls']} hits={stats['hits']} misses={stats['misses']} "
            f"hit_rate={hit_rate:.2%} mem_hits={stats['memory_hits']} disk_hits={stats['disk_hits']}",
            level="info",
            print_screen=False,
        )

    def print_stats(self) -> None:
        stats = self.get_cache_stats()
        hit_rate = (
            stats["hits"] / stats["total_calls"] if stats["total_calls"] else 0
        )
        lines = [
            "=== LLM Cache Statistics ===",
            f"Enabled            : {stats['enabled']}",
            f"Task Name          : {stats['task_name']}",
            f"Cache Root         : {stats['cache_root']}",
            f"Shard Count        : {stats['shard_count']}",
            f"Entries (on disk)  : {stats['entries']}",
            f"Database Size      : {stats['db_size_mb']:.2f} MB",
            f"Total Calls        : {stats['total_calls']}",
            f"Hits               : {stats['hits']}",
            f"Misses             : {stats['misses']}",
            f"Hit Rate           : {hit_rate:.2%}",
            f"Memory Hits        : {stats['memory_hits']}",
            f"Disk Hits          : {stats['disk_hits']}",
            f"Row Hit Sum        : {stats['total_row_hits']}",
            f"Memory Cache Size  : {stats['memory_cache_entries']} / {stats['max_memory_entries']}",
        ]
        report = "\n".join(lines)
        if isinstance(self.logger, utils.Logger):
            self.logger.log(report)
        elif hasattr(self.logger, "info"):
            self.logger.info(report)
        else:
            print(report)

    def close(self) -> None:
        with self._memory_lock:
            self._memory_cache.clear()
        for shard in list(self._shards.values()):
            shard.close()
        self._shards.clear()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


_global_cache: Optional[LLMCache] = None


def initialize_cache(
    cache_dir: str = ".llm_cache",
    enabled: bool = True,
    random_seed: Optional[int] = None,
    refresh_cache: bool = False,
    task_name: Optional[str] = None,
    max_cache_size_mb: int = 1024,
    max_response_size_kb: int = 512,
    max_memory_entries: int = 5000,
    shard_by_seed: bool = True,
    shard_by_provider: bool = True,
    logger: Optional[Any] = None,
) -> LLMCache:
    global _global_cache
    _global_cache = LLMCache(
        cache_dir=cache_dir,
        enabled=enabled,
        random_seed=random_seed,
        refresh_cache=refresh_cache,
        task_name=task_name,
        max_cache_size_mb=max_cache_size_mb,
        max_response_size_kb=max_response_size_kb,
        max_memory_entries=max_memory_entries,
        shard_by_seed=shard_by_seed,
        shard_by_provider=shard_by_provider,
        logger=logger,
    )
    return _global_cache


def get_cache() -> Optional[LLMCache]:
    return _global_cache


def clear_global_cache() -> None:
    if _global_cache:
        _global_cache.clear_cache()


def print_global_cache_stats() -> None:
    if _global_cache:
        _global_cache.print_stats()
    else:
        print("Cache not initialized.")
