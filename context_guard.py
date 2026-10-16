"""Shadow (Phase 1) context-limit instrumentation for hermes-session-reset-policy.

LOG-ONLY. On every inbound gateway message this module measures a conversation's
token footprint from *durable* state and compares it against the operator's
``context_guard`` thresholds, emitting structured ``context_guard:`` log lines.
It NEVER compacts, rotates, or otherwise mutates a conversation: enforcement is
a later phase, gated by ``context_guard.enabled`` and reviewed before it ships
(design contract: ``docs/context-limit-guard-spec.md``).

Why shadow first
----------------
A hard cap that silently rotates or compacts the wrong way is worse than slow
turns, so the enforcement policy is calibrated against real traffic before the
switch is flipped. This phase answers only: "for real conversations, what does
the live context actually measure, and when would the thresholds fire?"

Metrics (all durable, best-effort, fail-open — a missing field is None, never a
crash):

- ``active_context_tokens`` — SUM(messages.token_count) over the conversation's
  ACTIVE, non-compacted messages: the closest durable proxy for the live prefill
  size (what a cold turn pays for). The enforcement metric.
- ``input_tokens`` / ``output_tokens`` / ``billed_tokens`` — the ``sessions``
  row's cumulative counters: the conversation's total spend (a bloat signal that
  never under-reports).
- ``api_calls`` / ``message_count`` — the conversation's shape.

Standalone-plugin discipline: stdlib only at module scope; the single core touch
is a lazy, guarded read of the store handle the caller passes in. No Hermes
import, no global state, no side effects beyond a small metadata write (the same
channel the policy's own clock uses — see ``read_state``).
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.session_reset_policy.context_guard")

# Phase 1 is measurement-only. Enforcement is a separate, reviewed change; this
# flag is the single source of truth for "are we allowed to act?" so a config
# mistake (`enabled: true`) can never rotate a conversation from this build.
SHADOW = True

# Per-route metadata blob, on the SAME store/channel as the policy clock
# (``srp_last_user_inbound``): JSON-serializable, and — unlike the routing row —
# it does not advance ``updated_at``, so shadow bookkeeping never makes an idle
# conversation look fresh (set_session_metadata's documented contract).
_STATE_KEY = "srp_context_guard"

# Persisted schema version, so a future phase can migrate the (tiny) blob in place.
_STATE_VERSION = 1

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "shadow": True,
    "max_tokens": 120000,
    "hard_tokens": 150000,
    "mode": "compact",
    "min_turns": 5,
    "cooldown_seconds": 300,
    "log_level": "info",
}

# ``sessions`` carries the cumulative per-conversation counters (input/output
# tokens, api calls, messages); ``messages`` carries the live, uncompacted set.
_SESSIONS_SQL = (
    "SELECT input_tokens, output_tokens, api_call_count, message_count, model_config "
    "FROM sessions WHERE id = ?"
)
_ACTIVE_SQL = (
    "SELECT COALESCE(SUM(token_count), 0) FROM messages "
    "WHERE session_id = ? AND active = 1 AND compacted = 0"
)


def resolve_config(raw: Any) -> Dict[str, Any]:
    """Merge the operator's ``context_guard`` block over DEFAULTS.

    Returns a fresh dict; *raw* (the shared readonly config object) is never
    mutated. Unknown keys pass through so a future knob is not silently dropped
    by this build.
    """
    cfg = dict(DEFAULTS)
    if isinstance(raw, dict):
        cfg.update(raw)
    return cfg


def _as_int(value: Any) -> Optional[int]:
    """Best-effort int; None for anything non-numeric (bools excluded)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_iso(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _store_db(session_store: Any, session_key: Optional[str]) -> Any:
    """The conversation's SessionDB handle, or None.

    Prefers the per-key handle (``_db_for_key``: the profile-scoped DB that
    actually owns the route) and falls back to the store's default handle. The
    same seam the policy's in-flight probe uses, so both read one store.
    """
    if session_store is None:
        return None
    try:
        get_db = getattr(session_store, "_db_for_key", None)
        if session_key and callable(get_db):
            db = get_db(session_key)
            if db is not None:
                return db
    except Exception:
        logger.debug("context_guard: _db_for_key(key) failed", exc_info=True)
    return getattr(session_store, "_db", None)


def _anchor_prompt_tokens(model_config_json: Any) -> Optional[int]:
    """Last ACTUAL prefill size from ``model_config._usage_anchor.prompt_tokens``.

    This is the number that costs latency (the durable, live-context proxy), and it
    survives compression. ``messages.token_count`` is NULL on some deployments, so
    this anchor is the primary metric; the message SUM is only a fallback.
    """
    if not model_config_json:
        return None
    try:
        cfg = json.loads(model_config_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    anchor = cfg.get("_usage_anchor")
    if not isinstance(anchor, dict):
        return None
    return _as_int(anchor.get("prompt_tokens"))


def measure(session_store: Any, session_key: Optional[str], entry: Any) -> Dict[str, Any]:
    """Durable token metrics for *entry*'s conversation. Never raises.

    ``active_context_tokens`` = the conversation's last actual prefill size, from
    ``sessions.model_config._usage_anchor.prompt_tokens`` (the number that costs
    latency); when that anchor is absent it falls back to
    ``SUM(messages.token_count)`` over the active, non-compacted set. Totals
    (``input_tokens``/``output_tokens``/``billed_tokens``) come from the ``sessions``
    row. Every field is best-effort; a missing value is None, never a crash.
    """
    out: Dict[str, Any] = {
        "session_id": None,
        "active_context_tokens": None,
        "active_context_source": "none",
        "input_tokens": None,
        "output_tokens": None,
        "billed_tokens": None,
        "api_calls": None,
        "message_count": None,
    }
    sid = getattr(entry, "session_id", None)
    if sid is None:
        return out
    out["session_id"] = str(sid)
    db = _store_db(session_store, session_key)
    if db is None or not callable(getattr(db, "_read_one", None)):
        return out
    anchor = None
    try:
        row = db._read_one(_SESSIONS_SQL, (str(sid),))
        if row is not None:
            in_t = _as_int(row[0]) or 0
            out_t = _as_int(row[1]) or 0
            out["input_tokens"] = in_t
            out["output_tokens"] = out_t
            out["billed_tokens"] = in_t + out_t
            out["api_calls"] = _as_int(row[2])
            out["message_count"] = _as_int(row[3])
            if len(row) > 4:
                anchor = _anchor_prompt_tokens(row[4])
    except Exception:
        logger.debug("context_guard: sessions metrics read failed", exc_info=True)
    sum_tokens = None
    try:
        arow = db._read_one(_ACTIVE_SQL, (str(sid),))
        if arow is not None:
            sum_tokens = _as_int(arow[0]) or 0
    except Exception:
        logger.debug("context_guard: active-context read failed", exc_info=True)
    if anchor is not None:
        out["active_context_tokens"] = anchor
        out["active_context_source"] = "usage_anchor"
    elif sum_tokens:
        out["active_context_tokens"] = sum_tokens
        out["active_context_source"] = "messages"
    return out


def read_state(session_store: Any, session_key: Optional[str]) -> Dict[str, Any]:
    """The conversation's shadow state blob (dict), or {} — tolerant of every shape."""
    if session_store is None or session_key is None:
        return {}
    try:
        raw = session_store.get_session_metadata(session_key, _STATE_KEY, None)
    except Exception:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            loaded = json.loads(raw)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    return {}


def write_state(session_store: Any, session_key: Optional[str], state: Dict[str, Any]) -> None:
    """Persist the shadow state blob. Best-effort; never advances the idle clock."""
    if session_store is None or session_key is None:
        return
    with contextlib.suppress(Exception):
        session_store.set_session_metadata(session_key, _STATE_KEY, state)


def _size_metric(metrics: Dict[str, Any]) -> Optional[int]:
    """The enforcement size: live prefill proxy when known, else cumulative spend."""
    active = metrics.get("active_context_tokens")
    if active is not None:
        return active
    return metrics.get("billed_tokens")


def evaluate(cfg: Dict[str, Any], metrics: Dict[str, Any], now: datetime) -> Dict[str, Any]:
    """Classify the conversation against the cap. Pure; no logging, no I/O."""
    size = _size_metric(metrics)
    soft = _as_int(cfg.get("max_tokens"))
    hard = _as_int(cfg.get("hard_tokens"))
    if size is None:
        level = "unknown"
    elif hard is not None and size >= hard:
        level = "hard"
    elif soft is not None and size >= soft:
        level = "soft"
    else:
        level = "ok"
    return {
        "level": level,
        "size": size,
        "soft_tokens": soft,
        "hard_tokens": hard,
        "over_soft": level in ("soft", "hard"),
        "over_hard": level == "hard",
        "checked_at": now,
    }


def _cooldown_active(state: Dict[str, Any], now: datetime, seconds: Optional[int]) -> bool:
    if not seconds or seconds <= 0:
        return False
    last = _parse_iso(state.get("last_warned_at"))
    if last is None:
        return False
    return (now - last) < timedelta(seconds=seconds)


def _log_routine(cfg: Dict[str, Any], key: str, metrics: Dict[str, Any], verdict: Dict[str, Any]) -> None:
    level_name = str(cfg.get("log_level") or "info").lower()
    log = getattr(logger, level_name, logger.info)
    log(
        "context_guard: shadow_log session=%s in=%s out=%s total=%s active=%s active_src=%s "
        "api_calls=%s messages=%s level=%s",
        key,
        metrics.get("input_tokens"),
        metrics.get("output_tokens"),
        metrics.get("billed_tokens"),
        metrics.get("active_context_tokens"),
        metrics.get("active_context_source"),
        metrics.get("api_calls"),
        metrics.get("message_count"),
        verdict["level"],
    )


def _log_breach(cfg: Dict[str, Any], key: str, metrics: Dict[str, Any],
                verdict: Dict[str, Any], now: datetime, state: Dict[str, Any]) -> None:
    if not verdict["over_soft"]:
        return
    if _cooldown_active(state, now, _as_int(cfg.get("cooldown_seconds"))):
        return
    size = verdict["size"]
    mode = str(cfg.get("mode") or "compact")
    if verdict["over_hard"]:
        logger.warning(
            "context_guard: hard_cap_exceeded session=%s active=%s hard_tokens=%s mode=%s "
            "triggering_compaction=true (shadow: no action taken)",
            key, size, cfg.get("hard_tokens"), mode,
        )
    else:
        logger.warning(
            "context_guard: soft_cap_reached session=%s active=%s max_tokens=%s mode=%s "
            "(shadow: would compact)",
            key, size, cfg.get("max_tokens"), mode,
        )
    state["last_warned_at"] = now.isoformat()
    state["last_warned_level"] = verdict["level"]


def observe(cfg: Dict[str, Any], session_store: Any, session_key: Optional[str], entry: Any,
            *, now: Optional[datetime] = None,
            metrics: Optional[Dict[str, Any]] = None,
            state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """One shadow pass for *entry*'s conversation: measure, classify, log, remember.

    Returns the verdict dict (or None when there is nothing to do / no route).
    ``metrics`` and ``state`` are injectable for tests. This function never
    raises and never mutates the conversation: the only write is the shadow
    state blob (``set_session_metadata``).
    """
    if session_store is None or session_key is None or entry is None:
        return None
    if not cfg.get("shadow", True) and not cfg.get("enabled", False):
        return None
    if cfg.get("enabled") and SHADOW:
        logger.warning(
            "context_guard: 'enabled' requested but this build is shadow-only; "
            "no enforcement performed",
        )

    now = now or datetime.now()
    metrics = metrics if metrics is not None else measure(session_store, session_key, entry)
    state = state if state is not None else read_state(session_store, session_key)
    verdict = evaluate(cfg, metrics, now)

    # Cooldown bookkeeping is the ONLY thing worth persisting: the per-message
    # metrics already ride the shadow_log line, so persisting them would force a
    # full routing rewrite + sessions.json fsync on every message. Write only when
    # the persisted cooldown state actually changes (or on first seed), so
    # steady-state shadow logging costs ZERO store writes.
    seeded = bool(state)
    prev_warned = state.get("last_warned_at")
    prev_level = state.get("last_warned_level")

    min_turns = _as_int(cfg.get("min_turns")) or 0
    mc = metrics.get("message_count")
    too_young = mc is not None and mc < min_turns
    if not too_young:
        _log_routine(cfg, session_key, metrics, verdict)
        _log_breach(cfg, session_key, metrics, verdict, now, state)  # stamps last_warned_* on breach

    if (not seeded
            or state.get("last_warned_at") != prev_warned
            or state.get("last_warned_level") != prev_level):
        write_state(session_store, session_key, {
            "version": _STATE_VERSION,
            "last_warned_at": state.get("last_warned_at"),
            "last_warned_level": state.get("last_warned_level"),
        })
    return verdict
