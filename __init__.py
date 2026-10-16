"""Hermes session-reset-policy plugin.

Re-arms the config-declared ``session_reset`` policy (mode: idle|daily|both)
that upstream Hermes v2026.9.11 (commit 1d5d059410) made inert by removing the
time-triggered conversation-rotation timers. Upstream's stance is that only
explicit suspension should replace a routed conversation; this plugin restores
the time-based policy for operators who want it, as an opt-in plugin instead of
a core patch.

How it works
------------
On every inbound gateway message (``pre_gateway_dispatch``), before auth and
dispatch, the plugin:

1. Computes the session key for the message source (same helper the gateway
   uses, so multiplexed profiles and platform key layouts match exactly).
2. Reads that profile's ``config.yaml`` ``session_reset`` block. Under
   multiplexing the hook fires before the per-profile runtime scope is
   installed, so the profile home is resolved from ``event.source.profile``
   and the config is read directly from ``profiles/<name>/config.yaml``.
3. Compares the route's OWN user-activity clock — ``srp_last_user_inbound``
   session metadata, advanced only by real user messages that reach this hook
   — against the idle threshold and/or daily boundary. (Why not the entry's
   ``updated_at``: turn-start stamps it for EVERY turn, including internal
   background-review turns, so on an idle gateway where reviews chain the
   clock never goes stale and the idle reset can never fire. Slash commands
   such as ``/status`` also stamp ``updated_at`` — they are bookkeeping, not
   conversation, and this hook skips them: see "Slash commands" below.)
4. If past the boundary and no turn is currently in flight for that session,
   resets the conversation through the gateway's OWN ``/new`` reset funnel
   (``gateway._handle_reset_command``, fed a text-free event): the exact path
   a typed ``/new`` takes, so everything a conversation boundary owes is done
   — run-generation bump and cached-agent cleanup/eviction (the compressor's
   previous summary cannot carry over), the conversation-scope clear
   (per-conversation ``/model`` and reasoning overrides), interruption of
   in-flight async delegations, the durable store rotation
   (``SessionStore.reset_session``: route, transcript end, state.db row), the
   ``session:end``/``session:reset`` gateway hooks, and the
   ``on_session_finalize``/``on_session_reset`` plugin hooks (review #2).
   When the funnel is not reachable (older/renamed gateway internals) the
   plugin degrades to a bare ``SessionStore.reset_session()`` plus the
   lifecycle hook pair and the delegation interrupt — a consistent rotation,
   without the gateway-side cache/scope teardown.
5. Otherwise records this message's arrival time as the new clock value
   (via ``set_session_metadata``, which deliberately does not touch
   ``updated_at``).

The event handed to the reset funnel is stripped of its text: the funnel
treats ``event.get_command_args()`` as ``/new <title>``, and on a plain
message that returns the whole utterance — left intact, a policy reset would
title the fresh session with whatever the user happened to send.

Migration: routes that predate the plugin have no metadata clock; their first
hook-seen message falls back to ``updated_at``/``created_at`` once, then the
metadata clock takes over.

Because the reset happens lazily on the first *inbound message* past the
boundary (not on a timer), a session that goes idle is simply continued when
the user comes back within the threshold, and reset exactly once when they
come back after it. Internal/cron traffic never passes through this hook and
slash commands are skipped by it, so background activity can never keep a
session alive nor reset it — and can no longer keep one *un-reset* either.

Slash commands
--------------
The gateway DOES run ``pre_gateway_dispatch`` for slash commands: it fires
for every non-internal event, before command routing. The hook therefore
skips commands explicitly (``event.get_command()``; a bare leading ``/`` for
events without that helper). A command neither advances the user-activity
clock nor triggers a time reset. Without the skip, a ``/new`` or ``/reset``
sent past the boundary rotated the route TWICE: once here, then once more by
the command itself. That burned a run generation and left an empty,
already-ended session row behind. Message text that only looks like a path
(``/etc/hosts is broken``) is not a command (``get_command()`` rejects it)
and is handled as an ordinary message.

Deferral semantics: when the boundary is past but a turn is in flight, the
clock is NOT advanced, so the next message re-evaluates the same boundary
against the same last-user-inbound time.

Compatibility notes
-------------------
- Requires Hermes >= v2026.9.11 (needs ``pre_gateway_dispatch`` hook payload
  with ``session_store``, and ``SessionStore.reset_session``). The full funnel
  path additionally uses ``gateway._handle_reset_command`` (present ever since
  ``/new`` exists; private, per review #2 reused deliberately until a public
  reset funnel exists).
- Session key resolution: the hook payload's ``session_key`` first (when a
  future Hermes provides a public one), else ``gateway._session_key_for_source``
  (review #2: "the right helper today", private but blessed). There is NO bare
  ``build_session_key`` fallback on purpose: under multiplexed profiles that
  key is not profile-namespaced and could collide with ANOTHER profile's
  route — a wrong-key reset would be worse than no reset.
- The reset is an *explicit-style* reset (like ``/new``): the fresh session
  does not carry ``was_auto_reset``/``auto_reset_reason`` bookkeeping, so the
  agent does not receive the auto-reset context note or channel-continuity
  hint that the old core patch produced. First-turn topic/channel skill
  re-injection DOES apply (``is_fresh_reset`` is set by ``reset_session``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.session_reset_policy")

__version__ = "0.4.0"

_META_KEY = "srp_last_user_inbound"


def _read_config_block(profile: Optional[str], block: str = "session_reset") -> Optional[Dict[str, Any]]:
    """Return the top-level *block* dict for *profile*, or None when unset/invalid.

    ``pre_gateway_dispatch`` fires before the per-profile runtime scope is
    installed, so the ambient ``load_config_readonly()`` would read the launch
    profile. Instead the profile's home is installed as a context-local
    hermes-home override — the same mechanism the gateway uses for profile
    scoping — and the config cache is keyed by the resolved config path, so
    each profile's block is read from its own config.yaml. The home is
    resolved via :func:`hermes_cli.profiles.get_profile_dir`, which anchors
    names to the Hermes ROOT (not the current home) and handles ``default``
    as the root itself: a standalone gateway may run with ``HERMES_HOME``
    already set to its own profile dir, where ``<home>/profiles/<name>`` would
    double-nest and find nothing (issue #4). The returned dict is the SHARED
    cache object (readonly variant): never mutated. Never raises.
    """
    try:
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from hermes_cli.config import load_config_readonly
        from hermes_cli.profiles import get_profile_dir

        token = None
        if profile:
            token = set_hermes_home_override(get_profile_dir(profile))
        try:
            config = load_config_readonly()
        finally:
            if token is not None:
                reset_hermes_home_override(token)
        value = (config or {}).get(block)
        if not isinstance(value, dict) or not value:
            return None  # unset/empty block: treat as "none", skip quietly
        return value
    except Exception:
        logger.debug("config block %r read failed; skipping", block, exc_info=True)
        return None


def _read_session_reset_policy(profile: Optional[str]) -> Optional[Dict[str, Any]]:
    """The ``session_reset`` block for *profile* (thin wrapper over _read_config_block).

    Kept as this module's policy seam: callers and the test-suite patch THIS name.
    """
    return _read_config_block(profile, "session_reset")


def _parse_iso(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _last_user_inbound(entry: Any) -> Optional[datetime]:
    """The route's user-activity clock.

    Order: the plugin's own ``srp_last_user_inbound`` metadata (written only by
    messages that reach this hook), then ``updated_at``/``created_at`` as a
    one-time migration fallback for routes that predate the plugin.
    """
    meta = getattr(entry, "metadata", None) or {}
    stamp = _parse_iso(meta.get(_META_KEY)) if isinstance(meta, dict) else None
    if stamp is not None:
        return stamp
    return getattr(entry, "updated_at", None) or getattr(entry, "created_at", None)


def _time_reset_reason(policy: Dict[str, Any], last: Optional[datetime], now: datetime) -> Optional[str]:
    """Return "idle"/"daily" when *last* user activity is past the policy boundary, else None.

    Idle compares the user-activity clock against the threshold; daily fires on
    the first activity after the configured local hour. Fails open (None) on
    any malformed value.
    """
    try:
        mode = str(policy.get("mode") or "none")
        if mode == "none":
            return None
        if last is None:
            return None
        if not isinstance(last, datetime):
            last = _parse_iso(last)
            if last is None:
                return None
        if mode in ("idle", "both"):
            idle_minutes = policy.get("idle_minutes")
            if isinstance(idle_minutes, (int, float)) and idle_minutes > 0:
                if now - last >= timedelta(minutes=float(idle_minutes)):
                    return "idle"
        if mode in ("daily", "both"):
            at_hour = policy.get("at_hour")
            if isinstance(at_hour, int) and 0 <= at_hour <= 23:
                boundary = now.replace(hour=at_hour, minute=0, second=0, microsecond=0)
                if boundary > now:
                    boundary -= timedelta(days=1)
                if last < boundary:
                    return "daily"
    except Exception:
        logger.debug("session_reset boundary evaluation failed; skipping", exc_info=True)
    return None


class _SessionBusy(Exception):
    """The conversation has a live turn/worker; a policy reset must DEFER, not fail.

    Raised (never returned) so ``_maybe_reset`` can tell a benign "not now" — log a
    defer and leave the clock un-advanced — from a real reset failure.
    """


def _store_busy(exc: BaseException) -> bool:
    """True when *exc* (or a cause in its chain) is the store's retryable "busy" refusal.

    ``require_not_executing`` raises ``RuntimeStoreError('session_busy')`` (a live
    turn/worker) or ``'unknown_execution'`` (an outcome we cannot trust). Both mean
    "not now", never "broken": the caller must defer and re-arm, not attempt a second
    rotation. Matched by class name + message so a renamed/absent core module cannot
    break the plugin (it already guards every core import).
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ == "RuntimeStoreError" and str(cur) in ("session_busy", "unknown_execution"):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _session_in_flight(entry: Any, session_store: Any = None,
                       session_key: Optional[str] = None) -> bool:
    """True when a turn/worker is live for *entry*'s conversation, so a reset must defer.

    The durable ``active_turn_token`` marker covers the in-process case (set under the
    store's lock at turn start, CAS-cleared on unwind, recovered after a crash — so a
    stale marker is conservatively in-flight). But a turn driven OUT of process — the
    desktop, or a message dispatched while a prior turn still runs — leaves the marker
    clear while the store still holds a ``started`` admission; attempting the reset then
    trips ``require_not_executing`` and the funnel raises ``session_busy`` after a wasted
    rotation attempt (the "falling back to a bare store rotation" path). Probe the SAME
    durable state the guard refuses on so the plugin defers (its documented contract)
    instead of failing. Never raises.
    """
    if getattr(entry, "active_turn_token", None):
        return True
    sid = getattr(entry, "session_id", None)
    if session_store is None or not sid:
        return False
    try:
        db = None
        get_db = getattr(session_store, "_db_for_key", None)
        if session_key and callable(get_db):
            db = get_db(session_key)
        if db is None:
            db = getattr(session_store, "_db", None)
        if db is None:
            return False
        from hermes_state_mutation_guards import require_not_executing
        ids = [str(sid)]
        parent = getattr(entry, "parent_session_id", None)
        if parent:
            ids.append(str(parent))
        with db._read_ctx() as conn:
            require_not_executing(conn, list(dict.fromkeys(ids)))
        return False
    except Exception as exc:
        if _store_busy(exc):
            return True
        logger.debug("session-reset-policy: in-flight probe failed; treating as idle",
                     exc_info=True)
        return False


def _lookup_entry(session_store: Any, session_key: Optional[str]) -> Any:
    """Best-effort route lookup; None on any failure (fail open to "no route")."""
    if not session_key:
        return None
    try:
        return session_store.lookup_by_session_key(session_key)
    except Exception:
        return None


def _is_slash_command(event: Any) -> bool:
    """True when *event* is a gateway slash command (``/new``, ``/status``, ...).

    Uses the event's own ``get_command()`` when present: the same parse the
    gateway's command router uses, which honours ``allow_gateway_control`` and
    rejects path-like text (``/etc/hosts is broken``). Events without that
    helper fall back to a leading ``/`` on their text. Never raises.
    """
    get_command = getattr(event, "get_command", None)
    if callable(get_command):
        try:
            return bool(get_command())
        except Exception:
            logger.debug("session-reset-policy: get_command() failed; using the text check", exc_info=True)
    text = getattr(event, "text", None)
    return isinstance(text, str) and text.lstrip().startswith("/")


def _resolve_session_key(gateway: Any, source: Any, from_payload: Optional[str]) -> Optional[str]:
    """The routing key for *source*: the hook payload's ``session_key`` when a
    future Hermes exposes one publicly, else ``gateway._session_key_for_source``.

    The gateway helper is private but blessed by review #2 ("the right helper
    today"). Deliberately NO bare ``build_session_key`` fallback: that key is
    not profile-namespaced under multiplexing and could resolve ANOTHER
    profile's route for the same platform/chat — fail closed instead.
    """
    if from_payload:
        return str(from_payload)
    helper = getattr(gateway, "_session_key_for_source", None)
    if not callable(helper):
        logger.debug("session-reset-policy: no session-key helper in hook payload; skipping")
        return None
    try:
        key = helper(source)
        return str(key) if key else None
    except Exception:
        logger.debug("session-reset-policy: session key derivation failed; skipping", exc_info=True)
        return None


class _CommandFreeEvent:
    """Minimal event stand-in: same source, no command arguments."""

    __slots__ = ("source",)

    def __init__(self, source: Any) -> None:
        self.source = source

    def get_command_args(self) -> str:
        return ""


def _command_free_event(event: Any) -> Any:
    """A copy of *event* that reads as a bare command with no arguments.

    The reset funnel treats ``event.get_command_args()`` as ``/new <title>`` and
    would title the fresh session from it — and ``MessageEvent.get_command_args``
    returns the WHOLE message text for a non-command. Stripping the text keeps a
    policy reset identical to a plain ``/new``. Falls back to a source-only
    stand-in for non-dataclass events.
    """
    try:
        return dataclasses.replace(event, text="")
    except Exception:
        return _CommandFreeEvent(getattr(event, "source", None))


def _reset_via_store(source: Any, session_store: Any, session_key: str) -> Any:
    """Degraded reset without the gateway funnel: rotate the store and fire what
    is reachable from the plugin surface — in-flight async delegations end with
    the old conversation (as in ``/new``), then the ``on_session_finalize`` /
    ``on_session_reset`` lifecycle hooks report the boundary. No run-generation
    bump, cached-agent eviction, conversation-scope clear, or gateway
    ``session:end``/``session:reset`` hooks: those live on the funnel path.
    """
    old_entry = _lookup_entry(session_store, session_key)
    old_sid = getattr(old_entry, "session_id", None)
    with contextlib.suppress(Exception):
        from tools.async_delegation import interrupt_for_session
        interrupt_for_session(session_key=session_key, reason="session_reset",
                              parent_session_id=str(old_sid or ""))
    new_entry = session_store.reset_session(session_key)
    new_sid = getattr(new_entry, "session_id", None) if new_entry is not None else None
    try:
        from hermes_cli.lifecycle import invoke_hook
        platform = str(getattr(getattr(source, "platform", None), "value", "") or "")
        invoke_hook("on_session_finalize", session_id=old_sid, platform=platform,
                    reason="new_session", old_session_id=old_sid, new_session_id=new_sid)
        invoke_hook("on_session_reset", session_id=new_sid, platform=platform,
                    reason="new_session", old_session_id=old_sid, new_session_id=new_sid)
    except Exception:
        logger.debug("session-reset-policy: lifecycle boundary hooks failed", exc_info=True)
    return new_entry


async def _reset_session(gateway: Any, event: Any, source: Any, session_store: Any,
                         session_key: str) -> Any:
    """Rotate *session_key* through the gateway's ``/new`` reset funnel when
    reachable, else the degraded store rotation. Returns the new entry or None.
    """
    funnel = getattr(gateway, "_handle_reset_command", None)
    if callable(funnel):
        old_sid = getattr(_lookup_entry(session_store, session_key), "session_id", None)
        try:
            outcome = funnel(_command_free_event(event))
            if inspect.isawaitable(outcome):
                await outcome
            return _lookup_entry(session_store, session_key)
        except Exception as exc:
            # Failure AFTER the rotation (e.g. banner/tip bookkeeping in the
            # funnel's tail) must not rotate a second time.
            new_entry = _lookup_entry(session_store, session_key)
            new_sid = getattr(new_entry, "session_id", None)
            if new_entry is not None and new_sid != old_sid:
                logger.warning(
                    "session-reset-policy: gateway reset funnel failed after rotating %s; "
                    "keeping its result", session_key, exc_info=True)
                return new_entry
            if _store_busy(exc):
                # The session is executing (a race the pre-check could not close): the
                # store guard refused the rotation. Defer; do NOT fall through to the
                # bare store rotation, which would refuse identically.
                raise _SessionBusy(session_key) from exc
            logger.warning(
                "session-reset-policy: gateway reset funnel failed for %s; falling back to a "
                "bare store rotation", session_key, exc_info=True)
    else:
        logger.debug(
            "session-reset-policy: gateway reset funnel not available for %s; using the "
            "degraded store rotation", session_key)
    try:
        return _reset_via_store(source, session_store, session_key)
    except Exception as exc:
        if _store_busy(exc):
            raise _SessionBusy(session_key) from exc
        raise


_CONTEXT_GUARD_MODULE: Any = None
_CONTEXT_GUARD_LOADED = False


def _load_context_guard() -> Any:
    """Import the sibling ``context_guard`` module under either loader shape.

    Production loads this plugin as a package (``spec_from_file_location`` with
    submodule search locations), so the relative import resolves. The standalone
    test harness loads ``__init__.py`` as a bare module, where a relative import
    cannot work — so fall back to loading ``context_guard.py`` by path. Result is
    cached; returns None (guard disabled) on any failure.
    """
    global _CONTEXT_GUARD_MODULE, _CONTEXT_GUARD_LOADED
    if _CONTEXT_GUARD_LOADED:
        return _CONTEXT_GUARD_MODULE
    _CONTEXT_GUARD_LOADED = True
    mod = None
    try:
        if __package__:
            import importlib
            mod = importlib.import_module(f"{__package__}.context_guard")
    except Exception:
        mod = None
    if mod is None:
        try:
            import importlib.util
            from pathlib import Path
            path = Path(__file__).with_name("context_guard.py")
            spec = importlib.util.spec_from_file_location(f"{__name__}.context_guard", path)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
        except Exception:
            mod = None
            logger.debug("session-reset-policy: context_guard import failed; guard disabled",
                         exc_info=True)
    _CONTEXT_GUARD_MODULE = mod
    return mod


def _observe_context_guard(profile: Optional[str], session_store: Any,
                           key: Optional[str], entry: Any) -> None:
    """Run one shadow pass of the context-limit guard. Never raises, never enforces."""
    try:
        guard = _load_context_guard()
        if guard is None or key is None or entry is None:
            return
        cfg = guard.resolve_config(_read_config_block(profile, "context_guard"))
        guard.observe(cfg, session_store, key, entry)
    except Exception:
        logger.debug("session-reset-policy: context_guard shadow pass failed", exc_info=True)


async def _maybe_reset(event: Any, gateway: Any, session_store: Any,
                       session_key: Optional[str] = None, **_extra: Any) -> Optional[Dict[str, str]]:
    """pre_gateway_dispatch body: reset the session when the policy says so.

    ``session_key`` is consumed only if a future Hermes puts a public one in
    the hook payload (the dispatcher forwards just the payload fields a
    callback declares); otherwise resolution falls to the gateway helper.
    """
    if session_store is None:
        return None
    source = getattr(event, "source", None)
    if source is None:
        return None
    if _is_slash_command(event):
        # Bookkeeping, not conversation (module docstring, "Slash commands"):
        # no clock stamp, no time reset. /new and /reset rotate the route
        # themselves; resetting here first would rotate it twice.
        return None

    profile = getattr(source, "profile", None)
    key = _resolve_session_key(gateway, source, session_key)
    entry = _lookup_entry(session_store, key)
    if key is None or entry is None:
        return None  # no key or no route yet -> brand-new conversation anyway

    # Phase 1: context-limit shadow instrumentation (log-only, isolated). Runs on
    # every inbound message, independent of the session_reset policy below.
    _observe_context_guard(profile, session_store, key, entry)

    policy = _read_session_reset_policy(profile)
    if not policy:
        return None

    now = datetime.now()
    reason = _time_reset_reason(policy, _last_user_inbound(entry), now)
    if reason is None:
        # Not past the boundary: this message IS the user activity — advance
        # our own clock (never ``updated_at``; see module docstring).
        try:
            session_store.set_session_metadata(key, _META_KEY, now.isoformat())
        except Exception:
            logger.debug("session_reset clock stamp failed", exc_info=True)
        return None
    if _session_in_flight(entry, session_store, key):
        logger.info(
            "session-reset-policy: boundary '%s' reached for %s but a turn is in flight; deferring (clock NOT advanced)",
            reason, key,
        )
        return None  # normal dispatch; the next message re-evaluates the same boundary

    try:
        new_entry = await _reset_session(gateway, event, source, session_store, key)
    except _SessionBusy:
        # The store's guard refused a live turn the pre-check could not see (a race
        # between probe and funnel). Same contract as the in-flight defer above.
        logger.info(
            "session-reset-policy: boundary '%s' reached for %s but the session is busy; "
            "deferring (clock NOT advanced)", reason, key,
        )
        return None
    except Exception:
        logger.warning(
            "session-reset-policy: reset_session(%s) failed; continuing on the existing session",
            key, exc_info=True,
        )
        return None
    new_sid = getattr(new_entry, "session_id", "?") if new_entry is not None else "?"
    logger.info(
        "session-reset-policy: reset %s (reason=%s, new session %s)",
        key, reason, new_sid,
    )
    return None  # always allow normal dispatch; the fresh session picks the message up


def register(ctx: Any) -> None:
    ctx.register_hook("pre_gateway_dispatch", _maybe_reset)
    logger.info(
        "session-reset-policy v%s armed (pre_gateway_dispatch)", __version__,
    )


__all__ = ["register", "__version__"]