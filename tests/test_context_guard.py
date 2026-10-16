"""Tests for the Phase 1 (shadow) context-limit instrumenter.

Standalone-suite discipline (same as the plugin suite): no Hermes import — the
guard module is stdlib-only and is loaded directly by path. The store and db are
fakes; the point of these tests is the CONTRACT: the guard measures, classifies
and LOGS, and never mutates a conversation (no reset, no compaction) — plus the
threshold and cooldown bookkeeping.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
GUARD_PATH = PLUGIN_DIR / "context_guard.py"
NOW = datetime(2026, 1, 1, 12, 0, 0)


@pytest.fixture()
def guard():
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.hermes_session_reset_policy.context_guard", GUARD_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


class FakeDB:
    """Minimal SessionDB stand-in: ``_read_one`` answers the guard's two queries."""

    def __init__(self, sessions_row=None, active_tokens=None):
        self.sessions_row = sessions_row
        self.active_tokens = active_tokens

    def _read_one(self, sql, params=()):
        if "FROM sessions" in sql:
            return self.sessions_row
        if "SUM(token_count)" in sql:
            return (self.active_tokens,) if self.active_tokens is not None else (0,)
        return None


class FakeStore:
    """SessionStore stand-in. ``reset_session`` records — and must never be called."""

    def __init__(self, db):
        self._db = db
        self.metadata = {}
        self.reset_calls = []
        self.rotate_calls = []
        self.writes = 0

    def _db_for_key(self, key):
        return self._db

    def get_session_metadata(self, key, name, default=None):
        return self.metadata.get((key, name), default)

    def set_session_metadata(self, key, name, value):
        self.metadata[(key, name)] = value
        self.writes += 1
        return True

    def reset_session(self, key):
        self.reset_calls.append(key)

    def rotate(self, key):
        self.rotate_calls.append(key)


def _entry(session_id="s-1"):
    return SimpleNamespace(session_id=session_id)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_resolve_config_merges_defaults_and_passes_unknown_keys(guard):
    cfg = guard.resolve_config({"max_tokens": 5000, "future_knob": 7})
    assert cfg["max_tokens"] == 5000
    assert cfg["hard_tokens"] == guard.DEFAULTS["hard_tokens"]
    assert cfg["future_knob"] == 7


def test_resolve_config_tolerates_non_dict(guard):
    assert guard.resolve_config(None) == guard.DEFAULTS


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def test_measure_reads_sessions_and_active_context(guard):
    store = FakeStore(FakeDB(sessions_row=(100, 40, 3, 9), active_tokens=137))
    m = guard.measure(store, "k", _entry())
    assert m["session_id"] == "s-1"
    assert m["input_tokens"] == 100 and m["output_tokens"] == 40
    assert m["billed_tokens"] == 140
    assert m["active_context_tokens"] == 137
    assert m["active_context_source"] == "messages"       # no anchor -> message SUM
    assert m["api_calls"] == 3 and m["message_count"] == 9


def test_measure_prefers_usage_anchor(guard):
    cfg = json.dumps({"_usage_anchor": {"prompt_tokens": 149992}})
    store = FakeStore(FakeDB(sessions_row=(432485, 17142, 44, 102, cfg), active_tokens=0))
    m = guard.measure(store, "k", _entry())
    assert m["active_context_tokens"] == 149992
    assert m["active_context_source"] == "usage_anchor"
    assert m["billed_tokens"] == 449627


def test_measure_anchor_bad_json_falls_back_to_messages(guard):
    store = FakeStore(FakeDB(sessions_row=(100, 40, 3, 9, "not-json"), active_tokens=137))
    m = guard.measure(store, "k", _entry())
    assert m["active_context_tokens"] == 137 and m["active_context_source"] == "messages"


def test_measure_dead_token_count_falls_through_to_billed(guard):
    # messages.token_count NULL (SUM 0) and no anchor -> no live metric, source none
    store = FakeStore(FakeDB(sessions_row=(100, 40, 3, 9, None), active_tokens=0))
    m = guard.measure(store, "k", _entry())
    assert m["active_context_tokens"] is None and m["active_context_source"] == "none"
    cfg = guard.resolve_config({"max_tokens": 100, "hard_tokens": 200})
    assert guard.evaluate(cfg, m, NOW)["level"] == "soft"   # falls back to billed 140


def test_measure_fail_open_without_db(guard):
    store = SimpleNamespace()  # no _db_for_key / _db
    m = guard.measure(store, "k", _entry())
    assert m["active_context_tokens"] is None and m["billed_tokens"] is None


# ---------------------------------------------------------------------------
# evaluate — boundary math
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("active,level", [
    (0, "ok"), (999, "ok"), (1000, "soft"), (1999, "soft"), (2000, "hard"), (5000, "hard"),
])
def test_evaluate_levels(guard, active, level):
    cfg = guard.resolve_config({"max_tokens": 1000, "hard_tokens": 2000})
    verdict = guard.evaluate(cfg, {"active_context_tokens": active, "billed_tokens": 0}, NOW)
    assert verdict["level"] == level


def test_evaluate_unknown_when_unmeasured(guard):
    cfg = guard.resolve_config({})
    verdict = guard.evaluate(cfg, {"active_context_tokens": None, "billed_tokens": None}, NOW)
    assert verdict["level"] == "unknown"


def test_evaluate_falls_back_to_billed_when_active_missing(guard):
    cfg = guard.resolve_config({"max_tokens": 100, "hard_tokens": 200})
    verdict = guard.evaluate(cfg, {"active_context_tokens": None, "billed_tokens": 150}, NOW)
    assert verdict["level"] == "soft"


# ---------------------------------------------------------------------------
# observe — logs, never mutates
# ---------------------------------------------------------------------------

def _caplog_for(caplog, level=logging.INFO):
    caplog.set_level(level, logger="hermes.plugins.session_reset_policy.context_guard")


def test_observe_shadow_logs_and_never_mutates(guard, caplog):
    store = FakeStore(FakeDB(sessions_row=(100, 40, 3, 9), active_tokens=120))
    cfg = guard.resolve_config({})  # defaults: max 120000 -> ok
    _caplog_for(caplog)
    verdict = guard.observe(cfg, store, "k", _entry(), now=NOW)
    assert verdict["level"] == "ok"
    assert any("context_guard: shadow_log" in r.message for r in caplog.records)
    assert store.reset_calls == [] and store.rotate_calls == []
    # first pass seeds the cooldown blob (exactly one write)
    assert ("k", "srp_context_guard") in store.metadata
    assert store.writes == 1


def test_observe_steady_state_is_zero_write(guard, caplog):
    store = FakeStore(FakeDB(sessions_row=(100, 40, 3, 9), active_tokens=120))
    cfg = guard.resolve_config({})
    _caplog_for(caplog)
    guard.observe(cfg, store, "k", _entry(), now=NOW)          # seeds
    assert store.writes == 1
    guard.observe(cfg, store, "k", _entry(), now=NOW)          # no breach/change -> no write
    guard.observe(cfg, store, "k", _entry(), now=NOW)
    assert store.writes == 1
    assert store.reset_calls == [] and store.rotate_calls == []


def test_observe_soft_breach_warns_once_within_cooldown(guard, caplog):
    store = FakeStore(FakeDB(sessions_row=(10, 20, 3, 8), active_tokens=150))
    cfg = guard.resolve_config({"max_tokens": 100, "hard_tokens": 200, "cooldown_seconds": 300})
    _caplog_for(caplog, logging.WARNING)
    guard.observe(cfg, store, "k", _entry(), now=NOW)
    guard.observe(cfg, store, "k", _entry(), now=NOW)  # same instant -> cooled down
    warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("soft_cap_reached" in w for w in warns) == 1
    # first pass writes the breach/cooldown stamp; the cooled-down second pass does not
    assert store.writes == 1
    assert store.reset_calls == []


def test_observe_hard_breach_warns(guard, caplog):
    store = FakeStore(FakeDB(sessions_row=(10, 20, 3, 8), active_tokens=250))
    cfg = guard.resolve_config({"max_tokens": 100, "hard_tokens": 200})
    _caplog_for(caplog, logging.WARNING)
    guard.observe(cfg, store, "k", _entry(), now=NOW)
    assert any("hard_cap_exceeded" in r.message and "triggering_compaction=true" in r.message
               for r in caplog.records)
    assert store.reset_calls == []


def test_observe_skips_below_min_turns(guard, caplog):
    store = FakeStore(FakeDB(sessions_row=(10, 20, 1, 2), active_tokens=250))
    cfg = guard.resolve_config({"max_tokens": 100, "hard_tokens": 200, "min_turns": 5})
    _caplog_for(caplog, logging.INFO)
    guard.observe(cfg, store, "k", _entry(), now=NOW)
    assert not any("shadow_log" in r.message for r in caplog.records)
    # seeds the blob once (no shadow_log, no breach) and never writes again unchanged
    blob = store.metadata[("k", "srp_context_guard")]
    assert blob["version"] == guard._STATE_VERSION
    guard.observe(cfg, store, "k", _entry(), now=NOW)
    assert store.writes == 1


def test_observe_enabled_in_shadow_warns_but_acts_on_nothing(guard, caplog):
    store = FakeStore(FakeDB(sessions_row=(10, 20, 3, 8), active_tokens=999))
    cfg = guard.resolve_config({"enabled": True, "shadow": False, "max_tokens": 100, "hard_tokens": 200})
    _caplog_for(caplog, logging.WARNING)
    guard.observe(cfg, store, "k", _entry(), now=NOW)
    assert any("shadow-only" in r.message for r in caplog.records)
    assert store.reset_calls == [] and store.rotate_calls == []


def test_observe_no_route_returns_none(guard):
    store = FakeStore(FakeDB())
    assert guard.observe(guard.resolve_config({}), store, None, _entry(), now=NOW) is None
    assert guard.observe(guard.resolve_config({}), store, "k", None, now=NOW) is None
