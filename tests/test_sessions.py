"""Hosted demo: every browser session has its own receipt and decisions."""

from pathlib import Path

import pytest

pytest.importorskip("gradio")

from src.ui import app  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "demo"


def _load(st, name):
    out = app._in_session(app.process_json)(st, str(DEMO / name))
    return out[-1]


def _decide(fn, st, label_prefix):
    token = app._CUR.set(st)
    try:
        opts = app._get_anomaly_row_options(st["receipt"], st["result"],
                                            st["decisions"])
        label = next(o for o in opts if o.startswith(label_prefix))
    finally:
        app._CUR.reset(token)
    return app._in_session(fn)(st, label)[-1]


def test_two_sessions_do_not_share_receipts_or_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path)
    a = _load(None, "demo_clean.json")
    b = _load(None, "demo_multi_fault.json")
    assert a is not b and a["sid"] != b["sid"]
    assert a["receipt"]["receipt_id"] == "demo_clean"
    assert b["receipt"]["receipt_id"] == "demo_multi_fault"

    a = _decide(app.decide_reject, a, "Row T")
    assert a["decisions"] == {"total": "reject"}
    assert b["decisions"] == {}

    # B exports nothing yet (flags pending); A is blocked by its own Reject
    msg_b = app._in_session(app.do_export)(b)[0]
    msg_a = app._in_session(app.do_export)(a)[0]
    assert msg_a.startswith("⛔") and "TOTAL" in msg_a
    assert msg_b.startswith("⛔") and "Row 0" in msg_b

    a = _decide(app.decide_accept, a, "Row T")
    out = app._in_session(app.do_export)(a)
    assert out[0].startswith("✅")
    # A's files live under A's own session folder
    assert all(Path(f).is_relative_to(tmp_path / a["sid"]) for f in out[1])
    # the module-level state used by direct calls was never touched
    assert app._STATE["receipt"] is None or \
        app._STATE["receipt"].get("receipt_id") not in ("demo_clean",)


def test_login_gate_blocks_anonymous_visitors(monkeypatch):
    monkeypatch.setattr(app, "REQUIRE_LOGIN", True)
    out = app._login_gate(app.process_json)(None, str(DEMO / "demo_clean.json"),
                                            None)
    assert "Sign in" in out[0]
    assert out[-1]["receipt"] is None


def test_login_gate_lets_signed_in_visitors_through(monkeypatch):
    monkeypatch.setattr(app, "REQUIRE_LOGIN", True)
    out = app._login_gate(app.process_json)(None, str(DEMO / "demo_clean.json"),
                                            object())
    assert out[-1]["receipt"]["receipt_id"] == "demo_clean"
