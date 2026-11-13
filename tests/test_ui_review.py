"""Locks the review → export flow of the Gradio app (no browser needed).

Rules (team decision 13/11, src/ui/review.py):
    no warning, no action → auto-accepted
    warning,    no action → auto-rejected
    Accept → accepted · Correct → corrected · Reject → rejected
Export only when every row AND the TOTAL are accepted; otherwise
nothing is written, neither ledger.csv nor audit_export.json.
"""

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("gradio")

from src.ui import app, export, review  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "demo"

# index of each output in app._review_outputs()
TABLE, DROPDOWN, GATE, DETAILS, EXPORT_BTN, DOWNLOAD, LOG = 0, 1, 4, 5, 6, 7, 8


@pytest.fixture(autouse=True)
def _export_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "exports")
    yield


def _load(name):
    return app.process_json(str(DEMO / name))


def _options():
    return app._get_anomaly_row_options(app._STATE["receipt"],
                                        app._STATE["result"],
                                        app._STATE["decisions"])


def _opt(prefix):
    return next(o for o in _options() if o.startswith(prefix))


def _state():
    return review.review_state(app._STATE["receipt"], app._STATE["result"],
                               app._STATE["decisions"])


def _export():
    msg, files = app.do_export()
    return msg, files


def _csv_rows(files):
    path = next(f for f in files if f.endswith(".csv"))
    return list(csv.DictReader(open(path, encoding="utf-8")))


def _written(tmp_path):
    return sorted(p.name for p in (tmp_path / "exports").rglob("*")
                  if p.is_file()) if (tmp_path / "exports").exists() else []


def _dropdown_value(update):
    return update.get("value") if isinstance(update, dict) else None


# ── the status table ─────────────────────────────────────
@pytest.mark.parametrize("flagged,decision,status", [
    (False, None, review.AUTO_ACCEPTED),
    (True, None, review.AUTO_REJECTED),
    (False, "accept", review.ACCEPTED), (True, "accept", review.ACCEPTED),
    (False, "reject", review.REJECTED), (True, "reject", review.REJECTED),
    (True, "correct", review.CORRECTED),
])
def test_status_table(flagged, decision, status):
    assert review.row_status(flagged, decision) == status


# ── scenarios from the brief ─────────────────────────────
def test_clean_receipt_exports_without_any_click(tmp_path):
    _load("demo_clean.json")
    assert _state()["can_export"]
    msg, files = _export()
    assert msg.startswith("✅")
    rows = _csv_rows(files)
    assert [(r["item"], r["review"]) for r in rows] == [
        ("MILK 1L", "auto_accepted"), ("BREAD WHITE", "auto_accepted"),
        ("EGGS 10S", "auto_accepted")]
    audit = json.loads(Path(next(f for f in files if f.endswith(".json")))
                       .read_text(encoding="utf-8"))
    assert audit["receipts"][0]["total_review"] == "auto_accepted"


def test_one_unreviewed_anomaly_blocks_then_accept_unblocks(tmp_path):
    out = _load("demo_incorrect_total_line.json")
    review_out = out[3:]
    assert "Export blocked" in review_out[GATE] and "Row 0" in review_out[GATE]
    assert review_out[EXPORT_BTN]["interactive"] is False
    msg, files = _export()
    assert msg.startswith("⛔") and files is None and _written(tmp_path) == []

    after = app.decide_accept(_opt("Row 0"))
    assert after[EXPORT_BTN]["interactive"] is True
    msg, files = _export()
    assert msg.startswith("✅")
    rows = _csv_rows(files)
    assert len(rows) == 2
    milk = rows[0]
    assert milk["review"] == "accepted" and milk["line_total"] == "9.5"
    assert "incorrect_total" in milk["warnings"]        # trace kept


def test_two_flagged_rows_first_accept_still_blocked():
    _load("demo_duplicate_item.json")          # rows 0 and 2 flagged
    app.decide_accept(_opt("Row 0"))
    st = _state()
    assert not st["can_export"]
    assert [b["key"] for b in st["blockers"]] == [2]
    app.decide_accept(_opt("Row 2"))
    assert _state()["can_export"]


def test_rejecting_a_clean_row_blocks_and_accept_recomputes():
    _load("demo_duplicate_item.json")
    app.accept_all_normal()
    assert _state()["can_export"]
    out = app.decide_reject(_opt("Row 1"))      # RICE 5KG, no warning
    assert out[EXPORT_BTN]["interactive"] is False
    assert [b["key"] for b in _state()["blockers"]] == [1]
    assert "Row 1" in out[GATE]
    out = app.decide_accept(_opt("Row 1"))
    assert out[EXPORT_BTN]["interactive"] is True
    assert _state()["can_export"]


def test_rejected_total_blocks_csv_and_json(tmp_path):
    _load("demo_clean.json")
    out = app.decide_reject(_opt("Row T"))
    assert "TOTAL" in out[GATE]
    msg, files = _export()
    assert msg.startswith("⛔") and "TOTAL" in msg and files is None
    assert _written(tmp_path) == []
    # the export functions refuse too, before writing anything
    stamped = review.apply_review(app._STATE["receipt"], app._STATE["result"],
                                  app._STATE["decisions"])
    with pytest.raises(export.ExportBlocked):
        export.to_ledger_csv([stamped], str(tmp_path / "x.csv"))
    with pytest.raises(export.ExportBlocked):
        export.to_audit_json([stamped], [app._STATE["result"]],
                             str(tmp_path / "x.json"))
    assert not (tmp_path / "x.csv").exists()
    assert not (tmp_path / "x.json").exists()


def test_unreviewed_receipt_cannot_be_exported_directly(tmp_path):
    _load("demo_clean.json")
    raw = app._STATE["receipt"]                       # never stamped
    with pytest.raises(export.ExportBlocked):
        export.to_ledger_csv([raw], str(tmp_path / "x.csv"))
    # with the detector result the gate is recomputed: clean → allowed
    export.to_ledger_csv([raw], str(tmp_path / "y.csv"),
                         results=[app._STATE["result"]])
    assert (tmp_path / "y.csv").exists()


def test_flagged_raw_receipt_blocked_even_with_results(tmp_path):
    _load("demo_multi_fault.json")
    with pytest.raises(export.ExportBlocked) as e:
        export.to_audit_json([app._STATE["receipt"]], [app._STATE["result"]],
                             str(tmp_path / "x.json"))
    assert "auto_rejected" in str(e.value)


def test_multi_fault_flags_each_check_with_its_evidence():
    _load("demo_multi_fault.json")
    st = _state()
    types = {r["key"]: sorted({f["type"] for f in r["flags"]})
             for r in st["rows"] + [st["total"]] if r["flags"]}
    assert types == {0: ["unusual_price"], 2: ["duplicate"],
                     3: ["duplicate"], "total": ["incorrect_total"]}
    assert len(st["blockers"]) == 4
    # Accept all leaves an explicit Reject in place
    app.decide_reject(_opt("Row 3"))
    app.accept_all_normal()
    assert [b["key"] for b in _state()["blockers"]] == [3]


def test_row_with_several_warnings_lists_all_of_them():
    _load("demo_multi_fault.json")
    data, result = app._STATE["receipt"], app._STATE["result"]
    # add a second warning to row 0 the same way the OCR gate would
    data["items"][0]["review"] = "review_required"
    data["items"][0]["reason"] = "low OCR confidence (0.51)"
    result["anomalies"].append({"anomaly_type": "incorrect_total",
                                "reason": "Line 'MILK 1L': test",
                                "evidence": {"line_index": 0},
                                "confidence": 0.9})
    app.select_row(_opt("Row 0"))
    details = app.show_details(_opt("Row 0"))
    assert "2 warning(s)" in details
    assert "unusual_price" in details and "incorrect_total" in details
    html = app._build_html_table(data, result, selected_idx=0)
    row0 = html.split("<tr")[2]
    assert "unusual_price" in row0 and "incorrect_total" in row0


def test_correct_counts_as_accept_and_keeps_the_trace(tmp_path):
    _load("demo_incorrect_total_line.json")
    app.decide_correct(_opt("Row 0"), "7.00")
    assert _state()["can_export"]
    _, files = _export()
    rows = _csv_rows(files)
    assert rows[0]["review"] == "corrected" and rows[0]["line_total"] == "7.0"
    audit = json.loads(Path(next(f for f in files if f.endswith(".json")))
                       .read_text(encoding="utf-8"))
    corr = audit["receipts"][0]["corrections"][0]
    assert corr["from"] == 9.5 and corr["to"] == 7.0
    assert "incorrect_total" in corr["warnings"][0]


def test_any_change_clears_the_download_link():
    _load("demo_clean.json")
    _, files = _export()
    assert files
    out = app.decide_accept(_opt("Row 0"))
    assert out[DOWNLOAD] is None
    out = app.decide_reject(_opt("Row 0"))
    assert out[DOWNLOAD] is None and out[EXPORT_BTN]["interactive"] is False


def test_second_receipt_does_not_inherit_decisions():
    _load("demo_clean.json")
    app.decide_reject(_opt("Row T"))
    assert app._STATE["decisions"] == {"total": "reject"}
    out = _load("demo_incorrect_total_line.json")
    assert app._STATE["decisions"] == {}
    assert out[3 + DOWNLOAD] is None
    st = _state()
    assert st["total"]["status"] == review.AUTO_ACCEPTED
    assert [b["key"] for b in st["blockers"]] == [0]


def test_every_row_is_in_the_dropdown_including_clean_ones():
    _load("demo_incorrect_total_line.json")
    opts = _options()
    assert opts[0].startswith("Row 0: MILK 1L (incorrect_total)")
    assert opts[1].startswith("Row 1: BREAD WHITE (no warning)")
    assert opts[-1].startswith("Row T: TOTAL (no warning)")


# ── 62 / 31 and the coverage guard ───────────────────────
def test_case_a_duplicate_explains_the_total_gap():
    _load("demo_duplicate_62_31.json")
    res = app._STATE["result"]
    tc = res["total_check"]
    assert tc["status"] == "mismatch_explained_by_duplicate"
    assert tc["difference"] == -31.0 and tc["duplicate_rows"] == [0, 1]
    # the gap travels with the duplicate warning; no second, separate
    # incorrect_total anomaly (the error set counts this fault once)
    assert [a["anomaly_type"] for a in res["anomalies"]] == ["duplicate"]
    [dup] = res["anomalies"]
    assert "RM 62.00" in dup["reason"] and "RM 31.00" in dup["reason"]
    assert dup["evidence"]["total_gap"]["difference"] == -31.0
    total = _state()["total"]
    assert total["status"] == review.AUTO_REJECTED
    assert [f["type"] for f in total["flags"]] == ["total_vs_duplicate"]
    # both rows and the TOTAL must be decided before export
    assert {b["key"] for b in _state()["blockers"]} == {0, 1, "total"}


def test_case_b_missing_rows_is_not_a_pass_and_not_a_mismatch():
    out = _load("demo_missing_rows.json")
    res = app._STATE["result"]
    assert not any(a["anomaly_type"] == "incorrect_total"
                   for a in res["anomalies"])
    tc = res["total_check"]
    assert tc["status"] == "insufficient"
    assert tc["difference"] == 14.0 and tc["unreadable_rows"] == 1
    assert "Not enough data" in tc["reason"]
    assert "total not verified" in out[2]
    total = _state()["total"]
    assert total["status"] == review.AUTO_REJECTED
    assert total["flags"][0]["type"] == "total_unverified"
    app.decide_accept(_opt("Row T"))
    assert _state()["can_export"]


# ── unchanged UI behaviour ───────────────────────────────
def test_decision_can_be_changed():
    _load("demo_unusual_price.json")
    first = _opt("Row 0")
    app.decide_reject(first)
    again = _opt("Row 0")
    assert again.endswith("— rejected")
    app.decide_accept(again)
    assert app._STATE["decisions"]["0"] == "accept"


def test_anomaly_rows_have_readable_text_colour():
    _load("demo_multi_fault.json")
    html = app._build_html_table(app._STATE["receipt"], app._STATE["result"],
                                 selected_idx=None)
    assert "background-color:#fde2dd;color:#1b1b1b" in html
    assert "background-color:#f2d5d0;\"" not in html


def test_selecting_a_row_highlights_that_row():
    _load("demo_multi_fault.json")
    html, _ = app.select_row(_opt("Row 2"))
    rows = html.split("<tr")[2:]            # skip <thead> row
    assert "#f9c6bd" in rows[2] and "#f9c6bd" not in rows[0]


def test_dropdown_keeps_the_decided_row_selected():
    _load("demo_multi_fault.json")
    out = app.decide_accept(_opt("Row 2"))
    assert _dropdown_value(out[DROPDOWN]).startswith("Row 2")
    assert _dropdown_value(out[DROPDOWN]).endswith("— accepted")


def test_total_row_can_be_selected_and_decided():
    _load("demo_multi_fault.json")
    total = _opt("Row T")
    html, _ = app.select_row(total)
    assert "TOTAL" in html.split("<tr")[-1] and "#f9c6bd" in html.split("<tr")[-1]
    out = app.decide_reject(total)
    assert _dropdown_value(out[DROPDOWN]).startswith("Row T")
    assert app._STATE["decisions"]["total"] == "reject"


def test_css_scoped_above_gradio_prose_rule():
    """Gradio 6 `.prose *` colours every <td>; our rules need an id scope."""
    _load("demo_multi_fault.json")
    html = app._build_html_table(app._STATE["receipt"], app._STATE["result"])
    assert '<div id="da-wrap">' in html
    assert "#da-wrap .da-table td {" in html and "color:#1b1b1b" in html
