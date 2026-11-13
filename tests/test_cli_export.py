"""The terminal review/export path applies the same rules as the UI."""

from pathlib import Path

import pytest

pytest.importorskip("gradio")

from src.ui import cli  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "demo"


def _files(tmp_path):
    return sorted(p.name for p in tmp_path.rglob("*") if p.is_file())


def test_cli_clean_exports(tmp_path):
    assert cli.main([str(DEMO / "demo_clean.json"), "--out", str(tmp_path)]) == 0
    assert _files(tmp_path) == ["audit_export.json", "ledger.csv"]


def test_cli_rejected_total_blocks(tmp_path, capsys):
    rc = cli.main([str(DEMO / "demo_clean.json"), "--reject-total",
                   "--out", str(tmp_path)])
    assert rc == 2 and _files(tmp_path) == []
    assert "TOTAL" in capsys.readouterr().out


def test_cli_unreviewed_anomaly_blocks_then_accept(tmp_path):
    f = str(DEMO / "demo_multi_fault.json")
    assert cli.main([f, "--out", str(tmp_path)]) == 2
    assert cli.main([f, "--accept", "0,2", "--out", str(tmp_path)]) == 2
    assert _files(tmp_path) == []
    assert cli.main([f, "--accept", "0,2,3", "--accept-total",
                     "--out", str(tmp_path)]) == 0


def test_cli_accept_all_keeps_explicit_reject(tmp_path):
    f = str(DEMO / "demo_multi_fault.json")
    assert cli.main([f, "--accept-all", "--reject", "1",
                     "--out", str(tmp_path)]) == 2
