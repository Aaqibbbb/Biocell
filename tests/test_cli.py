"""Tests for the command-line interface."""

from __future__ import annotations

import pytest

from cellgraphfm.cli import main


def test_version():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "gcn" in out
    assert "UNI" in out


def test_demo(capsys):
    rc = main(["demo", "--cells", "60", "--method", "knn", "--k", "4", "--permutations", "20"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TissueGraph" in out
    assert "neighbourhood enrichment" in out


def test_demo_delaunay(capsys):
    rc = main(["demo", "--cells", "40", "--method", "delaunay", "--permutations", "10"])
    assert rc == 0
    assert "TissueGraph" in capsys.readouterr().out


def test_no_command(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()
