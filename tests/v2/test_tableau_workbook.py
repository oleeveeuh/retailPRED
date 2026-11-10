from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = REPO_ROOT / "bi/tableau/retailpred_v2.twb"
EXTRACT = REPO_ROOT / "bi/tableau/retailpred_v2.hyper"


def test_tableau_workbook_has_governed_extract_and_views() -> None:
    root = ET.parse(WORKBOOK).getroot()

    datasources = root.find("datasources")
    worksheets = root.find("worksheets")
    windows = root.find("windows")
    assert datasources is not None
    assert worksheets is not None
    assert windows is not None

    assert [node.attrib["caption"] for node in datasources] == [
        "Model Performance",
        "M5 Hierarchy Accuracy",
        "M5 Uncertainty Audit",
        "MRTS Regime Accuracy",
        "Data Status",
        "MRTS Research Windows",
        "MRTS Feature Ladder",
        "MRTS Horizon Accuracy",
        "MRTS Candidate Comparisons",
    ]
    assert [node.attrib["name"] for node in worksheets] == [
        "01 Model Scorecard",
        "02 M5 Hierarchy",
        "03 Uncertainty Audit",
        "04 MRTS Regimes",
        "05 Data Status",
        "06 MRTS Research Windows",
        "07 MRTS Feature Ladder",
        "08 MRTS Horizon Accuracy",
        "09 MRTS Candidate Comparisons",
    ]
    assert len(windows) == 9

    connections = root.findall(".//connection[@class='hyper']")
    assert len(connections) == 9
    assert {node.attrib["dbname"] for node in connections} == {
        "retailpred_v2.hyper"
    }
    assert EXTRACT.is_file()
    assert EXTRACT.stat().st_size > 0


def test_tableau_workbook_does_not_reference_public_profile() -> None:
    workbook_text = WORKBOOK.read_text(encoding="utf-8").lower()

    assert "public.tableau.com" not in workbook_text
    assert "password=" not in workbook_text
