"""Execute the public Streamlit interface, including both real simulation forms."""
from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
pytest.importorskip("stim")
pytest.importorskip("pymatching")

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_public_streamlit_first_render_has_no_exceptions():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert len(app.button) == 2
    assert len(app.selectbox) >= 6
    assert any("Aegis QEC" in item.value for item in app.title)


def test_public_streamlit_both_forms_execute_real_science():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert "public_demo_shot" in app.session_state
    shot = app.session_state["public_demo_shot"]
    assert "logical_failure" in shot
    assert "fired_detector_count" in shot

    app.button[1].click().run()
    assert not app.exception
    assert "public_demo_study" in app.session_state
    study = app.session_state["public_demo_study"]
    assert len(study["points"]) == 4
    assert all(0 <= row["logical_failures"] <= row["shots"] for row in study["points"])
