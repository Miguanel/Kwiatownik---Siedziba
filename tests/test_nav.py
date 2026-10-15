"""Gorny pasek: zakladki w grupach, nazwa biezacej zakladki w pasku i w tytule karty, logo (favicon)."""
from fastapi.testclient import TestClient

from app.main import app
from app.web.templating import NAV, nav_current


def test_nav_current_matches_longest_prefix():
    assert nav_current("/") == ("/", "Panel")
    assert nav_current("/plants/krwawnik") == ("/plants", "Rosliny")
    assert nav_current("/agents/runs/7") == ("/agents", "Agenci")
    assert nav_current("/tasks/queue") == ("/tasks", "Do zrobienia")
    assert nav_current("/commit") == ("/commit", "Commit")
    assert nav_current("/nie-ma-takiej") is None


def test_every_tab_once():
    hrefs = [h for group in NAV for h, _ in group]
    assert len(hrefs) == len(set(hrefs)) and all(len(g) >= 1 for g in NAV)


def test_header_rendered():
    with TestClient(app) as c:
        html = c.get("/agents").text
    assert "<title>Agenci · Siedziba Kwiatownika</title>" in html
    assert 'rel="icon"' in html and 'class="topnav"' in html
    assert '<a href="/agents" aria-current="page">Agenci</a>' in html
    assert html.count('class="tn-group"') == len(NAV)
    assert '<span class="tn-current">Agenci</span>' in html
