"""Unit tests for src/tools.py — no network, no LLM involved."""

from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture()
def tools(tmp_path, monkeypatch):
    """Reload src.tools with a fresh, isolated storage file per test."""
    monkeypatch.setenv("AGENT_DATA_FILE", str(tmp_path / "data.json"))
    from src import storage, tools as tools_mod
    importlib.reload(storage)
    importlib.reload(tools_mod)
    return tools_mod


def test_calendar_check_valid_range(tools):
    result = tools.calendar_check("2026-08-03", "2026-08-04")  # Mon-Tue
    assert result["ok"] is True
    assert len(result["free_slots"]) == 8  # 4 slots/day * 2 weekdays


def test_calendar_check_invalid_dates(tools):
    result = tools.calendar_check("not-a-date", "2026-08-04")
    assert result["ok"] is False
    assert "Invalid date format" in result["error"]


def test_calendar_check_end_before_start(tools):
    result = tools.calendar_check("2026-08-04", "2026-08-03")
    assert result["ok"] is False


def test_search_requires_a_filter(tools):
    result = tools.search_service()
    assert result["ok"] is False


def test_search_by_category_and_city(tools):
    result = tools.search_service(category="coworking", city="Warsaw")
    assert result["ok"] is True
    assert len(result["results"]) == 4
    assert all(r["category"] == "coworking" for r in result["results"])


def test_search_respects_max_price(tools):
    result = tools.search_service(category="coworking", max_price=16)
    assert result["ok"] is True
    assert [r["id"] for r in result["results"]] == ["cowork-003"]


def test_search_tolerates_typos(tools):
    exact = tools.search_service(query="dentist")
    typo = tools.search_service(query="dentst")
    assert exact["ok"] is True and typo["ok"] is True
    assert {r["id"] for r in typo["results"]} == {r["id"] for r in exact["results"]}


def test_search_no_matches_still_ok(tools):
    result = tools.search_service(category="dentist", city="atlantis")
    assert result["ok"] is True
    assert result["results"] == []


def test_booking_unknown_id(tools):
    result = tools.booking_service("does-not-exist")
    assert result["ok"] is False
    assert "Unknown option_id" in result["error"]


def test_booking_success_and_persistence(tools, monkeypatch):
    monkeypatch.setattr("src.tools.random.random", lambda: 0.99)  # no transient fail
    result = tools.booking_service("cowork-003", when="2026-08-03T09:00")
    assert result["ok"] is True
    assert result["confirmation"].startswith("BK-")

    listed = tools.list_bookings()
    assert listed["ok"] is True
    assert len(listed["bookings"]) == 1
    assert listed["bookings"][0]["confirmation"] == result["confirmation"]

    # Confirm it was actually flushed to disk, not just kept in memory.
    from src import storage
    on_disk = storage.load()
    assert len(on_disk["bookings"]) == 1


def test_booking_transient_failure_is_reported(tools, monkeypatch):
    monkeypatch.setattr("src.tools.random.random", lambda: 0.01)  # force fail
    result = tools.booking_service("cowork-003")
    assert result["ok"] is False
    assert "retry" in result["error"].lower()


def test_reminder_create_and_list(tools):
    result = tools.reminder_create("Call dentist", "2026-08-03T09:00")
    assert result["ok"] is True
    assert result["id"].startswith("REM-")

    listed = tools.list_reminders()
    assert len(listed["reminders"]) == 1


def test_reminder_requires_valid_iso_datetime(tools):
    result = tools.reminder_create("Call dentist", "next tuesday")
    assert result["ok"] is False


def test_reminder_requires_title(tools):
    result = tools.reminder_create("", "2026-08-03T09:00")
    assert result["ok"] is False


def test_state_survives_reload(tools, monkeypatch):
    """Simulates a follow-up CLI invocation: new process, same data file."""
    monkeypatch.setattr("src.tools.random.random", lambda: 0.99)
    tools.booking_service("cowork-001")
    tools.reminder_create("Follow up", "2026-08-05T10:00")

    import importlib
    from src import tools as tools_mod
    reloaded = importlib.reload(tools_mod)

    assert len(reloaded.list_bookings()["bookings"]) == 1
    assert len(reloaded.list_reminders()["reminders"]) == 1
