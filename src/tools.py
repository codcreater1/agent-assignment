"""Mock tools the agent can call.

Each tool returns a JSON-serialisable dict. On bad input or simulated
failure they return ``{"ok": False, "error": "..."}`` so the agent can
read the error and decide what to do next instead of crashing.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Any


# In-memory "database" so reminders and bookings persist across calls
# inside one agent run.
_REMINDERS: list[dict[str, Any]] = []
_BOOKINGS: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Mock seed data
# ---------------------------------------------------------------------------

_SEARCH_DB: list[dict[str, Any]] = [
    # Dentists
    {"id": "dent-001", "name": "Smile Clinic", "category": "dentist",
     "city": "warsaw", "price": 80, "rating": 4.6,
     "open_hours": "09:00-19:00"},
    {"id": "dent-002", "name": "City Dental", "category": "dentist",
     "city": "warsaw", "price": 120, "rating": 4.8,
     "open_hours": "10:00-20:00"},
    {"id": "dent-003", "name": "Dr. Kowalski Clinic", "category": "dentist",
     "city": "warsaw", "price": 60, "rating": 4.3,
     "open_hours": "08:00-18:00"},
    {"id": "dent-004", "name": "Antalya Dental Studio", "category": "dentist",
     "city": "antalya", "price": 70, "rating": 4.5,
     "open_hours": "10:00-20:00"},
    {"id": "dent-005", "name": "Falez Dis Klinigi", "category": "dentist",
     "city": "antalya", "price": 55, "rating": 4.4,
     "open_hours": "09:00-19:00"},

    # Coworking
    {"id": "cowork-001", "name": "Brain Embassy",
     "category": "coworking", "city": "warsaw", "price": 18,
     "rating": 4.7, "perks": ["wifi", "coffee", "meeting_room"]},
    {"id": "cowork-002", "name": "Mindspace Warsaw",
     "category": "coworking", "city": "warsaw", "price": 25,
     "rating": 4.5, "perks": ["wifi", "coffee"]},
    {"id": "cowork-003", "name": "Business Link",
     "category": "coworking", "city": "warsaw", "price": 15,
     "rating": 4.2, "perks": ["wifi"]},
    {"id": "cowork-004", "name": "The Heart",
     "category": "coworking", "city": "warsaw", "price": 19,
     "rating": 4.6, "perks": ["wifi", "coffee", "phone_booth"]},

    # Hotels (for trip planning)
    {"id": "hotel-001", "name": "Hotel Praha Centre",
     "category": "hotel", "city": "prague", "price_per_night": 75,
     "rating": 4.4},
    {"id": "hotel-002", "name": "Old Town Hostel",
     "category": "hotel", "city": "prague", "price_per_night": 35,
     "rating": 4.1},

    # Flights / transport
    {"id": "trans-001", "name": "Bus to Prague",
     "category": "transport", "city": "prague", "price": 40,
     "duration_h": 7},
    {"id": "trans-002", "name": "Train to Prague",
     "category": "transport", "city": "prague", "price": 60,
     "duration_h": 8},
]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def calendar_check(start_date: str, end_date: str) -> dict[str, Any]:
    """Return free slots between ``start_date`` and ``end_date`` (YYYY-MM-DD)."""
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError as exc:
        return {"ok": False,
                "error": f"Invalid date format: {exc}. Use YYYY-MM-DD."}

    if end < start:
        return {"ok": False,
                "error": "end_date must be on or after start_date."}

    # Pretend the user is busy 10:00-12:00 and 14:00-16:00 every weekday.
    free_slots: list[dict[str, str]] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:  # Mon-Fri
            for hour in (9, 13, 17, 18):
                slot_start = cursor.replace(hour=hour, minute=0)
                slot_end = slot_start + timedelta(hours=1)
                free_slots.append({
                    "start": slot_start.isoformat(timespec="minutes"),
                    "end": slot_end.isoformat(timespec="minutes"),
                })
        cursor += timedelta(days=1)

    return {"ok": True, "free_slots": free_slots}


def search_service(query: str | None = None,
                   category: str | None = None,
                   city: str | None = None,
                   max_price: float | None = None) -> dict[str, Any]:
    """Keyword search over a small mock catalogue.

    All filters are optional, but at least one of ``query``, ``category``,
    or ``city`` must be supplied so we don't return the entire catalogue.
    """
    if not any([query and query.strip(), category, city]):
        return {"ok": False,
                "error": "Provide at least one of query, category, or city."}

    q = (query or "").strip().lower()
    results: list[dict[str, Any]] = []
    for item in _SEARCH_DB:
        if q:
            haystack = " ".join(str(v).lower() for v in item.values())
            if q not in haystack and not any(w in haystack for w in q.split()):
                continue
        if category and item.get("category") != category.lower():
            continue
        if city and item.get("city") != city.lower():
            continue
        if max_price is not None:
            price = item.get("price") or item.get("price_per_night")
            if price is None or price > max_price:
                continue
        results.append(item)

    if not results:
        return {"ok": True, "results": [],
                "note": "No matches. Try broadening the query or removing filters."}

    return {"ok": True, "results": results}


def booking_service(option_id: str,
                    when: str | None = None,
                    notes: str | None = None) -> dict[str, Any]:
    """Book an item by id. Simulates a 10% transient failure rate."""
    if not option_id:
        return {"ok": False, "error": "option_id is required."}

    item = next((x for x in _SEARCH_DB if x["id"] == option_id), None)
    if item is None:
        return {"ok": False,
                "error": f"Unknown option_id '{option_id}'. "
                         "Call search_service first to get a valid id."}

    # Simulate transient failure so the agent has to handle it.
    if random.random() < 0.10:
        return {"ok": False,
                "error": "Booking provider timed out. Safe to retry once."}

    confirmation = f"BK-{uuid.uuid4().hex[:8].upper()}"
    record = {"confirmation": confirmation, "item": item,
              "when": when, "notes": notes}
    _BOOKINGS.append(record)
    return {"ok": True, **record}


def reminder_create(title: str,
                    when: str,
                    notes: str | None = None) -> dict[str, Any]:
    """Create a reminder. ``when`` should be ISO 8601."""
    if not title:
        return {"ok": False, "error": "title is required."}
    try:
        datetime.fromisoformat(when)
    except ValueError:
        return {"ok": False,
                "error": "when must be ISO 8601, e.g. 2026-05-12T17:30."}

    rid = f"REM-{uuid.uuid4().hex[:6].upper()}"
    record = {"id": rid, "title": title, "when": when, "notes": notes}
    _REMINDERS.append(record)
    return {"ok": True, **record}


# ---------------------------------------------------------------------------
# Tool registry — exposed to the agent / OpenAI function calling
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "calendar_check": calendar_check,
    "search_service": search_service,
    "booking_service": booking_service,
    "reminder_create": reminder_create,
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calendar_check",
            "description": "List free calendar slots between two dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string",
                                   "description": "Start date YYYY-MM-DD."},
                    "end_date": {"type": "string",
                                 "description": "End date YYYY-MM-DD."},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_service",
            "description": (
                "Search a catalogue of services (dentists, coworking spaces, "
                "hotels, transport). Use a short keyword for `query` (e.g. "
                "'dentist', 'coworking') — do NOT pass the user's full "
                "request. At least one of `query`, `category`, or `city` "
                "is required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Short keyword, e.g. 'dentist'."},
                    "category": {"type": "string",
                                 "enum": ["dentist", "coworking",
                                          "hotel", "transport"]},
                    "city": {"type": "string"},
                    "max_price": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "booking_service",
            "description": "Book a previously found option by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "option_id": {"type": "string"},
                    "when": {"type": "string",
                             "description": "ISO 8601 timestamp."},
                    "notes": {"type": "string"},
                },
                "required": ["option_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reminder_create",
            "description": "Create a reminder for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "when": {"type": "string",
                             "description": "ISO 8601 timestamp."},
                    "notes": {"type": "string"},
                },
                "required": ["title", "when"],
            },
        },
    },
]
