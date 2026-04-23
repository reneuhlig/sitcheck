#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

ROOT = Path(__file__).resolve().parents[2]
DB_FILE = ROOT / "admin_bookings_direct_test.db"
if DB_FILE.exists():
    DB_FILE.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{DB_FILE}"
os.environ["DEFAULT_ZONE_ID"] = "default-zone"
os.environ["DEFAULT_ZONE_CAPACITY"] = "100"

api_gateway_dir = ROOT / "apps" / "api-gateway"
if str(api_gateway_dir) not in sys.path:
    sys.path.insert(0, str(api_gateway_dir))

module_path = api_gateway_dir / "main.py"
spec = importlib.util.spec_from_file_location("api_admin_bookings_direct_main", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["api_admin_bookings_direct_main"] = module
spec.loader.exec_module(module)
module.startup()


def make_request(token: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if token:
        headers.append((b"cookie", f"{module.SESSION_COOKIE_NAME}={token}".encode("utf-8")))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
    )


with module.SessionLocal() as db:
    module.auth_register(
        module.AuthRegisterRequest(username="adminpaul", password="secret123", role="admin"),
        make_request(),
        Response(),
        db,
    )
    module.auth_register(
        module.AuthRegisterRequest(username="userlisa", password="secret123", role="user"),
        make_request(),
        Response(),
        db,
    )

    admin_user = (
        db.execute(module.select(module.User).where(module.User.username_normalized == "adminpaul"))
        .scalars()
        .first()
    )
    normal_user = (
        db.execute(module.select(module.User).where(module.User.username_normalized == "userlisa"))
        .scalars()
        .first()
    )
    assert admin_user is not None
    assert normal_user is not None

    _, admin_token = module._create_user_session(db, user_id=admin_user.user_id)
    _, user_token = module._create_user_session(db, user_id=normal_user.user_id)

    day_start = module._local_day_start_utc()
    zone = module._get_zone_or_404(db, module.DEFAULT_ZONE_ID)
    db.add_all(
        [
            module.Booking(
                booking_id="booking-admin-today",
                user_id=admin_user.user_id,
                zone_id=zone.zone_id,
                starts_at=day_start + timedelta(hours=10),
                ends_at=day_start + timedelta(hours=12),
                party_size=1,
                status="confirmed",
            ),
            module.Booking(
                booking_id="booking-user-future",
                user_id=normal_user.user_id,
                zone_id=zone.zone_id,
                starts_at=day_start + timedelta(days=1, hours=9),
                ends_at=day_start + timedelta(days=1, hours=11),
                party_size=1,
                status="confirmed",
            ),
            module.Booking(
                booking_id="booking-old",
                user_id=normal_user.user_id,
                zone_id=zone.zone_id,
                starts_at=day_start - timedelta(days=1, hours=3),
                ends_at=day_start - timedelta(days=1, hours=1),
                party_size=1,
                status="confirmed",
            ),
            module.Booking(
                booking_id="booking-cancelled",
                user_id=admin_user.user_id,
                zone_id=zone.zone_id,
                starts_at=day_start + timedelta(days=2, hours=9),
                ends_at=day_start + timedelta(days=2, hours=11),
                party_size=1,
                status="cancelled",
                cancelled_at=datetime.now(UTC),
            ),
        ]
    )
    db.commit()

with module.SessionLocal() as db:
    listed = module.list_admin_bookings(make_request(admin_token), db)
    assert len(listed) == 2, listed
    assert [item.username for item in listed] == ["adminpaul", "userlisa"], listed
    assert all(item.status == "confirmed" for item in listed), listed
    assert all(item.ends_at >= day_start for item in listed), listed

with module.SessionLocal() as db:
    try:
        module.list_admin_bookings(make_request(user_token), db)
        raise AssertionError("non-admin request unexpectedly succeeded")
    except HTTPException as exc:
        assert exc.status_code == 403, exc

print("admin bookings direct test passed")
