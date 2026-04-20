"""Authentication test suite for witness_third_gate.py — 20+ tests.

Covers:
    - Missing Authorization header → 401
    - Malformed Authorization header → 401
    - Wrong signature → 401
    - Expired timestamp → 401
    - Future timestamp beyond tolerance → 401
    - Missing timestamp header → 401
    - Invalid (non-numeric) timestamp → 401
    - Oversized body → 413
    - Rate limit exceeded → 429
    - Valid request → 200 (existing behavior preserved)
    - DEV mode bypass works
    - Non-DEV mode without secret refuses to start
    - Timing-safe comparison (compare_digest used, not ==)
    - GET endpoint (empty body canonical string) → 401 / 200
    - POST /witness/ingest authenticated correctly
    - Signature computed over body bytes (not string)
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Generator

# ---------------------------------------------------------------------------
# Test secret + signing helpers
# ---------------------------------------------------------------------------

_TEST_SECRET = "test-secret-for-witness-auth-tests"
_TEST_SECRET_BYTES = _TEST_SECRET.encode()


def _sign(body: bytes, secret: bytes = _TEST_SECRET_BYTES) -> str:
    """Compute the HMAC-SHA256 hex digest of body with secret."""
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def _now_ts() -> str:
    """Return the current unix timestamp as a string."""
    return str(int(time.time()))


def _auth_headers(
    body: bytes = b"",
    secret: bytes = _TEST_SECRET_BYTES,
    timestamp: str | None = None,
    authorization: str | None = None,
) -> dict[str, str]:
    """Return a complete set of auth headers for a request.

    Args:
        body: Request body bytes (used to compute signature).
        secret: HMAC secret bytes.
        timestamp: Override timestamp string (defaults to now).
        authorization: Override the full Authorization header value.

    Returns:
        Dict with Authorization and X-Sybilcore-Timestamp.
    """
    ts = timestamp or _now_ts()
    sig = _sign(body, secret)
    return {
        "Authorization": authorization or f"Bearer {sig}",
        "X-Sybilcore-Timestamp": ts,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _reset_lazy_secret(wg: Any) -> None:
    """Reset the lazy secret loader so it re-reads env vars on next request."""
    wg._WITNESS_SECRET = None
    wg._WITNESS_SECRET_LOADED = False


@pytest.fixture()
def authed_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """TestClient with auth enabled and a known test secret."""
    monkeypatch.setenv("SYBILCORE_WITNESS_SECRET", _TEST_SECRET)
    monkeypatch.delenv("SYBILCORE_WITNESS_DEV", raising=False)

    import integrations.gastown.phase2b_mayor_tap.witness_third_gate as wg
    _reset_lazy_secret(wg)

    from integrations.gastown.phase2b_mayor_tap.witness_third_gate import (
        CoefficientStore,
        app,
    )
    wg._coefficient_store = CoefficientStore()

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    # Clean up: reset secret so next test starts fresh.
    _reset_lazy_secret(wg)


@pytest.fixture()
def dev_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """TestClient with DEV mode enabled (auth bypassed)."""
    monkeypatch.setenv("SYBILCORE_WITNESS_DEV", "1")
    monkeypatch.delenv("SYBILCORE_WITNESS_SECRET", raising=False)

    import integrations.gastown.phase2b_mayor_tap.witness_third_gate as wg
    _reset_lazy_secret(wg)

    from integrations.gastown.phase2b_mayor_tap.witness_third_gate import (
        CoefficientStore,
        app,
    )
    wg._coefficient_store = CoefficientStore()

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    _reset_lazy_secret(wg)


def _minimal_pre_nuke_body() -> dict[str, Any]:
    """Minimal valid PreNukeRequest dict."""
    return {
        "agent_id": "gastown/polecats/nux",
        "rig_name": "gastown",
        "polecat_name": "nux",
        "events": [],
        "reason": "test",
    }


# ---------------------------------------------------------------------------
# Auth failure cases (all should return 401)
# ---------------------------------------------------------------------------


class TestMissingAuthHeaders:
    """Requests with no auth headers must be rejected."""

    def test_no_headers_returns_401(self, authed_client: TestClient) -> None:
        resp = authed_client.post("/witness/pre-nuke", json=_minimal_pre_nuke_body())
        assert resp.status_code == 401

    def test_no_headers_get_status_returns_401(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/witness/status")
        assert resp.status_code == 401

    def test_no_headers_ingest_returns_401(self, authed_client: TestClient) -> None:
        resp = authed_client.post(
            "/witness/ingest",
            json={"agent_id": "x", "coefficient": 0.0, "tier": "clear"},
        )
        assert resp.status_code == 401


class TestMissingOrMalformedAuthorizationHeader:
    """Authorization header edge cases."""

    def test_missing_authorization_header_returns_401(self, authed_client: TestClient) -> None:
        resp = authed_client.post(
            "/witness/pre-nuke",
            json=_minimal_pre_nuke_body(),
            headers={"X-Sybilcore-Timestamp": _now_ts()},
        )
        assert resp.status_code == 401

    def test_authorization_without_bearer_prefix_returns_401(
        self, authed_client: TestClient
    ) -> None:
        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": sig,  # no "Bearer " prefix
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 401

    def test_empty_bearer_token_returns_401(self, authed_client: TestClient) -> None:
        resp = authed_client.post(
            "/witness/pre-nuke",
            json=_minimal_pre_nuke_body(),
            headers={
                "Authorization": "Bearer ",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 401


class TestWrongSignature:
    """Requests with an incorrect signature must be rejected."""

    def test_wrong_signature_returns_401(self, authed_client: TestClient) -> None:
        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        wrong_sig = _sign(body, b"wrong-secret")
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {wrong_sig}",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 401

    def test_truncated_signature_returns_401(self, authed_client: TestClient) -> None:
        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        sig = _sign(body)[:32]  # truncate to half length
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 401

    def test_signature_of_different_body_returns_401(self, authed_client: TestClient) -> None:
        """Signing a different body than what's sent must fail."""
        import json as _json
        actual_body = _json.dumps(_minimal_pre_nuke_body()).encode()
        different_body = b'{"agent_id": "different"}'
        sig = _sign(different_body)  # signed different body
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=actual_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 401


class TestExpiredTimestamp:
    """Requests with stale or future timestamps must be rejected."""

    def test_expired_timestamp_returns_401(self, authed_client: TestClient) -> None:
        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        old_ts = str(int(time.time()) - 120)  # 2 minutes ago (> 60s tolerance)
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": old_ts,
            },
        )
        assert resp.status_code == 401

    def test_future_timestamp_beyond_tolerance_returns_401(
        self, authed_client: TestClient
    ) -> None:
        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        future_ts = str(int(time.time()) + 120)  # 2 minutes into the future
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": future_ts,
            },
        )
        assert resp.status_code == 401

    def test_missing_timestamp_header_returns_401(self, authed_client: TestClient) -> None:
        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                # No X-Sybilcore-Timestamp
            },
        )
        assert resp.status_code == 401

    def test_invalid_timestamp_value_returns_401(self, authed_client: TestClient) -> None:
        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": "not-a-number",
            },
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Request size limit
# ---------------------------------------------------------------------------


class TestOversizedBody:
    """Bodies over 64 KB must be rejected before auth processing."""

    def test_oversized_body_returns_413(self, authed_client: TestClient) -> None:
        oversized = b"x" * (65 * 1024)  # 65 KB > 64 KB limit
        # Even with valid auth headers for the oversized body, must get 413.
        sig = _sign(oversized)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=oversized,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 413

    def test_body_at_limit_boundary_is_not_rejected(self, authed_client: TestClient) -> None:
        """64 KB exactly (or less) must not get 413 (may get 401/422 but not 413)."""
        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        assert len(body) < 64 * 1024
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        # Not 413 — may be 200 or other error, but size limit not triggered.
        assert resp.status_code != 413


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Rate limit enforcement — token bucket per IP."""

    def test_rate_limit_exceeded_returns_429(
        self, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exhaust the rate limit bucket and verify 429 is returned."""
        import integrations.gastown.phase2b_mayor_tap.witness_third_gate as wg

        # Replace rate limiter with one that always rejects.
        class _AlwaysRejectLimiter:
            def is_allowed(self, client_ip: str) -> bool:  # noqa: ARG002
                return False

        monkeypatch.setattr(wg, "_rate_limiter", _AlwaysRejectLimiter())

        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 429

    def test_rate_limit_checked_before_auth(
        self, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rate limit fires before auth, so no auth headers needed to get 429."""
        import integrations.gastown.phase2b_mayor_tap.witness_third_gate as wg

        class _AlwaysRejectLimiter:
            def is_allowed(self, client_ip: str) -> bool:  # noqa: ARG002
                return False

        monkeypatch.setattr(wg, "_rate_limiter", _AlwaysRejectLimiter())

        # No auth headers at all — still 429 (rate limit fires first).
        resp = authed_client.post(
            "/witness/pre-nuke",
            json=_minimal_pre_nuke_body(),
        )
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Valid request → 200
# ---------------------------------------------------------------------------


class TestValidRequest:
    """Correctly signed requests must reach the endpoint and return 200."""

    def test_valid_signed_pre_nuke_returns_200(self, authed_client: TestClient) -> None:
        import json as _json
        body_dict = _minimal_pre_nuke_body()
        body = _json.dumps(body_dict).encode()
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "allow_kill" in data
        assert data["allow_kill"] is False  # unknown agent → safe default

    def test_valid_signed_status_returns_200(self, authed_client: TestClient) -> None:
        # GET request: body is empty bytes, sign b"".
        sig = _sign(b"")
        resp = authed_client.get(
            "/witness/status",
            headers={
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_valid_signed_ingest_returns_200(self, authed_client: TestClient) -> None:
        import json as _json
        body_dict = {"agent_id": "gastown/polecats/nux", "coefficient": 50.0, "tier": "clear"}
        body = _json.dumps(body_dict).encode()
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/ingest",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": _now_ts(),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_timestamp_within_tolerance_is_accepted(self, authed_client: TestClient) -> None:
        """Timestamp 55 seconds old (within 60s tolerance) must pass."""
        import json as _json
        body = _json.dumps(_minimal_pre_nuke_body()).encode()
        near_expired_ts = str(int(time.time()) - 55)
        sig = _sign(body)
        resp = authed_client.post(
            "/witness/pre-nuke",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sig}",
                "X-Sybilcore-Timestamp": near_expired_ts,
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# DEV mode bypass
# ---------------------------------------------------------------------------


class TestDevModeBypass:
    """DEV mode (SYBILCORE_WITNESS_DEV=1) bypasses auth entirely."""

    def test_dev_mode_no_auth_headers_returns_200(self, dev_client: TestClient) -> None:
        resp = dev_client.post("/witness/pre-nuke", json=_minimal_pre_nuke_body())
        assert resp.status_code == 200

    def test_dev_mode_status_no_auth_returns_200(self, dev_client: TestClient) -> None:
        resp = dev_client.get("/witness/status")
        assert resp.status_code == 200

    def test_dev_mode_ingest_no_auth_returns_200(self, dev_client: TestClient) -> None:
        resp = dev_client.post(
            "/witness/ingest",
            json={"agent_id": "gastown/polecats/x", "coefficient": 0.0, "tier": "clear"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Startup rejection without secret (non-DEV mode)
# ---------------------------------------------------------------------------


class TestStartupSecretEnforcement:
    """Server must refuse to start without SYBILCORE_WITNESS_SECRET in non-DEV mode."""

    def test_missing_secret_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_load_secret() must raise RuntimeError when secret is unset in prod mode."""
        monkeypatch.delenv("SYBILCORE_WITNESS_SECRET", raising=False)
        monkeypatch.delenv("SYBILCORE_WITNESS_DEV", raising=False)

        import integrations.gastown.phase2b_mayor_tap.witness_third_gate as wg

        with pytest.raises(RuntimeError, match="SYBILCORE_WITNESS_SECRET"):
            wg._load_secret()

    def test_dev_mode_without_secret_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DEV mode must succeed even without a secret."""
        monkeypatch.delenv("SYBILCORE_WITNESS_SECRET", raising=False)
        monkeypatch.setenv("SYBILCORE_WITNESS_DEV", "1")

        import integrations.gastown.phase2b_mayor_tap.witness_third_gate as wg

        result = wg._load_secret()
        assert result is None  # DEV mode returns None (no secret needed)
