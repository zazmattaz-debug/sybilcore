"""SDK exceptions — typed errors for client error handling."""

from __future__ import annotations


class SybilCoreError(Exception):
    """Base class for all SybilCore SDK errors."""


class SybilCoreLocalImportError(SybilCoreError):
    """Raised when local mode is requested but the sybilcore package isn't installed.

    Fix:
        pip install sybilcore-sdk[local]
        # or install the core package directly:
        pip install sybilcore
    """


class SybilCoreAPIError(SybilCoreError):
    """Raised when the remote API returns a non-success response."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class SybilCoreAuthError(SybilCoreAPIError):
    """Raised on 401/403 — missing or invalid API key."""


class SybilCoreRateLimitError(SybilCoreAPIError):
    """Raised on 429 — quota exceeded. Retry after `retry_after` seconds."""

    def __init__(self, message: str, retry_after: float | None = None, **kwargs: object) -> None:
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.retry_after = retry_after
