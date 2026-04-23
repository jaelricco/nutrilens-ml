"""Shared-secret auth for backend -> ML calls.

Only the Rust backend calls this service; the iOS app never does. The header
is set from a secrets-manager value in production. In development, an empty
secret disables the check so the service is ergonomic locally.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from nutrilens_ml.config import load_settings

HEADER_NAME = "X-Internal-Secret"


def require_shared_secret(
    x_internal_secret: str | None = Header(default=None, alias=HEADER_NAME),
) -> None:
    settings = load_settings()
    expected = (
        settings.serve_shared_secret.get_secret_value()
        if settings.serve_shared_secret is not None
        else ""
    )
    if settings.environment != "development" and not expected:
        # Fail closed in non-dev when the secret was forgotten in config.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="server misconfigured: SERVE_SHARED_SECRET unset",
        )
    if not expected:
        return
    if x_internal_secret is None or not hmac.compare_digest(x_internal_secret, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid secret")
