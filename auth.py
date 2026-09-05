"""
auth.py
--------
Minimal JWT auth for the control-room.

This intentionally does NOT invent a fake user/password database —
the original demo doesn't have real operator accounts, and bolting on
a pretend one would be worse than being upfront that this is a toy:
any operator name gets a signed token from /auth/login. What this
*does* demonstrate is the mechanism a production version would need:
the browser can no longer just set a localStorage flag to "log in" —
it holds a signed, expiring token that the backend actually verifies
on every state-changing request. Swapping this for real
username/password (or SSO) is a matter of changing what /auth/login
checks before it signs a token; nothing downstream changes.

Off by default (REQUIRE_AUTH=0) so the demo keeps running zero-config,
same philosophy as the mock LLM in decision_maker.py. Set
REQUIRE_AUTH=1 to actually enforce tokens on /recover.
"""
import os
import time

import jwt  # PyJWT

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
JWT_TTL_SECONDS = 8 * 60 * 60  # one operator shift

REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "0") == "1"


def issue_token(operator: str) -> str:
    now = int(time.time())
    payload = {"sub": operator, "iat": now, "exp": now + JWT_TTL_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def verify_token(token: str) -> dict:
    """Raises jwt.PyJWTError (expired, bad signature, malformed, ...)
    on anything invalid — callers should catch broadly and turn it
    into a 401."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
