"""
decision/nudge_copy.py
------------------------
Reason-aware customer nudge copy.

Every prior version of decision_maker sent the SAME generic "your
payment didn't go through" line regardless of why it failed. This
generates a short, reason-specific message instead — a customer whose
OTP timed out needs different words than one whose card expired.

Uses the real LLM (same anthropic client pattern as
decision_maker._call_real_llm) when ANTHROPIC_API_KEY is set, and
falls back to a curated static template per reason otherwise, so the
pipeline never blocks on this being "best effort."
"""
import os

STATIC_TEMPLATES = {
    "insufficient_funds": "Hi! Your last payment didn't go through — your account balance may be low. We'll retry shortly, or feel free to top up in the meantime.",
    "otp_failed": "Hi! Your OTP verification didn't complete in time. Please keep your phone handy — we'll retry shortly.",
    "card_expired": "Hi! Your card seems to have expired. Please update your card details to complete this payment.",
    "risk_block": "Hi! We couldn't complete this payment due to a security check. If this was you, please try again or use a different payment method.",
    "bank_timeout": "Hi! Your bank took a bit long to respond. We'll retry automatically — no action needed.",
    "network_error": "Hi! A temporary network issue interrupted your payment. We're retrying now.",
    "issuer_down": "Hi! Your card issuer is experiencing a brief outage. We'll retry once things stabilize.",
}

DEFAULT_TEMPLATE = "Hi! Your last payment didn't go through — could you please try again?"


def _static(reason: str) -> str:
    return STATIC_TEMPLATES.get(reason, DEFAULT_TEMPLATE)


def generate(reason: str, txn: dict) -> str:
    """Returns a short, friendly, reason-aware nudge message. Tries the
    real LLM first (if ANTHROPIC_API_KEY is set); falls back to the
    static template on any failure or absence of a key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _static(reason)

    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            f"Write ONE short, friendly SMS-length nudge message (under 160 characters) "
            f"to a customer whose payment of Rs {txn.get('amount')} failed due to "
            f"'{reason}'. Never pushy, no links, no emojis. Reply with ONLY the message text."
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text if text else _static(reason)
    except Exception:
        return _static(reason)