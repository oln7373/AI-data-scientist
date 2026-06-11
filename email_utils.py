"""Email extraction and formatting utilities for the agent system."""

import re

from fastapi import Request

_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")


def extract_email_and_clean_prompt(text: str) -> tuple[str | None, str]:
    """Extract the first email address from text without modifying the prompt.

    Args:
        text: Raw user prompt that may contain an email address.

    Returns:
        A tuple of (email_address_or_None, original_text_unchanged).
    """
    m = _EMAIL_RE.search(text)
    if not m:
        return None, text
    email = m.group(1).strip().strip(".,;:!?)\"]'")
    return email, text


def make_absolute_image_url(request: Request, rel_url: str) -> str:
    """Build an absolute URL for a generated image served by the backend.

    Args:
        request: The current FastAPI request (used to derive the base URL).
        rel_url: Relative image URL of the form ``/get_image/<filename>``.

    Returns:
        Absolute URL string suitable for inclusion in email bodies.
    """
    return str(request.url_for("get_image", filename=rel_url.rsplit("/", 1)[-1]))


def compose_email_payload(
    orig_question: str,
    result: dict,
    absolute_image_url: str | None,
) -> tuple[str, str]:
    """Build the subject and body for a result-delivery email.

    Args:
        orig_question: The original user question.
        result: The agent result dict (keys: ``type``, ``content``, ``image_url``).
        absolute_image_url: Pre-resolved absolute image URL, or None for text results.

    Returns:
        A tuple of (subject, body) strings ready for SMTP delivery.
    """
    subject = "Your requested result"

    if result.get("type") == "image":
        body = (
            "Hi,\n\nHere is the result you requested.\n\n"
            f"Question: {orig_question}\n"
            f"Result type: image\n"
            f"Image URL: {absolute_image_url}\n\n"
            "Best,\nMulti-AI Agent"
        )
    else:
        content = result.get("content", "")
        body = (
            "Hi,\n\nHere is the result you requested.\n\n"
            f"Question: {orig_question}\n"
            "Result:\n"
            f"{content}\n\n"
            "Best,\nMulti-AI Agent"
        )

    return subject, body
