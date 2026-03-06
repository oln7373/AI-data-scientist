import re
from typing import Optional, Tuple
from fastapi import Request

EMAIL_RE = re.compile(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})')


def extract_email_and_clean_prompt(text: str) -> Tuple[Optional[str], str]:
    """
    Extract the first email address from the text but DO NOT modify
    the original prompt. The backend can still use the email separately.
    """
    m = EMAIL_RE.search(text)
    if not m:
        return None, text

    email = m.group(1).strip().strip(".,;:!?)\"]'")
    
    # Return the original text unchanged
    return email, text


def make_absolute_image_url(request: Request, rel_url: str) -> str:
    return str(request.url_for("get_image", filename=rel_url.rsplit("/", 1)[-1]))


def compose_email_payload(orig_question: str, result: dict, absolute_image_url: Optional[str]):
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