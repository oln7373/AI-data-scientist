"""
gmail_send.py  →  smtp_send.py (drop-in replacement)
 
Sends email via plain SMTP — no OAuth, no Google APIs.
Works with Gmail, Outlook, Yahoo, corporate Exchange, or any SMTP relay.
 
Required environment variables:
    SMTP_HOST      e.g. smtp.gmail.com | smtp.office365.com | mail.company.com
    SMTP_PORT      e.g. 587 (STARTTLS, recommended) | 465 (SSL) | 25
    SMTP_USER      your login / sender address
    SMTP_PASSWORD  your password or app-password
 
Optional:
    SMTP_FROM      display "From" address (defaults to SMTP_USER)
    SMTP_TLS       "starttls" (default) | "ssl" | "none"
"""
 
import os
import smtplib
import ssl
from email.mime.text import MIMEText
 
 
def _smtp_settings() -> dict:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    from_addr = os.getenv("SMTP_FROM", user)
    tls_mode = os.getenv("SMTP_TLS", "starttls").lower()
 
    if not host or not user or not password:
        raise RuntimeError(
            "Missing SMTP config. Set SMTP_HOST, SMTP_USER, and SMTP_PASSWORD "
            "in your environment or .env file."
        )
 
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "from_addr": from_addr,
        "tls_mode": tls_mode,
    }
 
 
def send_email(recipient_email: str, subject: str, body: str) -> None:
    """
    Send a plain-text email via SMTP.
 
    Args:
        recipient_email: Destination address.
        subject:         Email subject line.
        body:            Plain-text body content.
    """
    cfg = _smtp_settings()
 
    msg = MIMEText(body, "plain")
    msg["To"] = recipient_email
    msg["From"] = cfg["from_addr"]
    msg["Subject"] = subject
 
    context = ssl.create_default_context()
 
    try:
        if cfg["tls_mode"] == "ssl":
            # Port 465 — connect inside an SSL tunnel from the start
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context) as server:
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["from_addr"], recipient_email, msg.as_string())
 
        elif cfg["tls_mode"] == "starttls":
            # Port 587 — connect plain, then upgrade to TLS
            with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["from_addr"], recipient_email, msg.as_string())
 
        else:
            # Port 25 / internal relay — no encryption (use only on trusted networks)
            with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
                if cfg["user"] and cfg["password"]:
                    server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["from_addr"], recipient_email, msg.as_string())
 
    except smtplib.SMTPAuthenticationError as e:
        raise RuntimeError(
            f"SMTP authentication failed for {cfg['user']}. "
            "Check SMTP_USER / SMTP_PASSWORD. "
            "Gmail users: enable 2FA and use an App Password."
        ) from e
    except smtplib.SMTPException as e:
        raise RuntimeError(f"SMTP error: {e}") from e

