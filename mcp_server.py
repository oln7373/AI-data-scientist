# mcp_server.py
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("Allstate Tools")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smtp_settings() -> dict:
    """
    Load SMTP config from environment variables.

    Required:
        SMTP_HOST      e.g. smtp.gmail.com | smtp.office365.com | mail.company.com
        SMTP_USER      sender login / address
        SMTP_PASSWORD  password or app-password

    Optional:
        SMTP_PORT      587 (default, STARTTLS) | 465 (SSL) | 25
        SMTP_FROM      display "From" address  (defaults to SMTP_USER)
        SMTP_TLS       starttls (default) | ssl | none
    """
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


def _send_via_smtp(cfg: dict, recipient: str, msg_str: str, from_addr: str) -> None:
    """Low-level SMTP dispatch — handles STARTTLS / SSL / plain connections."""
    context = ssl.create_default_context()

    try:
        if cfg["tls_mode"] == "ssl":
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context) as server:
                server.login(cfg["user"], cfg["password"])
                server.sendmail(from_addr, recipient, msg_str)

        elif cfg["tls_mode"] == "starttls":
            with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(cfg["user"], cfg["password"])
                server.sendmail(from_addr, recipient, msg_str)

        else:  # no TLS — internal relay
            with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
                if cfg["user"] and cfg["password"]:
                    server.login(cfg["user"], cfg["password"])
                server.sendmail(from_addr, recipient, msg_str)

    except smtplib.SMTPAuthenticationError as e:
        raise RuntimeError(
            f"SMTP authentication failed for {cfg['user']}. "
            "Check credentials. Gmail users: use an App Password."
        ) from e
    except smtplib.SMTPException as e:
        raise RuntimeError(f"SMTP error: {e}") from e


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool(description="Health check / connectivity test tool.")
def ping(message: str = "hello") -> str:
    return f"pong: {message}"


@mcp.tool(description="Add two numbers together and return the result.")
def add_numbers(a: float, b: float) -> float:
    return a + b


@mcp.tool(description="Compose a plain-text email body from subject/body fields (no sending).")
def compose_email(recipient_email: str, subject: str, body: str) -> dict:
    """
    Returns the email payload for review before committing to send.
    Keeping compose and send as separate tools lets the agent (or a human)
    inspect the draft before any side-effects occur.
    """
    return {
        "to": recipient_email,
        "subject": subject,
        "body": body,
    }


@mcp.tool(description="Send a plain-text email via SMTP. Works with any email provider.")
def send_email(recipient_email: str, subject: str, body: str) -> str:
    """
    Send a plain-text email using SMTP credentials configured in the environment.

    Provider quick-start:
      Gmail        → SMTP_HOST=smtp.gmail.com          SMTP_PORT=587  SMTP_TLS=starttls
      Outlook/O365 → SMTP_HOST=smtp.office365.com      SMTP_PORT=587  SMTP_TLS=starttls
      Yahoo        → SMTP_HOST=smtp.mail.yahoo.com     SMTP_PORT=587  SMTP_TLS=starttls
      Corporate    → SMTP_HOST=mail.company.com        SMTP_PORT=25   SMTP_TLS=none

    Returns a confirmation string on success; raises RuntimeError on failure.
    """
    cfg = _smtp_settings()

    msg = MIMEText(body, "plain")
    msg["To"] = recipient_email
    msg["From"] = cfg["from_addr"]
    msg["Subject"] = subject

    _send_via_smtp(cfg, recipient_email, msg.as_string(), cfg["from_addr"])
    return f"sent to: {recipient_email}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

#  __  __       _         ______               _   _             
# |  \/  | __ _(_)_ __   |  ____|   _ _ __ ___| |_(_) ___  _ __  
# | |\/| |/ _` | | '_ \  | |_ | | | | '__/ __| __| |/ _ \| '_ \ 
# | |  | | (_| | | | | | |  _|| |_| | | | (__| |_| | (_) | | | |
# |_|  |_|\__,_|_|_| |_| |_|   \__,_|_|  \___|\__|_|\___/|_| |_|

if __name__ == "__main__":
    mcp.settings.port = int(os.getenv("MCP_PORT", "8005"))
    mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport="streamable-http")