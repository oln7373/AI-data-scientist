"""MCP server exposing tools (ping, add_numbers, compose_email, send_email, select_data)."""

import csv
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import structlog
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from config import configure_logging, get_config

load_dotenv()
configure_logging()

logger = structlog.get_logger(__name__)

mcp = FastMCP("Allstate Tools")


def _smtp_settings() -> dict:
    """Load SMTP configuration from environment variables.

    Required env vars:
        SMTP_HOST: e.g. smtp.gmail.com | smtp.office365.com
        SMTP_USER: sender login / address
        SMTP_PASSWORD: password or app-password

    Optional env vars:
        SMTP_PORT: 587 (default STARTTLS) | 465 (SSL) | 25
        SMTP_FROM: display From address (defaults to SMTP_USER)
        SMTP_TLS: starttls (default) | ssl | none

    Returns:
        Dict with keys host, port, user, password, from_addr, tls_mode.

    Raises:
        RuntimeError: If any required env var is missing.
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
    """Dispatch an email over SMTP, handling STARTTLS, SSL, and plain modes.

    Args:
        cfg: SMTP settings dict returned by _smtp_settings().
        recipient: Destination email address.
        msg_str: Fully-encoded MIME message string.
        from_addr: Sender address for the SMTP envelope.

    Raises:
        RuntimeError: On authentication failure or any SMTP error.
    """
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

        else:
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


@mcp.tool(description="Health check / connectivity test tool.")
def ping(message: str = "hello") -> str:
    """Return a pong response for connectivity testing.

    Args:
        message: Optional message to echo back.

    Returns:
        A pong string containing the echoed message.
    """
    return f"pong: {message}"


@mcp.tool(description="Add two numbers together and return the result.")
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return their sum.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The sum a + b.
    """
    return a + b


@mcp.tool(description="Compose a plain-text email body from subject/body fields (no sending).")
def compose_email(recipient_email: str, subject: str, body: str) -> dict:
    """Build an email payload for review before committing to send.

    Keeping compose and send as separate tools lets the agent (or a human)
    inspect the draft before any side-effects occur.

    Args:
        recipient_email: Destination address.
        subject: Email subject line.
        body: Plain-text body content.

    Returns:
        Dict with keys to, subject, body.
    """
    return {
        "to": recipient_email,
        "subject": subject,
        "body": body,
    }


@mcp.tool(description="Send a plain-text email via SMTP. Works with any email provider.")
def send_email(recipient_email: str, subject: str, body: str) -> str:
    """Send a plain-text email using SMTP credentials from the environment.

    Provider quick-start:
        Gmail        → SMTP_HOST=smtp.gmail.com          SMTP_PORT=587  SMTP_TLS=starttls
        Outlook/O365 → SMTP_HOST=smtp.office365.com      SMTP_PORT=587  SMTP_TLS=starttls
        Yahoo        → SMTP_HOST=smtp.mail.yahoo.com     SMTP_PORT=587  SMTP_TLS=starttls
        Corporate    → SMTP_HOST=mail.company.com        SMTP_PORT=25   SMTP_TLS=none

    Args:
        recipient_email: Destination address.
        subject: Email subject line.
        body: Plain-text body content.

    Returns:
        Confirmation string on success.

    Raises:
        RuntimeError: On SMTP or authentication failure.
    """
    cfg = _smtp_settings()

    msg = MIMEText(body, "plain")
    msg["To"] = recipient_email
    msg["From"] = cfg["from_addr"]
    msg["Subject"] = subject

    _send_via_smtp(cfg, recipient_email, msg.as_string(), cfg["from_addr"])
    logger.info("email_sent", recipient=recipient_email)
    return f"sent to: {recipient_email}"


@mcp.tool(description="Return a sample of customer IDs from the shopping dataset.")
def select_data() -> list[str]:
    """Read the customer dataset and return a sample of customer IDs.

    Returns:
        List of customer_id strings (length controlled by
        data.select_data_sample_size in configs/default.json).

    Raises:
        FileNotFoundError: If the dataset CSV does not exist.
        RuntimeError: If the customer_id column is absent from the CSV.
    """
    cfg = get_config()
    csv_path = Path(__file__).parent / cfg.data.image_dir / cfg.data.dataset_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    customer_ids: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "customer_id" not in (reader.fieldnames or []):
            raise RuntimeError(f"'customer_id' column not found in {csv_path}")
        for row in reader:
            customer_ids.append(row["customer_id"])
            if len(customer_ids) >= cfg.data.select_data_sample_size:
                break

    logger.info("select_data_called", count=len(customer_ids))
    return customer_ids


if __name__ == "__main__":
    mcp.settings.port = int(os.getenv("MCP_PORT", "8005"))
    mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
    logger.info("mcp_server_starting", port=mcp.settings.port, host=mcp.settings.host)
    mcp.run(transport="streamable-http")
