"""
send-email-attach.py — send email with optional attachment via SMTP.
 
No OAuth. No Google APIs. Works with any SMTP provider.
See send-email.py for the full provider quick-start guide and .env variable list.
 
Extra env var:
    ATTACHMENT_PATH   local path to a file to attach (optional)
"""
 
import os
import smtplib
import ssl
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from dotenv import load_dotenv
 
load_dotenv()
 
 
def send_email_via_smtp(
    recipient_email: str,
    subject: str,
    body: str,
    attachment_path: str | None = None,
) -> None:
    """
    Send a plain-text email with an optional file attachment via SMTP.
 
    Args:
        recipient_email:  Destination address.
        subject:          Email subject line.
        body:             Plain-text body.
        attachment_path:  Local filesystem path to attach (optional).
    """
 
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    from_addr = os.getenv("SMTP_FROM", user)
    tls_mode = os.getenv("SMTP_TLS", "starttls").lower()
 
    if not host or not user or not password:
        raise ValueError(
            "Missing SMTP config. Set SMTP_HOST, SMTP_USER, and SMTP_PASSWORD "
            "in your environment or .env file."
        )
 
    # --- Build message ---
    msg = MIMEMultipart()
    msg["To"] = recipient_email
    msg["From"] = from_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
 
    # --- Attach file if provided ---
    if attachment_path:
        if not os.path.exists(attachment_path):
            raise FileNotFoundError(f"Attachment not found: {attachment_path}")
 
        filename = os.path.basename(attachment_path)
        with open(attachment_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
 
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)
 
    # --- Send ---
    context = ssl.create_default_context()
    raw = msg.as_string()
 
    try:
        if tls_mode == "ssl":
            with smtplib.SMTP_SSL(host, port, context=context) as server:
                server.login(user, password)
                server.sendmail(from_addr, recipient_email, raw)
 
        elif tls_mode == "starttls":
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(user, password)
                server.sendmail(from_addr, recipient_email, raw)
 
        else:
            with smtplib.SMTP(host, port) as server:
                if user and password:
                    server.login(user, password)
                server.sendmail(from_addr, recipient_email, raw)
 
        attachment_note = f" with attachment '{os.path.basename(attachment_path)}'" if attachment_path else ""
        print(f"✅ Email sent successfully to {recipient_email}{attachment_note}")
 
    except smtplib.SMTPAuthenticationError as e:
        raise RuntimeError(
            f"SMTP authentication failed for {user}. "
            "Check credentials. Gmail users: use an App Password."
        ) from e
    except smtplib.SMTPException as e:
        raise RuntimeError(f"SMTP error: {e}") from e
 
 
if __name__ == "__main__":
    recipient = os.getenv("EMAIL_RECIPIENT")
    if not recipient:
        raise ValueError("EMAIL_RECIPIENT is not set in the environment.")
 
    send_email_via_smtp(
        recipient_email=recipient,
        subject=os.getenv("EMAIL_SUBJECT", "Test Email via SMTP"),
        body=os.getenv(
            "EMAIL_BODY",
            "Hello,\n\nThis is a test email sent using Python's smtplib.\n\nBest regards,",
        ),
        attachment_path=os.getenv("ATTACHMENT_PATH"),
    )
)
