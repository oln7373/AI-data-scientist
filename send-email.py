"""
send-email.py — standalone script to send a plain-text email via SMTP.
 
No OAuth. No Google APIs. Works with any SMTP provider.
 
Configure via environment variables (or a .env file):
    SMTP_HOST        e.g. smtp.gmail.com | smtp.office365.com | mail.company.com
    SMTP_PORT        587 (STARTTLS, default) | 465 (SSL) | 25
    SMTP_USER        your login username / sender address
    SMTP_PASSWORD    your password or app-password
    SMTP_FROM        optional display "From" address (defaults to SMTP_USER)
    SMTP_TLS         starttls (default) | ssl | none
 
    EMAIL_RECIPIENT  destination address  [required]
    EMAIL_SUBJECT    subject line         [optional]
    EMAIL_BODY       body text            [optional]
 
Quick-start for common providers
─────────────────────────────────────────────────────────────────────────────
Gmail:
    SMTP_HOST=smtp.gmail.com  SMTP_PORT=587  SMTP_TLS=starttls
    → Create an App Password at myaccount.google.com/apppasswords
 
Outlook / Office 365:
    SMTP_HOST=smtp.office365.com  SMTP_PORT=587  SMTP_TLS=starttls
 
Yahoo Mail:
    SMTP_HOST=smtp.mail.yahoo.com  SMTP_PORT=587  SMTP_TLS=starttls
    → Create an App Password in Yahoo Account Security settings
 
Corporate relay (no auth):
    SMTP_HOST=mail.company.com  SMTP_PORT=25  SMTP_TLS=none
─────────────────────────────────────────────────────────────────────────────
"""
 
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from dotenv import load_dotenv
 
load_dotenv()
 
 
def send_email_via_smtp(recipient_email: str, subject: str, body: str) -> None:
    """Send a plain-text email using SMTP credentials from the environment."""
 
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
 
    msg = MIMEText(body, "plain")
    msg["To"] = recipient_email
    msg["From"] = from_addr
    msg["Subject"] = subject
 
    context = ssl.create_default_context()
 
    try:
        if tls_mode == "ssl":
            with smtplib.SMTP_SSL(host, port, context=context) as server:
                server.login(user, password)
                server.sendmail(from_addr, recipient_email, msg.as_string())
 
        elif tls_mode == "starttls":
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(user, password)
                server.sendmail(from_addr, recipient_email, msg.as_string())
 
        else:  # no TLS — internal relay
            with smtplib.SMTP(host, port) as server:
                if user and password:
                    server.login(user, password)
                server.sendmail(from_addr, recipient_email, msg.as_string())
 
        print(f"Email sent successfully to {recipient_email}")
 
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
 
    subject = os.getenv("EMAIL_SUBJECT", "Test Email via SMTP")
    body = os.getenv(
        "EMAIL_BODY",
        "Hello,\n\nThis is a test email sent using Python's smtplib.\n\nBest regards,",
    )
 
    send_email_via_smtp(recipient, subject, body)
 