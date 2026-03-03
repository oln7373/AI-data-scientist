# mcp_server.py
import os
import base64
from email.mime.text import MIMEText
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

mcp = FastMCP("Allstate Tools", json_response=True)

@mcp.tool(description="Health check / connectivity test tool.")
def ping(message: str = "hello") -> str:
    return f"pong: {message}"

@mcp.tool(description="Add two numbers together and return the result.")
def add_numbers(a: float, b: float) -> float:
    return a + b

# This function is part of the sned_email tool and hence is in the mcp server file
def _gmail_service():
    """
    Local helper: load OAuth token + build Gmail service.
    Assumes token.json / credentials.json live in the working directory
    (or adjust to your paths).
    """
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
        else:
            if not os.path.exists("credentials.json"):
                raise RuntimeError("Missing credentials.json for Gmail OAuth")
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


@mcp.tool(description="Compose a plain-text email body from subject/body fields (no sending).")
def compose_email(recipient_email: str, subject: str, body: str) -> dict:
    """
    Separating compose vs send is useful for the risk Jenny mentioned:
    you can log/review compose output before calling send.
    """
    return {
        "to": recipient_email,
        "subject": subject,
        "body": body,
    }

@mcp.tool(description="Send a plain-text email via Gmail API. Side-effecting tool.")
def send_email(recipient_email: str, subject: str, body: str) -> str:
    try:
        service = _gmail_service()

        msg = MIMEText(body)
        msg["to"] = recipient_email
        msg["subject"] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        sent = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": raw})
            .execute()
        )
        return f"sent: {sent.get('id', '')}".strip()
    except HttpError as e:
        raise RuntimeError(f"Gmail API error: {e}") from e







#  __  __       _         ______               _   _             
# |  \/  | __ _(_)_ __   |  ____|   _ _ __ ___| |_(_) ___  _ __  
# | |\/| |/ _` | | '_ \  | |_ | | | | '__/ __| __| |/ _ \| '_ \ 
# | |  | | (_| | | | | | |  _|| |_| | | | (__| |_| | (_) | | | |
# |_|  |_|\__,_|_|_| |_| |_|   \__,_|_|  \___|\__|_|\___/|_| |_|




if __name__ == "__main__":
    # Serves at http://localhost:8000/mcp by default for streamable-http
    # (exact host/port can be configured via .env if you want)
    mcp.settings.port = int(os.getenv("MCP_PORT", "8005"))
    mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport="streamable-http")