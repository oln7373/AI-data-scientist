import os.path
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def send_email_via_gmail_api(recipient_email, subject, body, attachment_path=None):
    """Send an email with optional attachment using Gmail API."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('gmail', 'v1', credentials=creds)

        # Create the email container
        message = MIMEMultipart()
        message['to'] = recipient_email
        message['subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        # Attach file if provided
        if attachment_path:
            filename = os.path.basename(attachment_path)
            with open(attachment_path, 'rb') as f:
                mime_part = MIMEBase('application', 'octet-stream')
                mime_part.set_payload(f.read())
            encoders.encode_base64(mime_part)
            mime_part.add_header(
                'Content-Disposition',
                f'attachment; filename={filename}',
            )
            message.attach(mime_part)

        # Encode and send
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_message = service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
        print(f"✅ Email sent successfully to {recipient_email}")
        print(f"Message ID: {send_message['id']}")

    except HttpError as error:
        print(f"❌ An error occurred: {error}")


send_email_via_gmail_api(
    recipient_email="ramankhurana1986@gmail.com",
    subject="Payment Method Distribution Report",
    body="Please find attached the payment method distribution chart.",
    attachment_path="paper/payment_method_distribution.png"
)
