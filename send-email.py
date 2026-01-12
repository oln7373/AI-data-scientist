import os.path
import base64
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.                                                                                                                                                              
# The 'gmail.send' scope is required to send emails.                                                                                                                                                                  
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def send_email_via_gmail_api(recipient_email, subject, body):
    """Sends an email using the Gmail API.                                                                                                                                                                            
                                                                                                                                                                                                                      
    Args:                                                                                                                                                                                                             
        recipient_email (str): The email address of the recipient.                                                                                                                                                    
        subject (str): The subject of the email.                                                                                                                                                                      
        body (str): The body content of the email.                                                                                                                                                                    
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is                                                                                                                                         
    # created automatically when the authorization flow completes for the first time.                                                                                                                                 
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.                                                                                                                                             
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # The 'credentials.json' file is the one you downloaded from Google Cloud Console.                                                                                                                        
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run                                                                                                                                                                       
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('gmail', 'v1', credentials=creds)

        # Create the email message                                                                                                                                                                                    
        message = MIMEText(body)
        message['to'] = recipient_email
        message['subject'] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        # Send the message                                                                                                                                                                                            
        send_message = (service.users().messages().send(
            userId="me", body={"raw": raw_message}).execute())
        print(f"Message Id: {send_message['id']}")
        print(f"Email sent successfully to {recipient_email}")

    except HttpError as error:
        print(f"An error occurred: {error}")


# Example usage:                                                                                                                                                                                                      
if __name__ == '__main__':
    #recipient = "monikamittal27@gmail.com"                                                                                                                                                                           
    recipient = "ramankhurana1986@gmail.com"
    email_subject = "Test Email from Gmail API"
    email_body = "Hello Monika,\n\nThis is a test email sent using the Google Gmail API with Python.\n\nBest regards,\nRaman Khurana"

    send_email_via_gmail_api(recipient, email_subject, email_body)


