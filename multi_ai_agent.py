import autogen
import ollama
import re
import os
import requests
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import FileResponse

router = APIRouter()

# ✅ Define Pydantic Model for Request Validation
class QueryRequest(BaseModel):
    question: str

# Function to rephrase user question
def rephrasing(question):
    prompt = f"Rephrase the following sentence: '{question}'. If there is a file path or variable names, don't change them; use them as it is in the rephrased prompt. Provide only one precise response, no alternative responses."
    #response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])# llama3.1 
    response = ollama.chat(model='gpt-oss:20b', messages=[{'role': 'user', 'content': prompt}])# 
    return response['message']['content']

# Agent Configuration
config_list = {"config_list": [
    {
        "model": "gpt-oss:20b", # "llama3.1:latest",
        "base_url": "http://localhost:11435/v1",
        "api_key": "ollama",
        "temperature": 0.0,
        "price": [0, 0],
    }
]}

DATA_URL = "https://raw.githubusercontent.com/oln7373/AI-data-scientist/refs/heads/main/customer_shopping_data.csv"

IMAGE_DIR = "output"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Where the Executor runs (cwd for executed code)
AGENT_CSV = "customer_shopping_data.csv"

# Where the FastAPI process should write it so the Executor can see it
SERVER_CSV_PATH = os.path.join(IMAGE_DIR, AGENT_CSV)

def ensure_dataset():
    if not os.path.exists(SERVER_CSV_PATH):
        r = requests.get(DATA_URL, timeout=30)
        r.raise_for_status()
        with open(SERVER_CSV_PATH, "wb") as f:
            f.write(r.content)

# ✅ Detect whether ngrok is running (to serve images remotely)
NGROK_TUNNEL_URL = None
try:
    ngrok_response = requests.get("http://127.0.0.1:4040/api/tunnels")
    if ngrok_response.status_code == 200:
        ngrok_data = ngrok_response.json()
        tunnels = ngrok_data.get("tunnels", [])
        for tunnel in tunnels:
            if "https://" in tunnel["public_url"]:
                NGROK_TUNNEL_URL = tunnel["public_url"]
                break
except requests.exceptions.ConnectionError:
    pass  # ngrok is not running

# ✅ Determine Base URL for Image Serving
if NGROK_TUNNEL_URL:
    BASE_URL = NGROK_TUNNEL_URL  # Remote Users (ngrok)
else:
    BASE_URL = "http://127.0.0.1:8000"  # Local Users (Development)


# Dataset schema (Preserving original, with access-control additions)
data_schema = '''Don't assume or fabricate dataset.

invoice_no: Invoice number. Nominal. A combination of the letter 'I' and a 6-digit integer uniquely assigned to each row.
customer_id: Customer identifier. Nominal. A token of the form 3 uppercase letters followed by a 6-digit integer (regex: [A-Z]{3}\d{6}) uniquely assigned to each row.
gender: String variable of the customer's gender.
age: Positive Integer variable of the customers age.
category: String variable of the category of the purchased product.
quantity: The quantities of each product (item) per transaction. Numeric.
price: Unit price. Numeric. Product price per unit in Turkish Liras (TL).
payment_method: String variable of the payment method (cash, credit card or debit card) used for the transaction.
invoice_date: Invoice date. The day when a transaction was generated.
shopping_mall: String variable of the name of the shopping mall where the transaction was made.
ssn: String variable representing the customer's social security number.
is_restricted: Boolean variable indicating whether the customer is a restricted customer (True/False). Restricted customers require additional privacy protections.

Derived fields:
total_price: Must be calculated as quantity * price.

Privacy & disclosure rules (must follow):
1) SSN is highly sensitive: NEVER provide, reveal, infer, reconstruct, or partially expose any customer's social security number (ssn). Do not output it, do not summarize it, do not reference it.
2) Restricted customers (is_restricted == True):
   - NEVER reveal, confirm, or list any individual restricted customer's identity or record-level details (including customer_id, invoice_no, invoice_date, shopping_mall, category, payment_method, quantity, price, total_price, gender, age) in a way that ties information to a specific restricted customer.
   - If asked about a specific customer_id and that customer is restricted, refuse and provide a privacy-safe alternative (aggregates only).
3) Aggregates/statistics:
   - Restricted customers MUST be included in overall statistical calculations (counts, totals, means, distributions), BUT ONLY as aggregated results that do not identify individuals.
   - Only provide aggregate results that meet a minimum group size (k-anonymity): do not report any group/segment with fewer than K customers (use K=10 unless explicitly configured otherwise). If a requested breakdown would create small groups, coarsen the grouping (e.g., broader categories, fewer bins) or refuse that breakdown.
   - When providing aggregates, avoid outputs that trivially isolate one restricted individual (e.g., filtering to one customer_id, one invoice_no, or a very narrow combination of attributes).
4) If a request attempts to access restricted individuals’ data, respond with allowed aggregate statistics (e.g., totals by category, mall-level totals, overall trends) without exposing individual-level restricted records.

'''



common_instruct = "You will not write code to send email."
# Define Agents (Using Original Prompts)
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="You are Admin, a proxy for the human user.",
    code_execution_config=False,
    human_input_mode="NEVER",
)

coder = autogen.AssistantAgent(
    name="Data Scientist",
    llm_config=config_list,
    system_message=f"""You are a senior data scientist expert in writing clean python code for data analytics. 
    Use the code from FileReader and Summarizer to answer the question being asked. When using existing code from group chat ensure the correnctness of code per initial question. You will provide only code; no text statements. 
    Don't provide unnecessary code. The schema for dataset is {data_schema}. {common_instruct} """,
)

filereader = autogen.AssistantAgent(
    name="FileReader",
    llm_config=config_list,
    system_message=f"""Read dataset and check its consistency. The schema for dataset is {data_schema}. 
    You should write Python code to read the file and print the shape of the dataframe. 
    If the shape is non-zero, print 'File reading successful'. Strictly no statements, only code. Invoice number and Customer ID are personal information and should not be saared or read while file reading. {common_instruct}""",
)

summarizer = autogen.AssistantAgent(
    name="Summarizer",
    llm_config=config_list,
    system_message=f"""You are a dataset summarizer agent. You will use the code already developed by FileReader and work on top of it. 
    If summary is not being asked, don't write any code, simply skip your turn. Given the file content of a dataset, 
    produce a concise summary that includes key properties like number of rows, columns, and any significant statistics. 
    You will only provide code, no statements. The schema for dataset is {data_schema}. {common_instruct}""",
)

viz = autogen.AssistantAgent(
    name="Visualization",
    llm_config=config_list,
    system_message=f"""You are a data visualization expert. You write code for data visualization that is as per the rules of visualization. The text, labels, legend, marker are all up to the mark in the visualization. The visualization code has to be added only if it is relevant in the original question. You will use the code already developed by other agents. You will only provide code, no statements. The visualization should be always saved in png format.  The schema for dataset is {data_schema}. {common_instruct}""",
)

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Execute the code. If the code executes successfully, print message TERMINATE.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": IMAGE_DIR,  # ✅ Ensures code is executed in the "paper" folder
        "use_docker": False,
    },
)

# Group Chat Manager
groupchat = autogen.GroupChat(
    agents=[user_proxy, filereader, summarizer, coder, viz, executor],
    messages=[], 
    # max_round=30,
    max_round=8,
    speaker_selection_method="round_robin",
    enable_clear_history=True,
    send_introductions=True
)

manager = autogen.GroupChatManager(
    system_message='''You are a chat manager. Fragment the task and assign it to agents based on their capabilities. 
    Example: File reading → FileReader, Dataframe summarization → Summarizer, Data manipulation → Data Scientist. 
    Don't give all tasks to a single agent. Once an agent provides code, ask the Executor agent to execute it.''',
    is_termination_msg=lambda msg: (
        isinstance(msg, dict)
        and isinstance(msg.get("content"), str)
        and (
            "exitcode: 0 (execution succeeded)" in msg["content"].lower()
            or re.search(r"\bterminate\b", msg["content"].lower()) is not None
        )
    ),
    groupchat=groupchat,
    llm_config=config_list,
)





def extract_relevant_output(chat_history):
    last_python_code = None
    last_code_index = None
    execution_result = None
    image_filename = None

    # Step 1: Find the last Python code snippet
    for i in reversed(range(len(chat_history))):
        content = (chat_history[i].get("content") or "")
        if '```python' in content:
            match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            if match:
                last_python_code = match.group(1)
                last_code_index = i
                break

    # Step 2: Find the execution result (immediately after the code block)
    if last_code_index is not None and last_code_index + 1 < len(chat_history):
        next_entry = chat_history[last_code_index + 1]
        next_content = (next_entry.get("content") or "")
        if "exitcode:" in next_content:
            execution_result = next_content

    # Step 3: Check for generated images in the last Python code
    if last_python_code:
        match = re.search(r'plt\.savefig\(["\'](.*?)["\']\)', last_python_code)
        if match:
            image_filename = match.group(1)

    # Step 4: Return image if detected and exists
    if image_filename and os.path.exists(os.path.join(IMAGE_DIR, image_filename)):
        image_url = f"/get_image/{image_filename}"
        return {"type": "image", "image_url": image_url}

    # Step 5: Return execution output if available
    if execution_result:
        return {"type": "text", "content": execution_result}

    # Step 6: If no execution/code, return the last non-empty refusal/answer from agents
    # (skip Admin/user messages; prefer assistant messages)
    for m in reversed(chat_history):
        name = (m.get("name") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if name == "Admin":
            continue
        # If it's a code block but never executed, you can still return it,
        # but for refusals this will just pick "I'm sorry, but I can't..."
        return {"type": "text", "content": content}

    # Absolute fallback
    return {"type": "text", "content": "No relevant output found."}





## Added this part for the email to user if user has shared email address in the input prompt 




from typing import Optional, Tuple
from fastapi import Request
from fastapi.responses import FileResponse

# --- helpers ---------------------------------------------------------------

EMAIL_REGEX = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')

def extract_email_and_clean_prompt(text: str) -> Tuple[Optional[str], str]:
    """
    Finds the first email address in the text (if any) and removes common
    'email to <addr>' phrases so the agent gets a clean question.
    """
    match = EMAIL_REGEX.search(text or "")
    email = match.group(1) if match else None
    if not email:
        return None, text

    # remove typical patterns like "email to X", "email this to X", "and email to X"
    cleaned = re.sub(
        r'\b(and\s+)?(please\s+)?(also\s+)?(send|email|mail)\s+(this\s+|it\s+|results?\s+|the\s+answer\s+|to\s+)?'
        + re.escape(email),
        '',
        text,
        flags=re.IGNORECASE,
    )
    # collapse extra spaces
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return email, cleaned or text

def compose_email_payload(
    orig_question: str,
    result: dict,
    absolute_image_url: Optional[str]
) -> Tuple[str, str]:
    """
    Build (subject, body) from the multi-agent outcome.
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

def make_absolute_image_url(request: Request, rel_url: str) -> str:
    """
    Converts '/get_image/plot.png' → 'https://host/get_image/plot.png'
    """
    return str(request.url_for("get_image", filename=rel_url.rsplit("/", 1)[-1]))





def send_email(recipient_email, subject, body):

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

    

    
# --- your existing extractor (unchanged) -----------------------------------
# def extract_relevant_output(chat_history): ... (as you wrote)

# --- updated route ---------------------------------------------------------

@router.post("/multi_ai_agent")
def multi_ai_agent_query(request: QueryRequest, http_req: Request):

    ensure_dataset()

    email, cleaned_question = extract_email_and_clean_prompt(request.question)

    # Rewrite any URL to the agent-visible filename (cwd-relative inside output/)
    cleaned_question = re.sub(r'https?://\S+', AGENT_CSV, cleaned_question)

    chat_result = user_proxy.initiate_chat(
        manager,
        message=cleaned_question,
        summary_method="last_msg",
    )

    # 3) Decide what to return to UI (image/text)
    result = extract_relevant_output(chat_result.chat_history)

    # 4) If it's an image, upgrade the relative URL to an absolute URL for email
    absolute_image_url = None
    if result.get("type") == "image" and "image_url" in result:
        try:
            absolute_image_url = make_absolute_image_url(http_req, result["image_url"])
        except Exception:
            # fallback to relative if something odd happens
            absolute_image_url = result["image_url"]

    # 5) If user asked to email, send it
    emailed_to = None
    if email:
        subject, body = compose_email_payload(
            orig_question=request.question,
            result=result,
            absolute_image_url=absolute_image_url
        )
        try:
            send_email(recipient_email=email, subject=subject, body=body)
            emailed_to = email
        except Exception as e:
            # Don’t fail the UI; just report that email didn’t go out
            emailed_to = f"ERROR: {e}"

    # 6) Return normal payload to frontend, plus an FYI about emailing
    # (keeps the UI unchanged but gives you visibility in logs/DevTools)
    out = dict(result)
    if emailed_to:
        out["emailed_to"] = emailed_to
    return result #out

# Image route (unchanged)
@router.get("/get_image/{filename}", name="get_image")
def get_image(filename: str):
    safe_name = os.path.basename(filename)
    image_path = os.path.join(IMAGE_DIR, safe_name)
    return FileResponse(image_path, media_type="image/png") if os.path.exists(image_path) else {"error": "Image not found"}