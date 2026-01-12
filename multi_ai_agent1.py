import autogen
import ollama
import re
import os
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import FileResponse

router = APIRouter()

# ✅ Define Pydantic Model for Request Validation
class QueryRequest(BaseModel):
    question: str

# Function to rephrase user question
def rephrasing(question):
    prompt = f"Rephrase the following sentence: '{question}'. If there is a file path or variable names, don't change them. Provide only one precise response, no alternative responses."
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

# Agent Configuration
config_list = {"config_list": [
    {
        "model": "llama3.1:latest",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "temperature": 0.0,
        "price": [0, 0],
    }
]}

# Directory where images are saved
IMAGE_DIR = "paper"

# Dataset schema (Preserving original)
data_schema = '''Don't assume or fabricate dataset. 
invoice_no: Invoice number. Nominal. A combination of the letter 'I' and a 6-digit integer uniquely assigned to each operation.
customer_id: Customer number. Nominal. A combination of the letter 'C' and a 6-digit integer uniquely assigned to each operation.
gender: String variable of the customer's gender.
age: Positive Integer variable of the customers age.
category: String variable of the category of the purchased product.
quantity: The quantities of each product (item) per transaction. Numeric.
price: Unit price. Numeric. Product price per unit in Turkish Liras (TL).
payment_method: String variable of the payment method (cash, credit card or debit card) used for the transaction.
invoice_date: Invoice date. The day when a transaction was generated.
shopping_mall: String variable of the name of the shopping mall where the transaction was made.
Total price should be calculated using quantity and price.
'''

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
    Use the code from FileReader and Summarizer to answer the question being asked. You will provide only code; no text statements. 
    Don't provide unnecessary code. The schema for dataset is {data_schema}. """,
)

filereader = autogen.AssistantAgent(
    name="FileReader",
    llm_config=config_list,
    system_message=f"""Read dataset and check its consistency. The schema for dataset is {data_schema}. 
    You should write Python code to read the file and print the shape of the dataframe. 
    If the shape is non-zero, print 'File reading successful'. Strictly no statements, only code.""",
)

summarizer = autogen.AssistantAgent(
    name="Summarizer",
    llm_config=config_list,
    system_message=f"""You are a dataset summarizer agent. You will use the code already developed by FileReader and work on top of it. 
    If summary is not being asked, don't write any code, simply skip your turn. Given the file content of a dataset, 
    produce a concise summary that includes key properties like number of rows, columns, and any significant statistics. 
    You will only provide code, no statements. The schema for dataset is {data_schema}.""",
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
    agents=[user_proxy, filereader, summarizer, coder, executor],
    messages=[], max_round=10,
    speaker_selection_method="round_robin",
    enable_clear_history=False,
    send_introductions=True
)

manager = autogen.GroupChatManager(
    system_message='''You are a chat manager. Fragment the task and assign it to agents based on their capabilities. 
    Example: File reading → FileReader, Dataframe summarization → Summarizer, Data manipulation → Data Scientist. 
    Don't give all tasks to a single agent. Once an agent provides code, ask the Executor agent to execute it.''',
    is_termination_msg=lambda msg: "exitcode: 0 (execution succeeded)" in msg["content"].lower(),
    groupchat=groupchat,
    llm_config=config_list,
)

# ✅ Function to Extract Relevant Output from Chat History
def extract_relevant_output(chat_history):
    """
    Extracts the last Python code snippet, execution result, and any generated image from the chat history.
    """
    last_python_code = None
    last_code_index = None
    image_filename = None

    # Step 1: Find the last Python code snippet
    for i in reversed(range(len(chat_history))):  # Iterate in reverse to get the last occurrence
        if '```python' in chat_history[i]['content']:
            match = re.search(r"```python\n(.*?)```", chat_history[i]['content'], re.DOTALL)
            if match:
                last_python_code = match.group(1)
                last_code_index = i
                break  # Stop after finding the last executed code

    # Step 2: Find the execution result (immediately after the code block)
    execution_result = None
    if last_code_index is not None and last_code_index + 1 < len(chat_history):
        next_entry = chat_history[last_code_index + 1]  # The next message in the chat history
        if "exitcode:" in next_entry["content"]:  # Check if it's an execution result
            execution_result = next_entry["content"]

    # Step 3: Extract the saved image filename
    if last_python_code:
        match = re.search(r'plt\.savefig\(["\'](.*?)["\']\)', last_python_code)
        if match:
            image_filename = match.group(1)

    # Step 4: Check if the image exists
    image_path = os.path.join(IMAGE_DIR, image_filename) if image_filename else None
    image_exists = os.path.exists(image_path) if image_path else False

    # Step 5: Format the final output
    if image_exists:
        return {"type": "image", "image_url": f"/get_image/{image_filename}"}
    elif last_python_code and execution_result:
        return {"type": "text", "content": f"**Generated Code:**\n```python\n{last_python_code}\n```\n\n**Execution Result:**\n{execution_result}"}
    elif last_python_code:
        return {"type": "text", "content": f"**Generated Code:**\n```python\n{last_python_code}\n```\n\n(No execution result found.)"}
    else:
        return {"type": "text", "content": "No relevant output found."}

# ✅ FastAPI endpoint to process chatbot queries with image support
@router.post("/multi_ai_agent")
def multi_ai_agent_query(request: QueryRequest):
    """
    Endpoint to handle Multi-AI-Agent chat queries.
    """
    chat_result = user_proxy.initiate_chat(
        manager,
        message=rephrasing(request.question),
        summary_method="last_msg",
    )

    # Extract only the relevant part (last Python code, execution result, or image)
    return extract_relevant_output(chat_result.chat_history)

# ✅ FastAPI endpoint to serve generated images
@router.get("/get_image/{filename}")
def get_image(filename: str):
    """
    Endpoint to serve images from the 'paper' directory.
    """
    image_path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    return {"error": "Image not found"}
