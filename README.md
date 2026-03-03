# Agentic AI Data Scientist + MCP Integration

## Overview

This repository implements an **Agentic AI Data Scientist** system extended with **MCP (Model Context Protocol)** tool integration.

The system consists of:

- A **FastAPI backend**
- A **Streamlit frontend**
- A **local LLM** served by **Ollama**
- A **multi-agent AutoGen** orchestration layer
- A separate **MCP server** for tool execution

The backend enforces strict tool governance via an allowlist, ensuring that all external side effects (e.g., sending emails) are executed through MCP rather than directly by the LLM.

The code and dependencies presuppose access to an **NVIDIA GPU** (e.g., Quadro RTX 8000, A100). While limited functionality may run on CPU, GPU usage is strongly recommended for practical performance.

---

## System Architecture

There are two primary execution pathways:

### 1. Multi-Agent Data Scientist

User → `/multi_ai_agent` → AutoGen multi-agent orchestration →  
Code generation → Executor → Result (text/image) → Optional email via MCP

This pathway performs structured data analysis over `customer_shopping_data.csv`.

### 2. Direct LLM Tool Calling

User → `/mcp_agent_add` → Ollama (LLM) → `tool_calls` →  
Backend allowlist enforcement → MCP execution → Return result

This pathway demonstrates OpenAI-style function calling.

---

## Dependencies

### Python

Python 3.10+. All required packages are listed in `myenv.txt`.

### Ollama (Required)

This system uses **Ollama** to serve large language models locally.  
Ollama must be installed and running **before** starting the backend.

### MCP Server (Required)

A separate MCP server must be running to execute external tools securely.

---

## Installing and Running Ollama

### 1. Install Ollama

On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Verify Ollama Installation

```bash
ollama --version
```

### 3. Start the Ollama Server

```bash
ollama serve
```

### 4. Pull Required Models

```bash
ollama pull gpt-oss:20b # or model of your choice
```

## Running the MCP Server

### 5. Start the MCP Server

```bash
python mcp_server.py
```

By default, the MCP server runs on port 8005. Ensure it is running before starting the FastAPI backend.

## Installing and Running the System

### 6. Clone the Repository

```bash
git clone https://github.com/oln7373/AI-data-scientist
cd AI-data-scientist
```

### 7. Install Requirements

```bash
pip install -r myenv.txt
```

### 8. Configure Environment Variables

```bash
cp env.sample .env
```

Edit ```.env``` and provide values for

```
OLLAMA_PORT=11435
OLLAMA_MODEL=gpt-oss:20b
MCP_PORT=8005
ALLOWED_TOOLS=add_numbers,ping,send_email,compose_email
```

The ```ALLOWED_TOOLS``` variable controls which MCP tools may be executed by the backend.

### 9. Launch the FastAPI Backend

```bash
uvicorn backend:app --host 127.0.0.1 --port 8001 --reload
```

(The host and port may be changed if needed.)

### 10. Launch the Streamlit Frontend

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

### 11. (OPTIONAL) Port Forwarding

If running on a remote machine and accessing locally:

```bash
ssh -L 8501:127.0.0.1:8501 \
-L 8001:127.0.0.1:8001 \
-L 8005:127.0.0.1:8005 \
-i ~/.ssh/<YOUR_KEY> \
<YOUR_USER>@<YOUR_MACHINE>.edu
```

### 12. Access the Application

Open your browser at:
```bash
http://localhost:8501/
```

## Backend Endpoints

### `/multi_ai_agent`

Primary multi-agent data science interface.

**Capabilities:**

- Data loading
- Statistical analysis
- Visualization (PNG output)
- Optional email delivery via MCP

If an email address is included in the prompt, the backend:

1. Extracts the email  
2. Generates subject and body content  
3. Calls the MCP `send_email` tool  
4. Returns `"emailed_to"` in the response  

---

### `/mcp_add_demo`

Simple MCP arithmetic test endpoint.

**Example:**

```bash
curl -X POST http://127.0.0.1:8001/mcp_add_demo \
  -H "Content-Type: application/json" \
  -d '{"question":"2 + 3"}'
  ```

### `/mcp_agent_add`

LLM-driven tool-calling demonstration endpoint.

**Execution Flow:**

1. The user prompt is sent to Ollama.  
2. The LLM emits `tool_calls` using OpenAI-style function-calling syntax.  
3. The backend validates the requested tool against the configured allowlist.  
4. If approved, the backend invokes the tool via the MCP server.  
5. The structured tool result is returned to the client.

---

### `/mcp_tools`

Returns the list of tools currently exposed by the MCP server.

---

## Dataset Handling

The system automatically downloads the dataset to:

`output/customer_shopping_data.csv`

The backend ensures the dataset exists before execution begins.

---

## Privacy and Governance

The dataset includes:

- `ssn` (highly sensitive)
- `is_restricted` (restricted customers)

The system enforces the following constraints:

- No SSN disclosure  
- No individual restricted record disclosure  
- k-anonymity (K = 10 minimum group size)  
- Aggregated statistics only for restricted customers  

All privacy and disclosure constraints are encoded directly into the agent system prompts.

---

## Design Principles

### Separation of Concerns

- **LLM** — reasoning and planning  
- **AutoGen** — orchestration and task coordination  
- **Executor** — deterministic Python execution  
- **Backend** — governance and allowlist enforcement  
- **MCP** — side-effect execution  

Each component has a clearly defined responsibility.

---

### Backend Authority

The LLM cannot:

- Override the tool allowlist  
- Access arbitrary system tools  
- Execute side effects directly  

All external actions must pass through backend policy enforcement.

---

### Side-Effect Isolation

Email and other external operations are executed exclusively through MCP.  
No direct side effects are performed by agents or the LLM.

---

## Repository Structure

### `output/`

Contains images and generated artifacts.  
Required for backend execution.

### `data/`

Contains datasets and documents provided as input.

### `figures/`

Static images used by the Streamlit frontend.

### `reports/`

Legacy or auto-generated artifacts from previous versions.

### `mcp_server.py`

Defines MCP tools and the execution layer.

### `multi_ai_agent.py`

Defines the AutoGen multi-agent orchestration system.

### `backend.py`

FastAPI entry point and policy enforcement layer.

---

## Summary

This system is:

- A multi-agent AI data scientist  
- A tool-governed LLM orchestration platform  
- A policy-enforced execution environment  
- A modular foundation for agentic experimentation  

It combines structured reasoning, deterministic execution, secure tool access, and extensible architecture within a controlled backend framework.

