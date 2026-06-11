# Agentic AI Data Scientist + MCP Integration

## Overview

This repository implements an **Agentic AI Data Scientist** system extended with **MCP (Model Context Protocol)** tool integration.

The system consists of:

- A **FastAPI backend**
- A **Streamlit frontend**
- An **LLM backend** — either a local model served by **Ollama**, or any **OpenAI-compatible API** (OpenAI, OpenRouter, Groq, etc.)
- A **multi-agent AutoGen** orchestration layer
- A separate **MCP server** for tool execution

The backend enforces strict tool governance via an allowlist, ensuring that all external side effects (e.g., sending emails) are executed through MCP rather than directly by the LLM.

When using Ollama, the code and dependencies presuppose access to an **NVIDIA GPU** (e.g., Quadro RTX 8000, A100). GPU usage is strongly recommended for practical performance. When using an OpenAI-compatible API, no local GPU is required.

---

## System Architecture

There are two primary execution pathways:

### 1. Multi-Agent Data Scientist

User → `/multi_ai_agent` → AutoGen multi-agent orchestration →  
Code generation → Executor → Result (text/image) → Optional email via MCP

This pathway performs structured data analysis over `customer_shopping_data.csv`.

### 2. Direct LLM Tool Calling

User → `/mcp_agent_add` → LLM → `tool_calls` →  
Backend allowlist enforcement → MCP execution → Return result

This pathway demonstrates OpenAI-style function calling.

---

## Dependencies

### Python

Python 3.10+. All required packages are listed in `myenv.txt`.

**Logging:** all modules use [structlog](https://www.structlog.org) for structured JSON logging. `print()` statements are not used in any committed code. Logs are emitted as machine-readable JSON to stdout. `structlog` must be installed (it is included in `myenv.txt`).

### LLM Backend (Required — choose one)

**Option A — Ollama (local):** serves models on your own hardware. Requires a GPU for practical performance.

**Option B — OpenAI-compatible API:** use OpenAI, [OpenRouter](https://openrouter.ai) (includes free models), Groq, or any other OpenAI-compatible provider. No local GPU needed.

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

Edit `.env` and configure your LLM backend. **Choose one of the two options below.**

---

#### Option A — Ollama (local GPU)

```env
LLM_PROVIDER=ollama
OLLAMA_PORT=11434        # 11435 for Quadro RTX 8000, 11434 for A100
OLLAMA_MODEL=gpt-oss:20b # or whichever model you have pulled
```

Ollama must be installed, running (`ollama serve`), and the model must be pulled before starting the backend.

---

#### Option B — OpenAI-compatible API (no local GPU)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-api-key>
OPENAI_MODEL=gpt-4o               # or any model your provider supports
LLM_BASE_URL=                     # leave blank for OpenAI; set for other providers (see below)
```

**Using OpenRouter (includes free models):**

1. Create a free account at [openrouter.ai](https://openrouter.ai) and generate an API key.
2. Set the following in `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-or-v1-...
OPENAI_MODEL=meta-llama/llama-3.1-8b-instruct:free
LLM_BASE_URL=https://openrouter.ai/api/v1
```

Browse available free models at [openrouter.ai/models](https://openrouter.ai/models?order=newest&supported_parameters=free).

**Using Groq:**

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-groq-key>
OPENAI_MODEL=llama-3.1-8b-instant
LLM_BASE_URL=https://api.groq.com/openai/v1
```

---

#### Common settings (required regardless of provider)

```env
MCP_PORT=8005
ALLOWED_TOOLS=add_numbers,ping,send_email,compose_email
```

The `ALLOWED_TOOLS` variable controls which MCP tools may be executed by the backend.

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

`data/customer_shopping_data.csv`

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

Contains images and generated artifacts produced by the agent system.  
Visualization outputs generated by the Executor agent are saved here.

Required for backend execution.

---

### `data/`

Stores datasets used by the agentic data scientist.

The backend automatically downloads the default dataset (`customer_shopping_data.csv`) if it does not already exist.

---

### `figures/`

Static images used by the Streamlit frontend (logos, diagrams, UI elements).

---

### `backend.py`

Primary **FastAPI backend entry point**.

Responsibilities include:

- Starting the API server
- Registering routers for the agent system, summarizer, and RAG pipelines
- Managing the MCP client connection
- Enforcing the MCP tool allowlist policy
- Serving responses to the Streamlit frontend

This file forms the central governance layer of the system.

---

### `multi_ai_agent.py`

Defines the **AutoGen multi-agent orchestration system**.

Key responsibilities:

- Implements the Agentic AI Data Scientist pipeline
- Defines agents such as:
  - `FileReader`
  - `Summarizer`
  - `DataScientist`
  - `Visualization`
  - `Executor`
- Handles LLM reasoning via Ollama or any OpenAI-compatible API
- Coordinates code generation and execution via AutoGen group chat
- Extracts relevant outputs from agent execution
- Supports optional email delivery via MCP tools

It also exposes several backend endpoints including:

- `/multi_ai_agent`
- `/mcp_add_demo`
- `/mcp_agent_add`
- `/mcp_tools`

---

### `mcp_server.py`

Defines the **Model Context Protocol (MCP) server** that exposes tools to the backend.

Available MCP tools include:

- `ping` — connectivity test
- `add_numbers` — arithmetic demonstration tool
- `compose_email` — constructs email payloads
- `send_email` — sends email using the Gmail API

The MCP server isolates all external side effects from the LLM and backend logic.

---

### `mcp_http_client.py`

Implements a **minimal MCP JSON-RPC client** used by the backend to communicate with the MCP server.

Capabilities include:

- Initializing MCP sessions
- Listing available MCP tools
- Calling MCP tools through JSON-RPC requests
- Managing session IDs required by the MCP protocol

This client avoids issues with the official streaming MCP client implementation.

---

### `mcp_manager.py`

Provides an **asynchronous MCP connection manager** used by the backend.

Responsibilities include:

- Managing lifecycle of the MCP client
- Maintaining persistent connections
- Ensuring thread-safe tool calls
- Providing utility functions for listing tools and invoking MCP tools

This module abstracts MCP connection management from the rest of the backend.

---

### `app.py`

Streamlit **frontend interface** for interacting with the system.

Key features include:

- Agentic AI Data Scientist chat interface
- Document summarization interface
- RAG pipeline demo
- Navigation across multiple GenAI tools
- Display of generated charts and results

The frontend communicates with the backend via HTTP requests.

---

### `summarizer.py`

Implements the **document summarization API**.

Capabilities include:

- PDF document ingestion
- Text extraction and chunking
- Summarization using the configured LLM (Ollama or OpenAI-compatible API)
- Multiple summary styles:
  - abstractive
  - extractive
  - long
  - short

The module also supports summary evaluation using:

- ROUGE
- BLEU
- BERTScore
- embedding cosine similarity

---

### `rag.py`

Implements a **Retrieval-Augmented Generation (RAG) pipeline**.

Functionality includes:

- Building or loading a persistent vector index from documents
- Embedding documents using HuggingFace embedding models
- Semantic retrieval using LlamaIndex
- Querying retrieved context with the configured LLM (Ollama or OpenAI-compatible API)

The API endpoint `/rag-query` allows users to perform semantic document queries.

---

### `config.py`

Central **configuration loader and logging setup** module.

Responsibilities:

- Defines Pydantic models (`LLMConfig`, `RAGConfig`, `RedteamConfig`, `MCPConfig`, etc.) that validate `configs/default.json` at startup
- Exposes `get_config() -> AppConfig` — a cached singleton used by every module that needs runtime parameters
- Provides `configure_logging()` — configures structlog with JSON output; called once at each application entry point

All magic numbers (timeouts, temperatures, thresholds, paths) are read from `configs/default.json` via this module rather than being hard-coded in source files.

---

### `configs/default.json`

Single source of truth for all **runtime parameters** — timeouts, temperatures, model paths, dataset URLs, privacy thresholds, and MCP configuration.

All modules load their parameters from this file through `config.py`. Do not hard-code numeric values in source code; add them here instead.

To override a value for a specific run, pass it as a CLI argument or add a per-experiment config file. See `config.py` for the full schema.

---

### `redteam_controller.py`

Automated **red-team controller** that attacks the backend's privacy policy enforcement.

Features:

- 20 static adversarial prompts across four categories: `ssn_leak`, `restricted_leak`, `k_anonymity`, `prompt_injection`
- Deterministic SSN regex check (always applied)
- LLM-as-judge evaluation for each attack
- `--dynamic N` flag to generate additional LLM-crafted attacks at runtime
- `--out FILE` flag to save full results as JSON
- `--static-only` flag to skip the LLM judge and use only deterministic checks

Uses the same `LLM_PROVIDER` / `OPENAI_API_KEY` / `OPENAI_MODEL` settings from `.env`.

**Important:** for accurate results, start the backend with `FAITHFUL_EXTRACTION=true` so the extraction layer does not sanitize leaked data before it reaches the grader.

---

### `email_utils.py`

Utility module supporting **email extraction and formatting**.

Functions include:

- extracting email addresses from user prompts
- cleaning prompts before agent execution
- constructing email subject/body payloads
- generating absolute URLs for generated images

These utilities support the email delivery functionality in the agent system.

---

### `gmail_send.py`

Standalone Gmail API email sending utility.

Handles:

- OAuth authentication
- message construction
- encoding email payloads
- sending messages via the Gmail API

---

### `send-email.py`

Standalone script for sending a plain-text email via **SMTP** (no OAuth, no Google APIs).

Works with any SMTP provider (Gmail app passwords, Outlook, Yahoo, corporate relay). Configured entirely through environment variables — see the script's module docstring for the full variable list.

---

### `send-email-attach.py`

Extended SMTP script that adds support for **optional file attachments**.

Provides:

- MIME multipart message construction
- binary attachment encoding (base64)
- SMTP delivery via the same provider configuration as `send-email.py`

Useful for sending analysis outputs or generated images as attachments.

---

### `myenv.txt`

List of required Python dependencies.

Install with:

```bash
pip install -r myenv.txt
```

---

### `env.sample`

Example environment configuration file.

Defines required environment variables. Key variables:

```
LLM_PROVIDER           # "ollama" or "openai"
LLM_BASE_URL           # optional override for OpenRouter, Groq, etc.
OPENAI_API_KEY         # API key for OpenAI-compatible providers
OPENAI_MODEL           # model name (e.g. gpt-4o, meta-llama/llama-3.1-8b-instruct:free)
OLLAMA_PORT            # port Ollama is listening on (Ollama only)
OLLAMA_MODEL           # model name pulled in Ollama (Ollama only)
MCP_PORT               # port the MCP server runs on
ALLOWED_TOOLS          # comma-separated list of permitted MCP tools
BACKEND_URL            # backend base URL used by the Streamlit frontend (default: http://127.0.0.1:8001)
FAITHFUL_EXTRACTION    # set to "true" for red-team accuracy; bypasses LLM sanitization in response extraction
```

See step 8 for full configuration instructions for each provider.

---

## Summary

This system is:

- A multi-agent AI data scientist  
- A tool-governed LLM orchestration platform  
- A policy-enforced execution environment  
- A modular foundation for agentic experimentation  

It combines structured reasoning, deterministic execution, secure tool access, and extensible architecture within a controlled backend framework.

