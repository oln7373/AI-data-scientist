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

### 2. Verify Ollama Installation

```bash
ollama --version


