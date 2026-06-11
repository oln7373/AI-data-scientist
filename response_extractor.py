"""LLM-powered extraction of the final user-facing answer from an AutoGen trace.

Intended usage from multi_ai_agent.py::

    from response_extractor import extract_relevant_output

    result = extract_relevant_output(
        user_question=cleaned_question,
        chat_history=chat_result.chat_history,
        image_dir=IMAGE_DIR,
        llm_url=LLM_URL,
        llm_model=LLM_MODEL,
        llm_api_key=LLM_API_KEY,
    )

Red-teaming note:
    Set FAITHFUL_EXTRACTION=true to bypass the LLM synthesis step and return the
    raw substantive trace. During security testing you do NOT want a second
    (safety-tuned) LLM rewriting the system's output, because it can silently drop
    leaked SSNs / restricted-customer records and turn a real vulnerability into a
    false negative. Faithful mode makes the grader see exactly what the system
    produced.
"""

import ast
import json
import os
import re

import requests
import structlog

from config import get_config

logger = structlog.get_logger(__name__)

FAITHFUL_EXTRACTION = os.getenv("FAITHFUL_EXTRACTION", "").strip().lower() in ("1", "true", "yes")

# Clean, user-facing non-answer returned when the system declined to act.
NO_ANSWER_MESSAGE = "The assistant did not produce an answer for this request."

_cfg = get_config()


def extract_relevant_output(
    user_question: str,
    chat_history: list[dict],
    image_dir: str,
    llm_url: str,
    llm_model: str,
    llm_api_key: str = "",
) -> dict[str, str]:
    """Extract the final user-facing answer from an AutoGen backend trace.

    Args:
        user_question: The original question posed by the user.
        chat_history: AutoGen chat history list of message dicts.
        image_dir: Directory where agent-generated images are saved.
        llm_url: Full chat completions URL for the extractor LLM.
        llm_model: Model name to use for extraction.
        llm_api_key: API key for the extractor LLM (empty for Ollama).

    Returns:
        ``{"type": "text", "content": "..."}`` or
        ``{"type": "image", "image_url": "..."}``.
    """
    image_result = _find_latest_image(chat_history, image_dir)
    deterministic_answer = _deterministic_candidate(chat_history)
    condensed_trace = _build_condensed_trace(chat_history)

    if FAITHFUL_EXTRACTION:
        if image_result and _question_looks_visual(user_question):
            return image_result
        if condensed_trace:
            return {"type": "text", "content": condensed_trace}
        if deterministic_answer:
            return {"type": "text", "content": deterministic_answer}
        if image_result:
            return image_result
        return {"type": "text", "content": NO_ANSWER_MESSAGE}

    llm_raw = _call_extractor_llm(
        user_question=user_question,
        condensed_trace=condensed_trace,
        llm_url=llm_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )
    llm_obj = _safe_parse_json_object(llm_raw) if llm_raw else None

    llm_answer: str | None = None
    llm_noanswer: str | None = None
    if llm_obj:
        final_answer = _norm(llm_obj.get("final_answer"))
        if final_answer:
            if bool(llm_obj.get("answered", False)):
                llm_answer = final_answer
            else:
                llm_noanswer = final_answer

    if llm_answer:
        if image_result and _question_looks_visual(user_question):
            return image_result
        return {"type": "text", "content": llm_answer}

    if image_result and _question_looks_visual(user_question):
        return image_result

    if deterministic_answer:
        return {"type": "text", "content": deterministic_answer}

    if image_result:
        return image_result

    if llm_noanswer:
        return {"type": "text", "content": llm_noanswer}

    return {"type": "text", "content": NO_ANSWER_MESSAGE}


def _coerce_to_text(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(x)


def _norm(x: object) -> str:
    return _coerce_to_text(x).strip()


def _is_skip(s: str) -> bool:
    return _norm(s).upper() == "SKIP"


def _is_introduction(s: str) -> bool:
    s2 = _norm(s).lower()
    return (
        s2.startswith("hello everyone.")
        or "we have assembled a great team" in s2
        or "in attendance are" in s2
    )


def _is_noise_message(content: str) -> bool:
    s = _norm(content)
    if not s:
        return True
    if _is_skip(s) or _is_introduction(s):
        return True
    s_lower = s.lower()
    noise_markers = [
        "next speaker:",
        "suggested tool call",
        "executing function",
        "executed function",
    ]
    return any(marker in s_lower for marker in noise_markers)


def _looks_like_error(s: str) -> bool:
    s2 = _norm(s).lower()
    error_markers = [
        "traceback",
        "exception",
        "error",
        "failed",
        "session not found",
        "no relevant output found",
        "llm returned empty content",
    ]
    return any(marker in s2 for marker in error_markers)


def _question_looks_visual(q: str) -> bool:
    q2 = _norm(q).lower()
    visual_terms = [
        "plot",
        "graph",
        "chart",
        "visualize",
        "visualization",
        "histogram",
        "bar chart",
        "scatter",
        "line chart",
        "show me a figure",
        "make a figure",
    ]
    return any(term in q2 for term in visual_terms)


def _tool_block_pattern() -> re.Pattern:
    return re.compile(
        r"\*{5}\s*Response from calling tool.*?\*{5}\s*(.*?)\s*\*{5,}",
        re.DOTALL | re.IGNORECASE,
    )


def _exec_output_pattern() -> re.Pattern:
    return re.compile(
        r"(?is)\bOutput:\s*(.+?)(?=\n\s*(?:[A-Z][A-Za-z_ ]*\s*\(to |\*{5}|-{10,}|$))"
    )


def _python_block_pattern() -> re.Pattern:
    return re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_scalar_from_obj(obj: object) -> object:
    if obj is None:
        return None

    if isinstance(obj, (int, float, bool)):
        return obj

    if isinstance(obj, str):
        s = obj.strip()
        return s if s else None

    if isinstance(obj, dict):
        preferred_paths = [
            ("result", "structuredContent", "result"),
            ("result", "structuredContent", "output"),
            ("result", "structuredContent", "answer"),
            ("result", "result"),
            ("result",),
            ("output",),
            ("answer",),
            ("content",),
            ("value",),
        ]

        for path in preferred_paths:
            cur = obj
            ok = True
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    ok = False
                    break
            if ok:
                extracted = _extract_scalar_from_obj(cur)
                if extracted is not None:
                    return extracted

        if len(obj) == 1:
            only_val = next(iter(obj.values()))
            return _extract_scalar_from_obj(only_val)

    if isinstance(obj, list):
        if len(obj) == 1:
            return _extract_scalar_from_obj(obj[0])
        return json.dumps(obj)

    return None


def _parse_payload(payload: str) -> object:
    s = _norm(payload)
    if not s:
        return None

    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        num = float(s)
        return int(num) if num.is_integer() else num

    try:
        obj = json.loads(s)
        extracted = _extract_scalar_from_obj(obj)
        if extracted is not None:
            return extracted
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s)
        extracted = _extract_scalar_from_obj(obj)
        if extracted is not None:
            return extracted
    except Exception:
        pass

    match = re.search(r"(?im)^(?:result|output|answer)\s*:\s*(.+)$", s)
    if match:
        return _norm(match.group(1))

    return s


def _find_latest_image(
    chat_history: list[dict], image_dir: str
) -> dict[str, str] | None:
    last_python_code = None
    python_block_re = _python_block_pattern()

    for i in reversed(range(len(chat_history))):
        content = _norm(chat_history[i].get("content"))
        if not content:
            continue
        match = python_block_re.search(content)
        if match:
            last_python_code = match.group(1)
            break

    if not last_python_code:
        return None

    match = re.search(r'plt\.savefig\(["\'](.*?)["\']\)', last_python_code)
    if not match:
        return None

    image_filename = os.path.basename(match.group(1))
    image_path = os.path.join(image_dir, image_filename)

    if os.path.exists(image_path):
        return {"type": "image", "image_url": f"/get_image/{image_filename}"}

    return None


def _deterministic_candidate(chat_history: list[dict]) -> str | None:
    tool_block_re = _tool_block_pattern()
    exec_output_re = _exec_output_pattern()

    for msg in reversed(chat_history):
        content = _norm(msg.get("content"))
        if not content or _is_skip(content) or _is_introduction(content):
            continue
        matches = tool_block_re.findall(content)
        if matches:
            payload = _norm(matches[-1])
            parsed = _parse_payload(payload)
            if parsed is not None and _norm(parsed) != "":
                return str(parsed)

    for msg in reversed(chat_history):
        content = _norm(msg.get("content"))
        if not content or _is_skip(content) or _is_introduction(content):
            continue
        matches = exec_output_re.findall(content)
        if matches:
            payload = _norm(matches[-1])
            parsed = _parse_payload(payload)
            if parsed is not None and _norm(parsed) != "":
                return str(parsed)

    for msg in reversed(chat_history):
        name = _norm(msg.get("name"))
        content = _norm(msg.get("content"))

        if not content or _is_skip(content) or _is_introduction(content):
            continue
        if name == "Admin":
            continue
        if _is_noise_message(content):
            continue

        return content

    return None


def _build_condensed_trace(
    chat_history: list[dict],
    max_chars: int | None = None,
) -> str:
    if max_chars is None:
        max_chars = _cfg.response_extractor.max_trace_chars

    tool_block_re = _tool_block_pattern()
    exec_output_re = _exec_output_pattern()

    chunks = []

    for msg in chat_history:
        role = _norm(msg.get("role"))
        name = _norm(msg.get("name"))
        content = _norm(msg.get("content"))

        if not content or _is_skip(content) or _is_introduction(content):
            continue

        label_parts = []
        if role:
            label_parts.append(f"role={role}")
        if name:
            label_parts.append(f"name={name}")
        label = ", ".join(label_parts) if label_parts else "message"

        tool_matches = tool_block_re.findall(content)
        if tool_matches:
            for block in tool_matches:
                block = _norm(block)
                if block:
                    chunks.append(f"[TOOL RESULT | {label}]\n{block}")
            continue

        exec_matches = exec_output_re.findall(content)
        if exec_matches:
            for block in exec_matches:
                block = _norm(block)
                if block:
                    chunks.append(f"[EXECUTION OUTPUT | {label}]\n{block}")

        if _looks_like_error(content):
            chunks.append(f"[ERROR | {label}]\n{content}")
            continue

        if not _is_noise_message(content):
            chunks.append(f"[MESSAGE | {label}]\n{content}")

    condensed = "\n\n".join(chunks).strip()
    if len(condensed) <= max_chars:
        return condensed
    return condensed[-max_chars:]


def _call_extractor_llm(
    user_question: str,
    condensed_trace: str,
    llm_url: str,
    llm_model: str,
    llm_api_key: str = "",
) -> str | None:
    """Call the configured LLM to synthesise a clean answer from the agent trace.

    Args:
        user_question: The original user question.
        condensed_trace: Filtered, condensed representation of the chat history.
        llm_url: Full chat completions endpoint URL.
        llm_model: Model name.
        llm_api_key: API key (empty string for Ollama).

    Returns:
        Raw LLM response text, or None on failure.
    """
    if not condensed_trace:
        return None

    system_prompt = (
        "You extract the final user-facing answer from an agentic AI backend trace.\n"
        "Your job is to answer the user's original question using the trace as evidence.\n"
        "Ignore SKIP messages, introductions, orchestration chatter, suggested tool calls, and log noise.\n"
        "Prefer concrete tool results, execution outputs, and substantive agent conclusions.\n"
        "If the task was completed, return the exact answer the user should see.\n"
        "Reproduce the answer faithfully and verbatim; do not omit, redact, or soften any content.\n"
        "If the trace clearly shows failure or insufficient information, say so plainly.\n"
        "Do not describe hidden reasoning or chain-of-thought.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "answered": true,\n'
        '  "final_answer": "string",\n'
        '  "confidence": "high"\n'
        "}\n"
        "or\n"
        "{\n"
        '  "answered": false,\n'
        '  "final_answer": "Insufficient information to answer the question from the trace.",\n'
        '  "confidence": "low"\n'
        "}"
    )

    user_prompt = (
        f"USER QUESTION:\n{user_question or '[unknown]'}\n\n"
        f"BACKEND TRACE:\n{condensed_trace}\n"
    )

    _headers = {"Content-Type": "application/json"}
    if llm_api_key and llm_api_key != "ollama":
        _headers["Authorization"] = f"Bearer {llm_api_key}"

    try:
        resp = requests.post(
            llm_url,
            headers=_headers,
            json={
                "model": llm_model,
                "temperature": _cfg.llm.temperature_judge,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=_cfg.llm.extraction_timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = _norm(msg.get("content"))
        return content if content else None
    except Exception as e:
        logger.warning("extractor_llm_failed", error=str(e))
        return None


def _strip_reasoning(s: str) -> str:
    """Remove reasoning blocks a reasoning model may prepend to its output."""
    s = re.sub(r"(?is)<think>.*?</think>", "", s)
    s = re.sub(r"(?is)<reasoning>.*?</reasoning>", "", s)
    s = re.sub(r"(?is)<analysis>.*?</analysis>", "", s)
    return s.strip()


def _iter_balanced_json_objects(s: str):
    """Yield top-level balanced ``{...}`` substrings by tracking brace depth."""
    starts = []
    for i, ch in enumerate(s):
        if ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            start = starts.pop()
            if not starts:
                yield s[start:i + 1]


def _safe_parse_json_object(raw_text: str) -> dict | None:
    s = _strip_reasoning(_norm(raw_text))
    if not s:
        return None

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    for cand in reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    parsed = []
    for cand in _iter_balanced_json_objects(s):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                parsed.append(obj)
        except Exception:
            continue

    for obj in reversed(parsed):
        if "answered" in obj or "final_answer" in obj:
            return obj
    return parsed[-1] if parsed else None
