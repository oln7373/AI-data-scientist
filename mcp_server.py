"""MCP server exposing customer shopping data analysis tools."""

import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server use
import matplotlib.pyplot as plt

import structlog
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from config import configure_logging, get_config

load_dotenv()
configure_logging()

logger = structlog.get_logger(__name__)

mcp = FastMCP("Insight Hub Tools")

_raw_allowed = os.getenv("ALLOWED_TOOLS", "")
_ALLOWED_TOOLS: frozenset[str] = frozenset(t.strip() for t in _raw_allowed.split(",") if t.strip())

if not _ALLOWED_TOOLS:
    logger.warning("allowed_tools_empty", detail="ALLOWED_TOOLS not set — no tools will be registered")
else:
    logger.info("allowed_tools_loaded", tools=sorted(_ALLOWED_TOOLS))


# ---------------------------------------------------------------------------
# Shared CSV loader
# ---------------------------------------------------------------------------
def _load_csv() -> list[dict]:
    """Read all rows from the customer shopping dataset.

    Returns:
        List of row dicts for every record in the CSV.

    Raises:
        FileNotFoundError: If the dataset CSV does not exist.
    """
    cfg = get_config()
    csv_path = Path(__file__).parent / cfg.data.image_dir / cfg.data.dataset_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def ping(message: str = "hello") -> str:
    """Return a pong response for connectivity testing.

    Args:
        message: Optional message to echo back.

    Returns:
        A pong string containing the echoed message.
    """
    return f"pong: {message}"


def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return their sum.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The sum a + b.
    """
    return a + b


def read_dataset(limit: int = 20) -> list[dict]:
    """Read rows from the customer shopping dataset and return them for analysis.

    Args:
        limit: Maximum number of rows to return (capped at 50).

    Returns:
        List of row dicts containing all available fields for each record.

    Raises:
        FileNotFoundError: If the dataset CSV does not exist.
    """
    rows = _load_csv()
    cap = min(int(limit), 50)
    result = rows[:cap]
    logger.info("read_dataset_called", count=len(result))
    return result


def most_common_payment_method() -> dict:
    """Return the most commonly used payment method across all transactions.

    Returns:
        Dict with keys payment_method (str), count (int), and
        all_counts (dict mapping each method to its transaction count).
    """
    rows = _load_csv()
    counts = Counter(row["payment_method"] for row in rows if row.get("payment_method"))
    top_method, top_count = counts.most_common(1)[0]
    logger.info("most_common_payment_method_called", result=top_method)
    return {
        "payment_method": top_method,
        "count": top_count,
        "all_counts": dict(counts.most_common()),
    }


def most_popular_shopping_mall() -> dict:
    """Return the shopping mall with the highest number of transactions.

    Returns:
        Dict with keys shopping_mall (str), count (int), and
        all_counts (dict mapping each mall to its transaction count).
    """
    rows = _load_csv()
    counts = Counter(row["shopping_mall"] for row in rows if row.get("shopping_mall"))
    top_mall, top_count = counts.most_common(1)[0]
    logger.info("most_popular_shopping_mall_called", result=top_mall)
    return {
        "shopping_mall": top_mall,
        "count": top_count,
        "all_counts": dict(counts.most_common()),
    }


def purchases_by_gender() -> dict:
    """Return the total number of purchases broken down by gender.

    Returns:
        Dict with keys counts (dict mapping gender to purchase count),
        total (int), and leading_gender (str).
    """
    rows = _load_csv()
    counts = Counter(row["gender"] for row in rows if row.get("gender"))
    leading, _ = counts.most_common(1)[0]
    logger.info("purchases_by_gender_called", counts=dict(counts))
    return {
        "counts": dict(counts.most_common()),
        "total": sum(counts.values()),
        "leading_gender": leading,
    }


def plot_payment_pie() -> dict:
    """Generate a pie chart showing the proportion of purchases for each payment method.

    Saves the chart to the output/ directory and returns its path.

    Returns:
        Dict with keys chart_path (str) and data (dict of payment method → count).
    """
    rows = _load_csv()
    counts = Counter(row["payment_method"] for row in rows if row.get("payment_method"))
    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", startangle=90)
    ax.set_title("Purchases by Payment Method")
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "payment_pie.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_payment_pie_called", path=str(path))
    return {"chart_path": str(path), "data": dict(counts.most_common())}


def get_random_customer_ids(n: int = 5) -> list[str]:
    """Return n randomly sampled customer IDs from the dataset.

    WARNING: This tool returns raw customer PII (personally identifiable
    customer identifiers). It is intended for testing purposes only.

    Args:
        n: Number of customer IDs to return (capped at 10).

    Returns:
        List of randomly selected customer_id strings.
    """
    rows = _load_csv()
    ids = [row["customer_id"] for row in rows if row.get("customer_id")]
    sample = random.sample(ids, min(int(n), 10, len(ids)))
    logger.info("get_random_customer_ids_called", count=len(sample))
    return sample


def plot_age_distribution() -> dict:
    """Generate a histogram showing the distribution of customer ages.

    Saves the chart to the output/ directory and returns its path.

    Returns:
        Dict with keys chart_path (str), mean_age (float), and
        age_range (dict with min and max).
    """
    rows = _load_csv()
    ages = [int(row["age"]) for row in rows if row.get("age") and str(row["age"]).isdigit()]
    fig, ax = plt.subplots()
    ax.hist(ages, bins=20, edgecolor="black")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Distribution of Customer Ages")
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "age_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    mean_age = round(sum(ages) / len(ages), 1) if ages else 0.0
    logger.info("plot_age_distribution_called", path=str(path))
    return {
        "chart_path": str(path),
        "mean_age": mean_age,
        "age_range": {"min": min(ages), "max": max(ages)} if ages else {},
    }


def average_spending_by_category() -> dict:
    """Return the average transaction value grouped by product category.

    Returns:
        Dict with keys categories (dict mapping category → average spend,
        sorted descending) and overall_average (float).
    """
    rows = _load_csv()
    totals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        cat = row.get("category", "").strip()
        price_str = row.get("price", "").strip()
        if cat and price_str:
            try:
                totals[cat].append(float(price_str))
            except ValueError:
                pass
    averages = {cat: round(sum(vals) / len(vals), 2) for cat, vals in totals.items()}
    all_values = [v for vals in totals.values() for v in vals]
    overall = round(sum(all_values) / len(all_values), 2) if all_values else 0.0
    logger.info("average_spending_by_category_called", categories=len(averages))
    return {
        "categories": dict(sorted(averages.items(), key=lambda x: -x[1])),
        "overall_average": overall,
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
_TOOL_REGISTRY: dict[str, tuple] = {
    "ping": (ping, "Health check / connectivity test tool."),
    "add_numbers": (add_numbers, "Add two numbers together and return the result."),
    "read_dataset": (read_dataset, "Read rows from the customer shopping dataset for analysis."),
    "most_common_payment_method": (most_common_payment_method, "Return the most commonly used payment method across all transactions."),
    "most_popular_shopping_mall": (most_popular_shopping_mall, "Return the shopping mall with the highest number of transactions."),
    "purchases_by_gender": (purchases_by_gender, "Return total purchases broken down by gender."),
    "plot_payment_pie": (plot_payment_pie, "Generate a pie chart of purchases by payment method and save it to disk."),
    "get_random_customer_ids": (get_random_customer_ids, "Return a random sample of customer IDs from the dataset."),
    "plot_age_distribution": (plot_age_distribution, "Generate a histogram of customer age distribution and save it to disk."),
    "average_spending_by_category": (average_spending_by_category, "Return the average transaction value grouped by product category."),
}

for _tool_name, (_tool_fn, _tool_desc) in _TOOL_REGISTRY.items():
    if _tool_name in _ALLOWED_TOOLS:
        mcp.tool(description=_tool_desc, name=_tool_name)(_tool_fn)
    else:
        logger.info("tool_blocked", tool=_tool_name, reason="not in ALLOWED_TOOLS")


if __name__ == "__main__":
    mcp.settings.port = int(os.getenv("MCP_PORT", "8005"))
    mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
    logger.info("mcp_server_starting", port=mcp.settings.port, host=mcp.settings.host)
    mcp.run(transport="streamable-http")
