"""Shared pytest configuration for the AI data scientist test suite."""

import sys
from pathlib import Path

# Make project root importable from every test module.
sys.path.insert(0, str(Path(__file__).parent.parent))
