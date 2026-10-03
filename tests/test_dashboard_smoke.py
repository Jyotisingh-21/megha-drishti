import importlib
import os
import re
import sys
from pathlib import Path

import pytest

# Add dashboard to sys.path so modules can find data_loader
sys.path.insert(0, str(Path("dashboard").absolute()))


def test_no_hardcoded_kpis():
    """Check dashboard files for suspicious hardcoded metric numbers."""
    dashboard_dir = Path("dashboard")

    # We look for st.metric calls with a literal float/int as the value
    # e.g. st.metric("Label", "12.5") or st.metric("Label", 12)
    bad_pattern = re.compile(
        r"st\.metric\s*\([^,]+,\s*(?:\"|\')?\d+(?:\.\d+)?(?:\"|\')?\s*(?:,|.*?)\)"
    )

    for file_path in dashboard_dir.rglob("*.py"):
        content = file_path.read_text(encoding="utf-8")
        matches = bad_pattern.findall(content)
        if matches:
            pytest.fail(f"Hardcoded KPI detected in {file_path}: {matches}")
