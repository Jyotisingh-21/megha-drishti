import importlib
import os
import re
import sys
from pathlib import Path

import pytest

# Add dashboard to sys.path so modules can find data_loader
sys.path.insert(0, str(Path("dashboard").absolute()))

def test_dashboard_smoke_imports():
    """Ensure all dashboard pages can be imported without crashing."""
    dashboard_dir = Path("dashboard/pages")
    for file in dashboard_dir.glob("*.py"):
        if file.name.startswith("__"):
            continue
        
        # We need to set up streamlit env just enough to not crash on import
        module_name = f"dashboard.pages.{file.stem}"
        try:
            importlib.import_module(module_name)
        except Exception as e:
            # We skip tests that fail on st.set_page_config inside pytest
            if "st.set_page_config" in str(e) or "set_page_config() can only be called once" in str(e):
                continue
            # Also ignore the Streamlit API Exception about no script thread
            if "missing ScriptRunContext" in str(e):
                continue
            pytest.fail(f"Failed to import {module_name}: {e}")

def test_no_hardcoded_kpis():
    """Check dashboard files for suspicious hardcoded metric numbers."""
    dashboard_dir = Path("dashboard")
    
    # We look for st.metric calls with a literal float/int as the value
    # e.g. st.metric("Label", "12.5") or st.metric("Label", 12)
    bad_pattern = re.compile(r'st\.metric\s*\([^,]+,\s*(?:\"|\')?\d+(?:\.\d+)?(?:\"|\')?\s*(?:,|.*?)\)')
    
    for file_path in dashboard_dir.rglob("*.py"):
        content = file_path.read_text(encoding="utf-8")
        matches = bad_pattern.findall(content)
        if matches:
            pytest.fail(f"Hardcoded KPI detected in {file_path}: {matches}")
