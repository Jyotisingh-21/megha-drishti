import re

with open("src/nwpblend/pipeline.py", "r") as f:
    content = f.read()

# Update signature
content = content.replace(
    "def run_daily(date: str, domain: dict, demo: bool = False, skip_download: bool = False):",
    "def run_daily(date: str, domain: dict, demo: bool = False, skip_download: bool = False, quick: bool = False, max_leads: int = None):",
)

# Update logic inside run_daily
# Find: leads = [i * 24 for i in config.get("lead_times_days", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
leads_str = """
        leads = [i * 24 for i in config.get("lead_times_days", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
        if quick:
            leads = [24, 48, 72]
        if max_leads:
            leads = leads[:max_leads]
        
        time_budget = config.get("pipeline_time_budget_sec", 1800)
"""
content = content.replace(
    '        leads = [i * 24 for i in config.get("lead_times_days", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]',
    leads_str,
)

with open("src/nwpblend/pipeline.py", "w") as f:
    f.write(content)
print("pipeline updated")
