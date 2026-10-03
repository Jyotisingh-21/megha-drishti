import re
import sys

content = open("src/nwpblend/pipeline.py", "r").read()

old_save = r"""        os\.makedirs\("data/logs", exist_ok=True\)
        timestamp = datetime\.now\(UTC\)\.strftime\("%Y%m%d_%H%M%S"\)
        with open\(f"data/logs/run_report_\{timestamp\}\.json", "w"\) as f:
            json\.dump\(report, f, indent=2\)

        # For dashboard
        with open\("data/logs/run_report_latest\.json", "w"\) as f:
            json\.dump\(report, f, indent=2\)"""

new_save = """        os.makedirs("data/logs", exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        with open(f"data/logs/run_report_{timestamp}.json", "w") as f:
            json.dump(report, f, indent=2)

        # For dashboard, only overwrite if not FAILED completely
        if report["status"] != "FAILED":
            with open("data/logs/run_report_latest.json", "w") as f:
                json.dump(report, f, indent=2)"""

if 'if report["status"] != "FAILED":' not in content:
    content = re.sub(old_save, new_save, content)

open("src/nwpblend/pipeline.py", "w").write(content)
