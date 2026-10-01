# Scheduling the Pipeline

Megha-Drishti is designed to be easily scheduled without requiring heavy workflow orchestration engines (like Prefect or Airflow), though they can be layered on top if desired. 

For operational use, a simple Linux `cron` configuration is the recommended and simplest approach.

## Cron Configuration

Open the crontab editor:
```bash
crontab -e
```

Add the following entry to run the pipeline automatically every day at 04:00 UTC (09:30 IST):

```cron
# Run Megha-Drishti pipeline daily at 04:00 UTC
0 4 * * * cd /path/to/megha-drishti && source .venv/bin/activate && python scripts/run_daily.py >> /var/log/megha-drishti.log 2>&1
```

## Resilience

The `run_daily.py` pipeline is built to handle upstream delays and errors gracefully:
1. **Timeouts**: Ingestion adapters have configurable timeouts.
2. **Missing Feeds**: If a model feed is completely missing or corrupted, the Mixture-of-Experts pipeline automatically dynamically renormalizes the weights across the remaining models.
3. **Reporting**: The pipeline dumps a structured JSON payload detailing successes and warnings into `data/logs/run_report_<date>.json`.
