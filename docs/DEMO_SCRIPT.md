# Megha-Drishti: 5-Minute Hackathon Demo Script

## 0:00 - 0:30 | The Problem
*(Jury introduction)*
"Good afternoon. Operational weather forecasting in India faces a unique challenge: we have access to incredible global physics models like ECMWF and NOAA, powerful new AI models like GraphCast, and domestic NCMRWF models. But how do forecasters decide which model to trust during complex events like the monsoon or western disturbances? 

Our solution is **Megha-Drishti**: an intelligent, dynamic forecast blending system."

## 0:30 - 2:00 | The Pipeline (Terminal)
*(Switch to terminal)*
"First, I'll show you how lightweight the system is. We trigger our daily operational pipeline."
*(Run `python scripts/run_daily.py --demo`)*
"Instantly, the orchestrator begins:
1. It ingests the distinct GRIB/NetCDF outputs.
2. It standardizes them to a strict 0.25° grid over India.
3. It maps the current synoptic state—is it an active monsoon, or a break?
4. **The core innovation**: A PyTorch Mixture-of-Experts gating network reviews the historical 14-day skill of each model, looks at the current weather regime, and assigns a hyper-local weight to each model per pixel."
*(Pipeline completes in seconds)*
"It's already done. The output is a single, optimal, bias-corrected NetCDF."

## 2:00 - 3:30 | The Dashboard (Streamlit)
*(Switch to Dashboard - Forecast & Weights Page)*
"We've built a native API and dashboard on top of this. 
Here on the **Weight Maps**, you can physically see the AI reasoning. Notice how ECMWF might be given heavy weighting over the Western Ghats due to its historical accuracy in orographic rainfall, while GraphCast is trusted more over the central plains for temperature."

## 3:30 - 4:30 | Extreme Guidance
*(Switch to Extreme Guidance Page)*
"Because numerical models often underpredict the extremes of heavy rain, we apply **EMOS Calibration** to stretch the probabilities into a true predictive distribution. 
Here, we map these PDFs strictly against the official IMD thresholds.
With one click, we can view the Heatwave or Heavy Rain exceedance probability maps, localized down to a specific district sorting table for disaster management authorities."

## 4:30 - 5:00 | Transparency & Conclusion
*(Switch to Skill and Drift Page)*
"Finally, we don't hide our errors. The system automatically tracks 'Model Drift', raising flags if a newly deployed AI model suddenly degrades in performance. 
We've achieved a dynamic, scientifically grounded, regime-aware blending system ready for NCMRWF deployment. Thank you."
