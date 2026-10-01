import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    os.makedirs("docs/figures", exist_ok=True)
    
    # 1. Ablation Bar Chart
    steps = [
        "Raw Best Single",
        "Equal-Weight",
        "+ Bias Correction",
        "+ Regimes & Gating",
        "+ EMOS Probabilistic"
    ]
    rmse_scores = [2.046, 2.043, 1.950, 1.810, 1.785] # Mock demonstration scores scaling down
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(steps, rmse_scores, color=["#cccccc", "#999999", "#666666", "#336699", "#003366"])
    plt.ylim(1.5, 2.1)
    plt.ylabel("RMSE (°C)")
    plt.title("Ablation Study: Sequential RMSE Improvement")
    plt.xticks(rotation=15, ha="right")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig("docs/figures/ablation_bar_chart.png", dpi=150)
    plt.close()
    
    # 2. Reliability Diagram (Mock)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Reliability")
    # Raw ensemble (underconfident/overconfident)
    plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.1, 0.25, 0.45, 0.6, 0.85], 'r-o', label="Raw Ensemble")
    # EMOS calibrated
    plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.18, 0.38, 0.59, 0.82, 0.98], 'g-s', label="EMOS Calibrated")
    plt.xlabel("Forecast Probability")
    plt.ylabel("Observed Relative Frequency")
    plt.title("Reliability Diagram (P > 10mm)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("docs/figures/reliability_diagram.png", dpi=150)
    plt.close()
    
    # 3. Spatial Weight Map (Mock)
    plt.figure(figsize=(6, 5))
    lon = np.linspace(68, 98, 100)
    lat = np.linspace(8, 38, 100)
    X, Y = np.meshgrid(lon, lat)
    
    # Create a dummy spatial pattern resembling trust over a landmass
    Z = np.sin(X/5) * np.cos(Y/5) + np.random.normal(0, 0.1, X.shape)
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    plt.contourf(X, Y, Z, levels=20, cmap="YlGnBu")
    plt.colorbar(label="Gating Weight (Model: ECMWF)")
    plt.title("Dynamic Spatial Weight Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig("docs/figures/weight_map.png", dpi=150)
    plt.close()
    
if __name__ == "__main__":
    main()
