import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

# === Load data ===
df = pd.read_csv('/home/alexa/experiment_2/times.csv')
time = df.iloc[:, 0].to_numpy()

df2 = pd.read_csv('/home/alexa/experiment_2/counts.csv')
counts = df2.iloc[:, 0].to_numpy()

# === Calibration ===
def calibration_function(x):
    return 1*2/360*1/5000000*(x - 176)*1e9  # in ns

calibrated_time = calibration_function(time)

# === Define exponential model ===
def exponential(x, A, B):
    return B * np.exp(-A * x)

# === Apply mask: keep only times < 200 ns ===
mask = calibrated_time > 100
time_masked = calibrated_time[mask]
counts_masked = counts[mask]

# === Fit ===
params, cov = curve_fit(exponential, time_masked, counts_masked, p0=(0.01, max(counts_masked)))
A, B = params

# === Plot ===
plt.scatter(calibrated_time, counts, label='All Data', alpha=0.5)
plt.scatter(time_masked, counts_masked, color='orange', label='Fitted Range (>200 ns)')
plt.plot(time_masked, exponential(time_masked, A, B), color='red', label='Fit')
plt.xlabel('Calibrated Time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()

# === Print results ===
print(f"Decay constant A = {A:.5f} ns⁻¹")
print(f"Muon lifetime τ = {1/A:.2f} ns")

