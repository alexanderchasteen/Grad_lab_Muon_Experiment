import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === Load data ===
time = pd.read_csv('times.csv').iloc[:,0].to_numpy()
counts = pd.read_csv('counts.csv').iloc[:,0].to_numpy()

# === Calibration ===
def calibration_function(x):
    return 1*2/360*1/5000000*(x - 176)*1e9  # in ns

calibrated_time = calibration_function(time)

# === Define exponential model using lifetime τ ===
def exponential_model(t, N0, tau):
    return N0 * np.exp(-t / tau)

# === Apply mask: use times > 150 ns to avoid rapid early drop ===
mask = calibrated_time > 150
time_masked = calibrated_time[mask]
counts_masked = counts[mask]

# === Fit exponential ===
p0 = [max(counts_masked), 2200]  # initial guess: N0, τ ~ 2200 ns
params, cov = curve_fit(exponential_model, time_masked, counts_masked, p0=p0)
N0_fit, tau_fit = params

# === Plotting ===
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22
})

plt.scatter(calibrated_time, counts, label='All Data', alpha=0.5)
plt.xlabel('Calibrated Time (ns)')
plt.ylabel('Counts')
plt.title('Counts vs Time')
plt.legend()
plt.show()


from sklearn.metrics import r2_score
r_squared = r2_score(counts_masked, exponential_model(time_masked, N0_fit, tau_fit))
print(f"R^2 of the fit = {r_squared:.4f}")
perr = np.sqrt(np.diag(cov))
a_err, b_err = perr

print(a_err,b_err)
# === Plot with R^2 in legend ===
plt.scatter(time_masked, counts_masked, color='orange', label='Fit Range (>150 ns)')
plt.plot(time_masked, exponential_model(time_masked, N0_fit, tau_fit), color='red',
          label=f'Fit: τ={tau_fit:.1f} ± {b_err:.2f} ns, R²={r_squared:.3f}')
plt.xlabel('Calibrated Time (ns)')
plt.ylabel('Counts')
plt.title('Exponential Fit to Muon Decay')
plt.legend()
plt.show()

# === Print results ===
print(f"Fitted initial count N0 = {N0_fit:.2f}")
print(f"Muon lifetime τ = {tau_fit:.2f} ns")

