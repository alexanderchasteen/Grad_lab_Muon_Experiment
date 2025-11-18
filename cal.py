import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Data
x = np.array([178,179,180,181,182,183])
degree_to_ns = 1*2/360*1/5000000*10**9
y = np.array([2*degree_to_ns, 3*degree_to_ns, 4*degree_to_ns, 
              5*degree_to_ns, 6*degree_to_ns, 7*degree_to_ns])

# Linear model
def linear(x, m, b):
    return m*x + b

# Fit
popt, pcov = curve_fit(linear, x, y)
m, b = popt

# Predictions for R^2
y_pred = linear(x, m, b)
R2 = r2_score(y, y_pred)

print("slope m =", m)
print("intercept b =", b)
print("R^2 =", R2)

# Plot
plt.plot(x, y, 'o', label='Calibration Data')
plt.plot(x, y_pred, label=f'Linear Fit (R²={R2:.4f})')
plt.xlabel("Channel")
plt.title("Calibration Curve")
plt.ylabel("Time (ns)")
plt.legend()
plt.show()
