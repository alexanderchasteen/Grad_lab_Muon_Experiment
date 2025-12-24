import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

# Data
x = np.array([1.04959213, 2.88958209, 5.47271974, 8.55664291, 8.9031786])
y = np.array([0.728, 1.023, 1.48, 1.59, 1.75])

# x-errors (your specification)
xerr = 2*1.11      # sigma_x
yerr=[0.001,0.001,0.001,0.001,0.001]        # no y errors provided

# Linear model for ODR: y = m*x + b
def linear(beta, x):
    m, b = beta
    return m*x + b

model = Model(linear)

# Set up data with x-errors
data = RealData(x, y, sx=xerr, sy=yerr)

# Run ODR
odr = ODR(data, model, beta0=[1, 0])  # initial guess
out = odr.run()

m, b = out.beta
dm, db = out.sd_beta

print("Slope     = {:.6f} ± {:.6f}".format(m, dm))
print("Intercept = {:.6f} ± {:.6f}".format(b, db))

# Generate fit line
xfit = np.linspace(min(x), max(x), 300)
yfit = m * xfit + b

# Plot
plt.figure(figsize=(8,5))
plt.errorbar(x, y, xerr=xerr, fmt='o', capsize=4, label='Data with x-errors')
plt.plot(xfit, yfit, label='ODR linear fit')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Fit with x-Errors (ODR)")
plt.legend()
plt.tight_layout()
plt.show()
