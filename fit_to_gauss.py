import numpy as np 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import os
from scipy.special import erf
from sklearn.metrics import r2_score

speed=pd.read_csv('/home/alexa/experiment_2/speed.csv',sep='\s+')
column_index = speed.columns.tolist()
column_index.pop(0)

channels=speed['Channels'].to_numpy()

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


uncalibrated_time=[]
uncalibrated_time_stdev=[]
jacknife_error=[]

def jackknife_gaussian(x, y, initial_guess):
    """
    Returns: (mu_mean, mu_stderr, sigma_mean)
    """
    n = len(x)
    mu_samples = []
    sigma_samples = []

    for i in range(n):
        # leave-one-out sample
        x_jk = np.delete(x, i)
        y_jk = np.delete(y, i)
        try:
            params, _ = curve_fit(gaussian, x_jk, y_jk, p0=initial_guess, maxfev=5000)
            A_jk, mu_jk, sigma_jk = params
            mu_samples.append(mu_jk)
            sigma_samples.append(sigma_jk)
        except RuntimeError:
            # skip failed fits
            continue
    return  np.std(mu_samples, ddof=1) * np.sqrt((n - 1) / n)

for distance in column_index: 
    x=channels
    y=speed[distance].to_numpy()
    initial_guess = [max(y), np.mean(x), 5]
    params, cov = curve_fit(gaussian,x,y,p0=initial_guess)
    A, mu, sigma = params
    uncalibrated_time.append(mu)
    uncalibrated_time_stdev.append(sigma)
    print(f"mu = {mu:.3f}"+f", sigma = {sigma:.3f}")
    jacknife_error.append(jackknife_gaussian(x, y, initial_guess))
    # Plot
    plt.scatter(x, y, color='red', label='Data')
    plt.plot(x, gaussian(x, *params), color='blue', label='Gaussian fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
x_error=jacknife_error


def calibration_function(x):
    return 1*2/360*1/5000000 * (x-176)*10**9 /2 # in ns 




calibrated_time=calibration_function(np.array(uncalibrated_time))
calibrated_time_stdev=1*2/360*1/5000000*10**9*(np.array(uncalibrated_time_stdev))
print(calibrated_time)
print(column_index)
print(calibrated_time_stdev)



# find y err by delta V=\sqrt(delta d/d*2+delta t/t *2)
y_error=[0.001,0.001,0.001,0.001,0.001]




def linear(x,a,b):
    return a*x + b



column_index_float = [float(i) for i in column_index]
params, cov = curve_fit(linear,calibrated_time,column_index_float,sigma=calibrated_time_stdev)
a, b = params 
plt.errorbar(calibrated_time,column_index_float,yerr=y_error,xerr=x_error,fmt='o',label='Speed Data at Various Distances', elinewidth=2,  capsize=5, capthick=2 )
plt.xlabel('Calibrated Time (ns)')
plt.title('Muon Distance vs Time')
plt.ylabel('Distance (m)')
plt.plot(calibrated_time,linear(np.array(calibrated_time),*params),color='red')
plt.show()



print(f"Calibration parameters: speed of muon = {a*10:.6f} x10^8 m/s")




