import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import sem, norm, t

measures = np.array([296.74540119, 302.50714306, 300.31993942, 298.98658484, 294.56018640,
 294.55994520,  293.58083612, 301.66176146, 299.01115012, 300.08072578,
 293.20584494, 302.69909852, 301.32442641, 295.12339111, 294.81824967,
 294.83404510,  296.04242243, 298.24756432, 297.31945019, 295.91229140])

# Calculate the mean and standard error of the mean
mean = np.mean(measures)
sem_val = sem(measures)

# Calculate the degrees of freedom
dof = len(measures) - 1

# Calculate the t-statistic for a 95% confidence interval
t_stat = t.ppf(0.975, dof)  # 0.975 for two-tailed 95% CI

# Calculate the confidence interval
ci95 = t_stat * sem_val

# Round the error to one significant figure or two if the first digit is 1
if 1 <= abs(ci95) < 2:
    error_ci95 = round(ci95, -int(np.floor(np.log10(abs(ci95)))) + 1)
else:
    error_ci95 = round(ci95, -int(np.floor(np.log10(abs(ci95)))))

# Round the mean to the last significant digit of the error interval
precision = -int(np.floor(np.log10(abs(error_ci95))))  # Determine decimal places for rounding
rounded_mean = round(mean, precision)

print(f"The 95% confidence interval is: {rounded_mean:.{precision}f} +/- {error_ci95:.2g}")