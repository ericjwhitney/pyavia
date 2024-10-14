#!/usr/bin/env python3
"""
Example use of a generalised Savitzky–Golay filter for unequally
spaced datapoints.
"""
import matplotlib.pyplot as plt
import numpy as np

from pyavia.data import savgol_variable


# Written by Eric J. Whitney, March 2023

# ======================================================================

def func(x):
    # return x ** 2 - 2 * x - 3  # Different trial functions.
    return np.sin(x) + 0.35


# Generate reference curve.
x_ref = np.linspace(-5, 5, num=100)
y_ref = func(x_ref)

# Generate unequally spaced x 'data' points by adding uniform random
# noise.  Then generate y values using the example equation and add
# normally distributed noise.
x_data = np.linspace(np.min(x_ref), np.max(x_ref), num=40)
Δx = x_data[1] - x_data[0]
x_data += np.random.uniform(-0.5 * Δx, 0.5 * Δx, x_data.shape)
y_σ = 0.05 * (np.max(y_ref) - np.min(y_ref))
y_data = func(x_data) + np.random.normal(0.0, y_σ, x_data.shape)

# Apply Savitzky–Golay filter (different options).
# y_filt = savgol_variable(x_data, y_data, window=21, order=5)
y_filt = savgol_variable(x_data, y_data, window=7, order=3, passes=5)


# Plot results.
plt.figure()
plt.plot(x_data, y_data, '^k', label="RAW DATA")
plt.plot(x_data, y_filt, '-b', label="SAV-GOL FILTER")
plt.plot(x_ref, y_ref, '--r', label="REF $f(x)$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid()
plt.legend(loc='lower left')
plt.show()
