import math
import numpy as np
import scipy
# Symbolic Python
from sympy import *
import matplotlib.pyplot as plt


# Defining the function for Taylor Series
def taylor_series(X, p, nth=2):
    # Defining x as a symbol
    x = symbols('x')
    # Defining the function. Modify the f for your desired function.
    #f = math.e**x
    f = (x**2) - (0.7 * x**3) + math.e**x
    # Summing the values of Taylor Series into g
    g = np.zeros(X.shape)
    for n in range(nth+1):
        # Taking the nth derivative of f respect to x
        dnf_dxn = diff(f, x, n)
        # lambdify method to convert the function to take values for x
        dnf_dxn = lambdify(x, dnf_dxn)
        gnx = dnf_dxn(p) * np.power((X-p),n) * (1/math.factorial(n))
        g += gnx

    # The real function
    f = lambdify(x, f)
    fx = f(X)
    fp = f(p)
    return g, fx, fp


# Plotting the real function vs. the nth order Taylor
def plot_taylor_series(X, p, order=0):
    function_vals = taylor_series(X, p, 0)[1]
    fp = taylor_series(X, p, 0)[2]
    n_plots = order + 1

    fig, axs = plt.subplots(n_plots, 1, figsize=(6, 18))
    taylor_series_vals = []
    for i in range(order+1):
        taylor_series_vals.append(taylor_series(X, p, i)[0])

    for i in range(n_plots):
        axs[i].plot(X, taylor_series_vals[i], label=f"{i} Order Maclaurin Series", linestyle='--', color='red')
        axs[i].plot(X, function_vals, label='Original function', color='black')
        axs[i].axvline(x=p, color='orange', linestyle='dotted', label=f"p = {p}")
        axs[i].legend()
        
    plt.tight_layout()
    plt.savefig('taylor.png', format='png')
    plt.show();


# An example
x = np.linspace(-3, 3, 50)
plot_taylor_series(x, 1, 5)
