import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import t

# System parameters
global m, c, k
m = 2
c = 0.5
k = 1.5

# Initial conditions
global T, x0, v0
T = 20
x0 = 10
v0 = 25

# Parameters for the analytic solution
nu = np.sqrt(4 * k * m - c**2) / (2 * m)
A = x0
B = (v0 + c * x0 / (2 * m)) / nu
r = c / (2 * m)

def analytic_solution(t):
    return np.exp(-r * t) * (A * np.cos(nu * t) + B * np.sin(nu * t))

# Define the differential equation
def dy(t, y, K, C):
    return [y[1], -K * y[0] - C * y[1]]

# Numerical solution using ode solver
K = k / m
C = c / m

sol = solve_ivp(lambda t, y: dy(t, y, K, C), [0, T], [x0, v0], t_eval=np.linspace(0, T, 200))

# Plot the numerical and analytic solutions
plt.figure()
plt.plot(sol.t, sol.y[0], '-o', label='Numerical Solution')
plt.plot(sol.t, analytic_solution(sol.t), label='Analytic Solution')
plt.xlabel('Time, t')
plt.ylabel('x(t)')
plt.legend()
plt.title(f'Numerical and Analytic Solutions (m={m}, c={c}, k={k})')
plt.grid()
plt.show()

# Plot the difference between solutions
plt.figure()
plt.plot(sol.t, sol.y[0] - analytic_solution(sol.t), '-o')
plt.xlabel('Time, t')
plt.ylabel('Difference x_h(t) - x(t)')
plt.title('Numerical Residual')
plt.grid()
plt.show()

# Generate simulated data
M = 100
global h, Sim_data
h = T / (M - 1)
t_vals = np.linspace(0, T, M)
Sim_data = analytic_solution(t_vals)

# Define least squares objective function
def myfun(q):
    sol = solve_ivp(lambda t, y: dy(t, y, q[0], q[1]), [0, T], [x0, v0], t_eval=t_vals)
    return sol.y[0] - Sim_data

# Compute LSOF
def LSOF(Xh, X):
    return 0.5 * np.sum((Xh - X)**2)

# Optimization using least squares
initial_guesses = [
    [K, C], [K * 1.1, C * 0.89], [K * 0.9, C * 1.3], [K * 2.3, C * 4], [K * 5, C * 5], [K * 20, C * 12]
]
results = []
for guess in initial_guesses:
    res = least_squares(myfun, guess)
    results.append([*guess, 0.5 * np.sum(res.fun**2), *res.x, 0.5 * res.cost, res.nfev])

# Add noise and re-estimate parameters
noise_levels = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
res_list = []
original_data = Sim_data.copy()
for noise in noise_levels:
    Sim_data = original_data + noise * np.random.randn(M)
    res = least_squares(myfun, results[3][3:5])
    res_list.append([noise, 0.5 * np.sum(res.fun**2), *res.x, 0.5 * res.cost, res.nfev])

# Plot minimum LSOF vs noise level
plt.figure()
plt.plot(noise_levels, [r[4] for r in res_list], '-o')
plt.xlabel('Noise Level')
plt.ylabel('Minimum LSOF')
plt.title('The Minimum of the Objective Function Value')
plt.grid()
plt.show()

# Log-log plot
plt.figure()
plt.loglog(noise_levels[1:], [r[4] for r in res_list[1:]], '-o')
plt.xlabel('Noise Level')
plt.ylabel('Minimum LSOF')
plt.title('The Minimum of the Objective Function Value (Log-Log)')
plt.grid()
plt.show()

# Plot confidence intervals
plt.figure()
plt.errorbar(noise_levels, [r[2] for r in res_list], fmt='o')
plt.xlabel('Noise Level')
plt.ylabel('Estimated K')
plt.title('Confidence Interval for K')
plt.grid()
plt.show()

plt.figure()
plt.errorbar(noise_levels, [r[3] for r in res_list], fmt='o')
plt.xlabel('Noise Level')
plt.ylabel('Estimated C')
plt.title('Confidence Interval for C')
plt.grid()
plt.show()
