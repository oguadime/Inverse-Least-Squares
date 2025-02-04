# Inverse Least Squares Problems 

This repository contains MATLAB scripts implementing numerical methods for solving second-order linear differential equations, estimating system parameters, and optimizing models using least squares and nonlinear methods. The primary focus is on solving and analyzing damped harmonic oscillators and related inverse problems.

## Description

This repository includes MATLAB scripts for numerical analysis:

- **Differential Equation Solvers**:
  - `MassSpringDashpot.m`: Models a damped harmonic oscillator governed by the equation:
    
    \[
    m\ddot{x} + c\dot{x} + kx = 0
    \]
    
    where \( m \) is mass, \( c \) is damping coefficient, and \( k \) is the stiffness parameter.
  - Solves the system using `ode45` and compares the numerical solution with an analytical solution.
  - Computes and visualizes solution errors.

- **Optimization and Least Squares Methods**:
  - Uses `lsqnonlin` to estimate unknown system parameters \( k \) and \( c \) from simulated noisy data.
  - Computes the least squares objective function (LSOF) to evaluate model accuracy.
  - Tests parameter estimation robustness by adding noise to the simulated data and analyzing results.
  - Computes confidence intervals for estimated parameters using statistical methods.

## Usage

Each script is self-contained. To use the main solver, execute `lab1.m` in MATLAB.

Example:

```matlab
% Run the lab1 script to solve the system
MassSpringDashpot
```

For parameter estimation:

```matlab
% Run nonlinear least squares optimization with different initial guesses
lsqnonlin(@myfun, [initial_K, initial_C])
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

