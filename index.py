import numpy as np
import matplotlib.pyplot as plt

# Define the decay function
def decay(t, y, k):
    dydt = -k * y
    return dydt

# Define the Runge-Kutta 4 method
def RK4_step(fun, t, y, h, k):
    k1 = h * fun(t, y, k)
    k2 = h * fun(t + h / 2, y + k1 / 2, k)
    k3 = h * fun(t + h / 2, y + k2 / 2, k)
    k4 = h * fun(t + h, y + k3, k)
    y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_next

# Calculate error
def calculate_error(y_h, y_2h, p):
    error = abs(y_2h[-1] - y_h[-1]) / (2**p - 1)
    return error


# Define the simulation function
def simulation(fun, t_span, y0, h, k):
    t_eval = np.arange(t_span[0], t_span[1] + h, h)
    y_eval = np.zeros_like(t_eval)
    y_eval[0] = y0
    for i in range(len(t_eval) - 1):
        y_eval[i+1] = RK4_step(fun, t_eval[i], y_eval[i], h, k)
    return t_eval, y_eval

# Set parameters for simulation
h = 0.01
t_span = [0, 10]
k = 0.5

# Simulate the decay process with exponentially distributed initial values
y0_exp = np.random.exponential(1, 100)  # Generate 100 exponential initial values
y_eval_exp = np.zeros((len(y0_exp), len(np.arange(t_span[0], t_span[1] + h, h))))
y_eval_exp_error = np.zeros((len(y0_exp), len(np.arange(t_span[0], t_span[1] + h /2 , h / 2))))

errors = []

for i in range(len(y0_exp)):
  t_eval, y_eval_exp[i,:] = simulation(decay, t_span, y0_exp[i], h, k)
  _, y_eval_exp_error[i, :] = simulation(decay, t_span, y0_exp[i], h / 2, k)
  errors.append(calculate_error(y_eval_exp[i], y_eval_exp_error[i], 4))

# Plot the results
plt.figure(figsize=(10,6))
for i in range(len(y0_exp)):
    plt.plot(t_eval, y_eval_exp[i,:], 'b',errors[i], alpha=0.2)

plt.xlabel('Time')
plt.ylabel('Population (number of particles)')
plt.title('Decay Process with Exponentially Distributed Initial Values')

# Calculate and plot the mean and variance
mean = np.mean(y_eval_exp, axis=0)
variance = np.var(y_eval_exp, axis=0)

plt.figure(figsize=(10,6))
plt.plot( [i for i in range(1, 101)], errors, linestyle="",marker="o")

plt.figure(figsize=(10,6))
plt.plot(t_eval, mean, 'r', label='Mean')
plt.fill_between(t_eval, mean-variance, mean+variance, color='gray', alpha=0.2, label='Variance')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Mean and Variance of Decay Process with Exponentially Distributed Initial Values')

plt.show()
