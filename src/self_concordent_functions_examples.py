import numpy as np
import matplotlib.pyplot as plt

# Define the derivatives of the log function
def neg_log_function(x):
    return -np.log(x), 1/x**2,  -2/x**3

def neg_entropy_negative_log(x):
  return x*np.log(x)-np.log(x), (x+1)/x**2, -(x+2)/x**3


# Define the self-concordance condition function
def self_concordance_condition(hessian):
    return 2 * (hessian)**(3/2)

# Define the range for x values
x = np.linspace(0.1, 10, 400)  # Avoid zero to prevent division by zero

# Compute the function values for log
_, hess_0, jerk_0 = neg_log_function(x)
jerk_conditions_0 = np.abs(jerk_0)
hess_conditions_0 = self_concordance_condition(hess_0)

_, hess_1, jerk_1 = neg_entropy_negative_log(x)
jerk_conditions_1 = np.abs(jerk_1)
hess_conditions_1 = self_concordance_condition(hess_1)

a, b = 5, 113
x_affine = a * x + b
_, hess_2, jerk_2 = neg_log_function(x_affine)
jerk_conditions_2 = np.abs(jerk_2)
hess_conditions_2 = self_concordance_condition(hess_2)

_, hess_3, jerk_3 = neg_entropy_negative_log(x_affine)
jerk_conditions_3 = np.abs(jerk_3)
hess_conditions_3 = self_concordance_condition(hess_3)


# Create the plot
plt.figure(figsize=(14, 10))

# Logarithm function plots
plt.subplot(2, 2, 1)
plt.plot(x, jerk_conditions_0, label="|f'''(x)|")
plt.plot(x, hess_conditions_0, label="2(f''(x))^(3/2)")
plt.title('Self-Concordance Condition for -log(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.subplot(2, 2, 2)
plt.plot(x, jerk_conditions_1, label="|f'''(x)|")
plt.plot(x, hess_conditions_1, label="2(f''(x))^(3/2)")
plt.title('Self-Concordance Condition for xlog(x)-log(x)')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.yscale('log')  

plt.subplot(2, 2, 3)
plt.plot(x, jerk_conditions_2, label=f"|({a}^3)f'''(x)|")
plt.plot(x, hess_conditions_2, label=f"2(({a}^2)f''(x))^(3/2)")
plt.title('Self-Concordance Condition + Affine y={a}x+{b}, -log(y)')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.subplot(2, 2, 4)
plt.plot(x, jerk_conditions_3, label=f"|({a}^3)f'''(x)|")
plt.plot(x, hess_conditions_3, label=f"2(({a}^2)f''(x))^(3/2)")
plt.title(f'Self-Concordance Condition + Affine y={a}x+{b}, ylog(y)-log(y)')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.yscale('log')

# Show the plots
plt.tight_layout()
plt.show()